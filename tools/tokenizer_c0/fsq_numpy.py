"""
Pure-numpy FSQ tokenizer + manual-backprop MLP autoencoder.

WHY NUMPY: in the C0 sandbox JAX and torch execution are blocked (even on CPU);
only numpy/sklearn execute. To still deliver honest gate numbers we reimplement
the SAME design in numpy:
   RevIN per-window instance-norm -> MLP encoder -> FSQ -> MLP decoder.
FSQ (arXiv:2309.15505) with a straight-through estimator is collapse-proof; the
manual backprop below passes gradients through the round() via STE (identity).

Architecture (matches fsq.py):
   flat(window*F) -> Dense(H) gelu -> Dense(H) gelu -> Dense(d)  [encoder]
   FSQ bound(tanh)+round_ste -> zhat
   zhat -> Dense(H) gelu -> Dense(H) gelu -> Dense(window*F)      [decoder]
Loss: MSE on normalized features. Adam optimizer, hand-rolled.
"""
import numpy as np


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def gelu_grad(x):
    # derivative of the tanh-approx gelu
    c = np.sqrt(2.0 / np.pi)
    u = c * (x + 0.044715 * x ** 3)
    t = np.tanh(u)
    du = c * (1.0 + 3 * 0.044715 * x ** 2)
    return 0.5 * (1 + t) + 0.5 * x * (1 - t ** 2) * du


class Dense:
    def __init__(self, nin, nout, rng):
        scale = np.sqrt(2.0 / nin)
        self.W = (rng.standard_normal((nin, nout)) * scale).astype(np.float64)
        self.b = np.zeros(nout, np.float64)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b); self.vb = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, gout):
        self.gW = self.x.T @ gout
        self.gb = gout.sum(0)
        return gout @ self.W.T

    def step(self, lr, t, b1=0.9, b2=0.999, eps=1e-8):
        for P, g, m, v in [("W", self.gW, "mW", "vW"), ("b", self.gb, "mb", "vb")]:
            par = getattr(self, P); mm = getattr(self, m); vv = getattr(self, v)
            mm[:] = b1 * mm + (1 - b1) * g
            vv[:] = b2 * vv + (1 - b2) * g * g
            mhat = mm / (1 - b1 ** t); vhat = vv / (1 - b2 ** t)
            par -= lr * mhat / (np.sqrt(vhat) + eps)


class FSQNumpy:
    def __init__(self, levels):
        self.levels = np.asarray(levels, np.float64)
        self.half = (self.levels - 1) / 2.0
        basis = np.concatenate([[1], np.cumprod(np.asarray(levels[:-1]))])
        self.basis = basis.astype(np.int64)
        self.codebook_size = int(np.prod(levels))

    def forward(self, z):
        # bound via tanh to (-half, half)
        self.tanh = np.tanh(z)
        zb = self.tanh * self.half
        zhat = np.round(zb)  # rounding; STE on backward (treat as identity)
        self.zb = zb
        return zhat

    def backward(self, gout):
        # STE: d round / d zb = 1 ; d zb / d z = half * (1 - tanh^2)
        return gout * self.half * (1 - self.tanh ** 2)

    def indices(self, zhat):
        idx = np.round(zhat + self.half).astype(np.int64)
        idx = np.clip(idx, 0, (self.levels.astype(np.int64) - 1))
        return (idx * self.basis).sum(1)


class TokenizerNumpy:
    def __init__(self, window, n_feat=5, hidden=128, fsq_levels=(8, 8, 8, 5, 5), seed=0):
        rng = np.random.default_rng(seed)
        self.window = window; self.n_feat = n_feat
        din = window * n_feat; d = len(fsq_levels)
        self.e1 = Dense(din, hidden, rng)
        self.e2 = Dense(hidden, hidden, rng)
        self.e3 = Dense(hidden, d, rng)
        self.fsq = FSQNumpy(fsq_levels)
        self.d1 = Dense(d, hidden, rng)
        self.d2 = Dense(hidden, hidden, rng)
        self.d3 = Dense(hidden, din, rng)
        self.layers = [self.e1, self.e2, self.e3, self.d1, self.d2, self.d3]
        self.t = 0

    def _norm(self, x):
        # x: [B, window, F]; RevIN per-window per-channel
        mean = x.mean(1, keepdims=True)
        std = x.std(1, keepdims=True) + 1e-5
        xn = (x - mean) / std
        return xn, mean, std

    def forward(self, x, train=True):
        B = x.shape[0]
        xn, mean, std = self._norm(x)
        flat = xn.reshape(B, -1)
        h1 = self.e1.forward(flat); a1 = gelu(h1)
        h2 = self.e2.forward(a1); a2 = gelu(h2)
        z = self.e3.forward(a2)
        zhat = self.fsq.forward(z)
        g1 = self.d1.forward(zhat); b1 = gelu(g1)
        g2 = self.d2.forward(b1); b2 = gelu(g2)
        recon_flat = self.d3.forward(b2)
        recon_n = recon_flat.reshape(B, self.window, self.n_feat)
        cache = dict(h1=h1, h2=h2, g1=g1, g2=g2, xn=xn, mean=mean, std=std,
                     recon_flat=recon_flat, B=B, z=z, zhat=zhat)
        self.cache = cache
        idx = self.fsq.indices(zhat)
        recon = recon_n * std + mean
        return dict(recon_n=recon_n, recon=recon, xn=xn, idx=idx,
                    mean=mean, std=std, zhat=zhat)

    def train_step(self, x, lr):
        self.t += 1
        out = self.forward(x, train=True)
        c = self.cache; B = c["B"]
        # loss = mse(recon_n, xn)
        target = c["xn"].reshape(B, -1)
        diff = (c["recon_flat"] - target)
        loss = float(np.mean(diff ** 2))
        gflat = (2.0 / diff.size) * diff
        # decoder back
        gb2 = self.d3.backward(gflat)
        gg2 = gb2 * gelu_grad(c["g2"])
        gb1 = self.d2.backward(gg2)
        gg1 = gb1 * gelu_grad(c["g1"])
        gzhat = self.d1.backward(gg1)
        # through FSQ (STE)
        gz = self.fsq.backward(gzhat)
        ga2 = self.e3.backward(gz)
        gh2 = ga2 * gelu_grad(c["h2"])
        ga1 = self.e2.backward(gh2)
        gh1 = ga1 * gelu_grad(c["h1"])
        _ = self.e1.backward(gh1)
        for L in self.layers:
            L.step(lr, self.t)
        return loss

    def encode(self, x):
        out = self.forward(x, train=False)
        return out["idx"], out["recon"]
