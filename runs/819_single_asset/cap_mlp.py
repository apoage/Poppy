#!/usr/bin/env python3
"""TASK_819 capacity ladder — ARM 3: MLP (deep tabular, fully iterated).

A small 2-3 layer MLP on the RICH feature set, trained to CONVERGENCE with val-IC
early-stopping. The full loss/val-IC-vs-EPOCH curve is recorded to PROVE the model
is not under-iterated (the operator's worry). VAL-selected over a small width/depth
sweep, frozen TEST is the verdict, TRAIN-vs-TEST IC gap = overfit signature.

Runs on a free server GPU if CUDA is visible (set CUDA_VISIBLE_DEVICES upstream),
else CPU — the panel is tiny either way.

Outputs: _cap_mlp_results.json  (consumed by cap_aggregate.py).
"""
import json
import time
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

import cap_common as C

_T = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
LOG = []


def P(s=""):
    print(s, flush=True)
    LOG.append(s)


class MLP(nn.Module):
    def __init__(self, n_in, hidden=(128, 64), drop=0.1):
        super().__init__()
        layers = []
        d = n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(drop)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(s, hidden, lr, wd, drop, max_epochs, patience, batch=512,
              seed=SEED, shuffle_labels=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr = torch.tensor(s["Xtr"], dtype=torch.float32, device=DEVICE)
    ytr = torch.tensor(s["ytr"], dtype=torch.float32, device=DEVICE)
    if shuffle_labels:
        ytr = ytr[torch.randperm(ytr.shape[0])]
    Xva = torch.tensor(s["Xva"], dtype=torch.float32, device=DEVICE)
    yva_np = s["yva"]
    model = MLP(s["F"], hidden, drop).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.MSELoss()
    n = Xtr.shape[0]
    best_ic, best_state, best_ep = -1e9, None, -1
    curve = []
    ep = 0
    for ep in range(max_epochs):
        model.train()
        idx = torch.randperm(n, device=DEVICE)
        tot = 0.0
        for b in range(0, n, batch):
            bi = idx[b:b + batch]
            opt.zero_grad()
            loss = lossf(model(Xtr[bi]), ytr[bi])
            loss.backward()
            opt.step()
            tot += loss.item() * bi.shape[0]
        tr_mse = tot / n
        model.eval()
        with torch.no_grad():
            pv = model(Xva).cpu().numpy()
        v_ic = C.rank_ic(pv, yva_np)
        v_mse = float(np.mean((pv - yva_np) ** 2))
        curve.append((ep, tr_mse, v_mse, v_ic))
        if v_ic > best_ic:
            best_ic, best_ep = v_ic, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep - best_ep >= patience:
            break
    model.load_state_dict(best_state)
    converged = (ep - best_ep >= patience)   # early-stop fired -> plateaued
    return model, curve, best_ic, best_ep, converged


def predict(model, X):
    model.eval()
    with torch.no_grad():
        out = []
        Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        for b in range(0, Xt.shape[0], 8192):
            out.append(model(Xt[b:b + 8192]).cpu().numpy())
    return np.concatenate(out)


def main():
    t0 = time.time()
    close, ts, logc, r = C.load_daily()
    Td, A = logc.shape
    mu, sd = C.per_asset_stats(r, 1, C.VAL_START_DAY)
    P(f"[cap_mlp] device={DEVICE} torch={torch.__version__} "
      f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','<unset>')}")
    P(f"[cap_mlp] daily panel Td={Td} A={A}  train<{C.VAL_START_DAY} val<{C.TEST_START_DAY} test>={C.TEST_START_DAY}")

    SWEEP = [((128, 64), 1e-3, 1e-4, 0.1),
             ((256, 128), 1e-3, 1e-4, 0.2),
             ((64, 32), 2e-3, 1e-5, 0.1),
             ((128, 64, 32), 1e-3, 1e-4, 0.2)]

    results = {"model": "mlp", "device": DEVICE}
    for H in (1, 5):
        P(f"\n{'='*78}\n### HORIZON H={H}d\n{'='*78}")
        feat = C.build_features(logc, r, mu, sd, H, rich=True)
        s = C.split_xy(feat)
        P(f"[H{H}] rich features={s['F']}  samples train={s['Xtr'].shape[0]} val={s['Xva'].shape[0]} test={s['Xte'].shape[0]}")

        best = None
        P(f"[H{H}] MLP VAL-IC sweep (max_epochs=600 patience=60):")
        for (hid, lr, wd, drop) in SWEEP:
            model, curve, vic, vep, conv = train_mlp(
                s, hid, lr, wd, drop, max_epochs=600, patience=60, seed=SEED)
            P(f"    {str(hid):16s} lr{lr:.0e} wd{wd:.0e} drop{drop}: "
              f"best_val_IC={vic:+.4f} @ep{vep} epochs={curve[-1][0]+1} converged={conv}")
            if best is None or vic > best["vic"]:
                best = dict(model=model, curve=curve, vic=vic, vep=vep, conv=conv,
                            cfg=(hid, lr, wd, drop))
        hid, lr, wd, drop = best["cfg"]
        P(f"[H{H}] WINNER: {hid} lr{lr:.0e} wd{wd:.0e} drop{drop}  val_IC={best['vic']:+.4f}  converged={best['conv']}")
        cv = best["curve"]
        P(f"[H{H}] curve(ep,tr_mse,va_mse,va_IC) head/mid/best/tail:")
        for e in [cv[0], cv[len(cv) // 2], cv[best["vep"]], cv[-1]]:
            P(f"      ep{e[0]:3d}  tr_mse={e[1]:.4f}  va_mse={e[2]:.4f}  va_IC={e[3]:+.4f}")

        m = best["model"]
        pred_te = predict(m, s["Xte"])
        pred_tr = predict(m, s["Xtr"])
        ev, r2, regs = C.eval_model_from_flat(pred_te, feat, ts, H)
        ic_train = C.rank_ic(pred_tr, s["ytr"])
        gap = ic_train - ev["ic_pooled"]
        P(f"\n[H{H}] MLP FROZEN-TEST: OOS IC pooled={ev['ic_pooled']:+.4f}  BTC={ev['ic_btc']:+.4f}  "
          f"R2={r2:+.4f}  train_IC={ic_train:+.4f}  gap={gap:+.4f}")
        for cbp in C.COSTS_BP:
            sh = ev["sharpe"][cbp]
            P(f"      net@{int(cbp):2d}bp  pooled={sh['pooled']:+.2f}  BTC={sh['btc']:+.2f}")
        P(f"[H{H}] per-regime OOS:")
        for nm, v in regs.items():
            if v is None:
                P(f"      {nm}: <6 rows"); continue
            P(f"      {nm}: IC={v['ic_pooled']:+.4f}  net@8 pooled={v['sharpe'][8.0]['pooled']:+.2f}  BTC={v['sharpe'][8.0]['btc']:+.2f}")

        results[H] = dict(
            oos_ic=ev["ic_pooled"], oos_ic_btc=ev["ic_btc"], oos_r2=float(r2),
            ic_train=float(ic_train), gap=float(gap), converged=bool(best["conv"]),
            val_ic=float(best["vic"]), winner=str(best["cfg"]), nfeat=int(s["F"]),
            sharpe=C.sharpe_row(ev["sharpe"], ev.get("ci")),
            curve=[(int(a), float(b), float(c), float(d)) for a, b, c, d in cv],
            regimes={k: (None if v is None else dict(ic=v["ic_pooled"],
                     net8_pooled=v["sharpe"][8.0]["pooled"], net8_btc=v["sharpe"][8.0]["btc"]))
                     for k, v in regs.items()})

    # ---- shuffled-label leak detector (H=1) ----
    P(f"\n{'='*78}\n### SELF-TEST shuffled-label MLP (H=1, leak detector)\n{'='*78}")
    feat1 = C.build_features(logc, r, mu, sd, 1, rich=True)
    s1 = C.split_xy(feat1)
    sm, sc, svic, svep, sconv = train_mlp(s1, (128, 64), 1e-3, 1e-4, 0.1,
                                          max_epochs=200, patience=40, shuffle_labels=True)
    ev_sh, _, _ = C.eval_model_from_flat(predict(sm, s1["Xte"]), feat1, ts, 1)
    leak_ok = abs(ev_sh["ic_pooled"]) < 0.02
    P(f"[selftest] shuffled-label MLP OOS IC={ev_sh['ic_pooled']:+.4f}  (leak-free<0.02: {leak_ok})")
    results["selftest_shuffle_ic"] = float(ev_sh["ic_pooled"])
    results["selftest_leak_ok"] = bool(leak_ok)

    runtime = time.time() - t0
    P(f"\n[cap_mlp] runtime {runtime:.1f}s")
    results["runtime"] = runtime
    json.dump(results, open(_T / "_cap_mlp_results.json", "w"), indent=2, default=float)
    (_T / "_cap_mlp_log.txt").write_text("\n".join(LOG))
    print("[cap_mlp] wrote _cap_mlp_results.json", flush=True)


if __name__ == "__main__":
    main()
