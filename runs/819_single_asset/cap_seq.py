#!/usr/bin/env python3
"""TASK_819 capacity ladder — ARM 4: GRU/LSTM SEQUENCE MODEL.

Feeds a small GRU/LSTM the RAW sequence of the last N=30 daily log-returns per asset
(NOT hand-crafted features — the temporal model's home turf) and asks whether it beats,
OUT OF SAMPLE, BOTH the linear ridge (hand-crafted features, IC≈0.03) and the pure TSMOM
trend rule (sign of 30-day return). VAL-selected over a small architecture sweep, trained
to convergence with val-IC early-stopping; the loss/val-IC-vs-EPOCH curve is recorded to
prove it is not under-iterated. Runs on a free server GPU when CUDA is visible.

Shared data/feature/eval/baseline utilities come from cap_common (single source of truth).
Outputs: _cap_seq_results.json  (consumed by cap_aggregate.py) + cap_seq.md.
"""
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

import cap_common as C

_T = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SEQ = 30
SEED = 0
LOG = []


def P(s=""):
    print(s, flush=True)
    LOG.append(s)


# ----------------------------------------------------------------------------- sequences
def build_sequences(logc, r, mu, sd, H):
    """For each decision day t and asset a: X = clip((r[t-N..t-1]-mu)/sd, ±Z) (N,1);
    raw forward target fwd = logc[t+H]-logc[t]; normalized yn = clip(fwd/(sd*sqrt(H)),±Z)."""
    Td, A = logc.shape
    rn = np.clip((r - mu[None, :]) / sd[None, :], -C.CLIP_Z, C.CLIP_Z)
    t_hi = Td - 1 - H
    days = np.arange(C.MIN_HIST, t_hi + 1)
    fwd = logc[days + H] - logc[days]
    yn = np.clip(fwd / (sd * np.sqrt(H))[None, :], -C.CLIP_Z, C.CLIP_Z)
    nd = days.shape[0]
    Xseq = np.empty((nd, A, N_SEQ, 1), dtype=np.float32)
    for k, t in enumerate(days):
        Xseq[k, :, :, 0] = rn[t - N_SEQ:t, :].T
    tr = days < C.VAL_START_DAY
    va = (days >= C.VAL_START_DAY) & (days < C.TEST_START_DAY)
    te = days >= C.TEST_START_DAY

    def flat(mask):
        d_idx = np.where(mask)[0]
        return dict(X=Xseq[d_idx].reshape(-1, N_SEQ, 1), Y=yn[d_idx].reshape(-1))
    return dict(train=flat(tr), val=flat(va), test=flat(te), days=days, fwd=fwd, yn=yn,
                A=A, Xseq=Xseq, te_mask=te, tr_mask=tr)


# ----------------------------------------------------------------------------- model
class SeqNet(nn.Module):
    def __init__(self, kind="gru", hidden=64, layers=2, drop=0.1):
        super().__init__()
        rnn = nn.GRU if kind == "gru" else nn.LSTM
        self.rnn = rnn(input_size=1, hidden_size=hidden, num_layers=layers,
                       batch_first=True, dropout=drop if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                  nn.Dropout(drop), nn.Linear(hidden, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def train_seq(data, kind, hidden, layers, lr, wd, max_epochs, patience,
              batch=256, seed=SEED, shuffle_labels=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr, va = data["train"], data["val"]
    Xtr = torch.tensor(tr["X"], device=DEVICE)
    Ytr = torch.tensor(tr["Y"], dtype=torch.float32, device=DEVICE)
    if shuffle_labels:
        Ytr = Ytr[torch.randperm(Ytr.shape[0], device=DEVICE)]
    Xva = torch.tensor(va["X"], device=DEVICE)
    yva = va["Y"]
    model = SeqNet(kind, hidden, layers).to(DEVICE)
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
            loss = lossf(model(Xtr[bi]), Ytr[bi])
            loss.backward()
            opt.step()
            tot += loss.item() * bi.shape[0]
        tr_mse = tot / n
        model.eval()
        with torch.no_grad():
            pv = model(Xva).cpu().numpy()
        v_ic = C.rank_ic(pv, yva)
        v_mse = float(np.mean((pv - yva) ** 2))
        curve.append((ep, tr_mse, v_mse, v_ic))
        if v_ic > best_ic:
            best_ic, best_ep = v_ic, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep - best_ep >= patience:
            break
    model.load_state_dict(best_state)
    converged = (ep - best_ep >= patience)
    return model, curve, best_ic, best_ep, converged


def predict_grid(model, Xseq, d_idx):
    X = Xseq[d_idx].reshape(-1, N_SEQ, 1)
    model.eval()
    out = []
    with torch.no_grad():
        for b in range(0, X.shape[0], 4096):
            xb = torch.tensor(X[b:b + 4096], device=DEVICE)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


# ----------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    close, ts, logc, r = C.load_daily()
    Td, A = logc.shape
    mu, sd = C.per_asset_stats(r, 1, C.VAL_START_DAY)
    P(f"[cap_seq] device={DEVICE} torch={torch.__version__} "
      f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','<unset>')}")
    P(f"[cap_seq] daily panel Td={Td} A={A}  train<{C.VAL_START_DAY} val<{C.TEST_START_DAY} test>={C.TEST_START_DAY}")

    sweep = [("gru", 64, 2, 1e-3, 1e-5),
             ("gru", 32, 1, 1e-3, 1e-5),
             ("gru", 96, 2, 5e-4, 1e-5),
             ("lstm", 64, 2, 1e-3, 1e-5)]

    results = {"model": "gru", "device": DEVICE}
    for H in (1, 5):
        P(f"\n{'='*78}\n### HORIZON H={H}d\n{'='*78}")
        data = build_sequences(logc, r, mu, sd, H)
        P(f"[H{H}] samples train={data['train']['X'].shape[0]} val={data['val']['X'].shape[0]} test={data['test']['X'].shape[0]}")

        best = None
        P(f"[H{H}] GRU/LSTM val-IC sweep (max_epochs=600 patience=60):")
        for (kind, hid, lay, lr, wd) in sweep:
            model, curve, vic, vep, conv = train_seq(
                data, kind, hid, lay, lr, wd, max_epochs=600, patience=60, seed=SEED)
            P(f"    {kind:4s} h{hid} L{lay} lr{lr:.0e}: best_val_IC={vic:+.4f} @ep{vep} "
              f"epochs={curve[-1][0]+1} converged={conv}")
            if best is None or vic > best["vic"]:
                best = dict(model=model, curve=curve, vic=vic, vep=vep, conv=conv,
                            cfg=(kind, hid, lay, lr, wd))
        kind, hid, lay, lr, wd = best["cfg"]
        P(f"[H{H}] WINNER: {kind} h{hid} L{lay} lr{lr:.0e}  val_IC={best['vic']:+.4f}  converged={best['conv']}")
        cv = best["curve"]
        P(f"[H{H}] curve(ep,tr_mse,va_mse,va_IC) head/mid/best/tail:")
        for e in [cv[0], cv[len(cv) // 2], cv[best["vep"]], cv[-1]]:
            P(f"      ep{e[0]:3d}  tr_mse={e[1]:.4f}  va_mse={e[2]:.4f}  va_IC={e[3]:+.4f}")

        # frozen TEST forecast (dense matrix) + train IC + R2
        te_idx = np.where(data["te_mask"])[0]
        tr_idx = np.where(data["tr_mask"])[0]
        g_pred = predict_grid(best["model"], data["Xseq"], te_idx).reshape(len(te_idx), A)
        g_fwd = data["fwd"][te_idx]
        g_days = data["days"][te_idx]
        gp_tr = predict_grid(best["model"], data["Xseq"], tr_idx)
        ic_train = C.rank_ic(gp_tr, data["yn"][data["tr_mask"]].reshape(-1))
        yn_te = data["yn"][data["te_mask"]].reshape(-1)
        g_r2 = C.oos_r2(g_pred.reshape(-1), yn_te)

        g_eval = C.eval_signal(g_pred, g_fwd, g_days, ts, H, "tanh", ci_costs=(8.0, 21.0))
        gap = ic_train - g_eval["ic_pooled"]
        regs = {nm: C.eval_signal(g_pred, g_fwd, g_days, ts, H, "tanh", regime=win)
                for nm, win in C.regime_epochs().items()}

        P(f"\n[H{H}] GRU FROZEN-TEST: OOS IC pooled={g_eval['ic_pooled']:+.4f}  BTC={g_eval['ic_btc']:+.4f}  "
          f"R2={g_r2:+.4f}  train_IC={ic_train:+.4f}  gap={gap:+.4f}")
        for cbp in C.COSTS_BP:
            s = g_eval["sharpe"][cbp]
            P(f"      net@{int(cbp):2d}bp  pooled={s['pooled']:+.2f}  BTC={s['btc']:+.2f}")
        P(f"[H{H}] per-regime OOS:")
        for nm, v in regs.items():
            if v is None:
                P(f"      {nm}: <6 rows"); continue
            P(f"      {nm}: IC={v['ic_pooled']:+.4f}  net@8 pooled={v['sharpe'][8.0]['pooled']:+.2f}  BTC={v['sharpe'][8.0]['btc']:+.2f}")

        # references (shared, for this script's standalone verdict)
        feat = C.build_features(logc, r, mu, sd, H, rich=False)
        rd = C.ridge_baseline(feat, ts, H)
        tm = C.tsmom_baseline(logc, ts, H)
        g8 = g_eval["sharpe"][8.0]["pooled"]
        r8 = rd["sharpe"][8.0]["pooled"]
        t8 = tm["sharpe"][8.0]["pooled"]
        beats = bool((g_eval["ic_pooled"] > rd["oos_ic"] + 0.005) and
                     (g_eval["ic_pooled"] > tm["oos_ic"] + 0.005) and
                     (g8 > r8 + 0.1) and (g8 > t8 + 0.1))
        P(f"\n[H{H}] vs refs: GRU IC {g_eval['ic_pooled']:+.4f} | ridge {rd['oos_ic']:+.4f} | TSMOM {tm['oos_ic']:+.4f}")
        P(f"       net@8 pooled: GRU {g8:+.2f} | ridge {r8:+.2f} | TSMOM {t8:+.2f}")
        P(f"[H{H}] VERDICT GRU beats ridge&trend OOS (IC AND net@8)? -> {beats}   gap={gap:+.4f}")

        results[H] = dict(
            oos_ic=g_eval["ic_pooled"], oos_ic_btc=g_eval["ic_btc"], oos_r2=float(g_r2),
            ic_train=float(ic_train), gap=float(gap), converged=bool(best["conv"]),
            val_ic=float(best["vic"]), winner=str(best["cfg"]), beats=beats,
            sharpe=C.sharpe_row(g_eval["sharpe"], g_eval.get("ci")),
            curve=[(int(a), float(b), float(c), float(d)) for a, b, c, d in cv],
            regimes={k: (None if v is None else dict(ic=v["ic_pooled"],
                     net8_pooled=v["sharpe"][8.0]["pooled"], net8_btc=v["sharpe"][8.0]["btc"]))
                     for k, v in regs.items()})

    # ---- self-test shuffled-label control (H=1) ----
    P(f"\n{'='*78}\n### SELF-TEST shuffled-label GRU (H=1, leak detector)\n{'='*78}")
    data1 = build_sequences(logc, r, mu, sd, 1)
    sm, sc, svic, svep, sconv = train_seq(data1, "gru", 64, 2, 1e-3, 1e-5,
                                          max_epochs=200, patience=40, shuffle_labels=True)
    te_idx = np.where(data1["te_mask"])[0]
    sp = predict_grid(sm, data1["Xseq"], te_idx).reshape(len(te_idx), data1["A"])
    se = C.eval_signal(sp, data1["fwd"][te_idx], data1["days"][te_idx], ts, 1, "tanh")
    leak_ok = abs(se["ic_pooled"]) < 0.02
    P(f"[selftest] shuffled-label GRU OOS IC={se['ic_pooled']:+.4f}  (leak-free<0.02: {leak_ok})")
    results["selftest_shuffle_ic"] = float(se["ic_pooled"])
    results["selftest_leak_ok"] = bool(leak_ok)

    runtime = time.time() - t0
    P(f"\n[cap_seq] runtime {runtime:.1f}s")
    results["runtime"] = runtime
    json.dump(results, open(_T / "_cap_seq_results.json", "w"), indent=2, default=float)
    (_T / "_cap_seq_log.txt").write_text("\n".join(LOG))
    print("[cap_seq] wrote _cap_seq_results.json", flush=True)


if __name__ == "__main__":
    main()
