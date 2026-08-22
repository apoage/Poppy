#!/usr/bin/env python3
"""TASK_819 capacity ladder — ARM 2: GRADIENT-BOOSTED TREES (nonlinear tabular).

Gives a nonlinear, interaction-aware tabular model (sklearn HistGradientBoosting;
xgboost absent on server) every fair chance to find structure the linear ridge
missed. HYPERPARAMETERS TUNED ON the temporal VAL set (small grid); the
val-IC-vs-boosting-iteration curve is reported via staged_predict to PROVE the model
is iterated to its plateau (the operator's under-iteration worry — max_iter up to 1500).
Built-in early-stopping is OFF so the FULL curve is observed; the best stage is chosen
on MY temporal 2024-H1 VAL (not a random internal split). Rich feature set (cap_common).

Outputs: _cap_gbm_results.json  (consumed by cap_aggregate.py). CPU only.
"""
import json
import time
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import cap_common as C

_T = Path(__file__).resolve().parent
LOG = []


def P(s=""):
    print(s, flush=True)
    LOG.append(s)


def staged_ic(model, X, y):
    """rank-IC at every boosting iteration (cumulative). One efficient pass."""
    return np.array([C.rank_ic(pv, y) for pv in model.staged_predict(X)])


def staged_pred_at(model, X, stage):
    pv = None
    for i, p in enumerate(model.staged_predict(X)):
        pv = p
        if i == stage:
            break
    return pv


def main():
    t0 = time.time()
    close, ts, logc, r = C.load_daily()
    Td, A = logc.shape
    mu, sd = C.per_asset_stats(r, 1, C.VAL_START_DAY)
    P(f"[cap_gbm] daily panel Td={Td} A={A}  train<{C.VAL_START_DAY} val<{C.TEST_START_DAY} test>={C.TEST_START_DAY}")

    GRID = [dict(max_iter=1500, max_depth=2, learning_rate=0.02, l2_regularization=1.0, min_samples_leaf=200),
            dict(max_iter=1500, max_depth=3, learning_rate=0.02, l2_regularization=1.0, min_samples_leaf=200),
            dict(max_iter=1500, max_depth=3, learning_rate=0.01, l2_regularization=5.0, min_samples_leaf=400),
            dict(max_iter=800, max_depth=4, learning_rate=0.03, l2_regularization=2.0, min_samples_leaf=100)]

    results = {"model": "gbm"}
    for H in (1, 5):
        P(f"\n{'='*78}\n### HORIZON H={H}d\n{'='*78}")
        feat = C.build_features(logc, r, mu, sd, H, rich=True)
        s = C.split_xy(feat)
        P(f"[H{H}] rich features={s['F']}  samples train={s['Xtr'].shape[0]} val={s['Xva'].shape[0]} test={s['Xte'].shape[0]}")

        best = None
        P(f"[H{H}] GBM VAL tuning (HistGBR, early_stopping OFF, full staged curve):")
        for cfg in GRID:
            m = HistGradientBoostingRegressor(random_state=0, early_stopping=False, **cfg)
            m.fit(s["Xtr"], s["ytr"])
            ics = staged_ic(m, s["Xva"], s["yva"])
            best_stage = int(np.argmax(ics))
            v_ic = float(ics[best_stage])
            P(f"    md{cfg['max_depth']} lr{cfg['learning_rate']} it{cfg['max_iter']} "
              f"l2_{cfg['l2_regularization']} msl{cfg['min_samples_leaf']}: "
              f"best_val_IC={v_ic:+.4f} @iter{best_stage}/{m.n_iter_}  final_val_IC={ics[-1]:+.4f}")
            if best is None or v_ic > best["v_ic"]:
                stride = max(1, len(ics) // 40)
                curve = [(i, float(ics[i])) for i in range(0, len(ics), stride)]
                if curve[-1][0] != len(ics) - 1:
                    curve.append((len(ics) - 1, float(ics[-1])))
                best = dict(model=m, cfg=cfg, v_ic=v_ic, best_stage=best_stage,
                            curve=curve, n_iter=int(m.n_iter_), final_ic=float(ics[-1]))

        m, bs, cfg = best["model"], best["best_stage"], best["cfg"]
        P(f"[H{H}] WINNER: md{cfg['max_depth']} lr{cfg['learning_rate']} it{cfg['max_iter']} "
          f"l2_{cfg['l2_regularization']}  val_IC={best['v_ic']:+.4f} @iter{bs}/{best['n_iter']}")
        cv = best["curve"]
        P(f"[H{H}] VAL-IC curve (iter, val_IC) head/mid/tail:")
        for e in [cv[0], cv[len(cv) // 2], cv[-1]]:
            P(f"      iter{e[0]:4d}  val_IC={e[1]:+.4f}")

        pred_te = staged_pred_at(m, s["Xte"], bs)
        pred_tr = staged_pred_at(m, s["Xtr"], bs)
        ev, r2, regs = C.eval_model_from_flat(pred_te, feat, ts, H)
        ic_train = C.rank_ic(pred_tr, s["ytr"])
        gap = ic_train - ev["ic_pooled"]
        # converged = VAL-IC peaked BEFORE the last iteration (plateau, not still-climbing)
        converged = bs < (best["n_iter"] - 1)

        P(f"\n[H{H}] GBM FROZEN-TEST: OOS IC pooled={ev['ic_pooled']:+.4f}  BTC={ev['ic_btc']:+.4f}  "
          f"R2={r2:+.4f}  train_IC={ic_train:+.4f}  gap={gap:+.4f}  converged(peak<last)={converged}")
        for cbp in C.COSTS_BP:
            sh = ev["sharpe"][cbp]
            ci = ev["ci"].get(cbp)
            P(f"      net@{int(cbp):2d}bp  pooled={sh['pooled']:+.2f}  BTC={sh['btc']:+.2f}"
              + (f"  CI[{ci[0]:+.2f},{ci[1]:+.2f}]" if ci else ""))
        P(f"[H{H}] per-regime OOS:")
        for nm, v in regs.items():
            if v is None:
                P(f"      {nm}: <6 rows"); continue
            P(f"      {nm}: IC={v['ic_pooled']:+.4f}  net@8 pooled={v['sharpe'][8.0]['pooled']:+.2f}  BTC={v['sharpe'][8.0]['btc']:+.2f}")

        results[H] = dict(
            oos_ic=ev["ic_pooled"], oos_ic_btc=ev["ic_btc"], oos_r2=float(r2),
            ic_train=float(ic_train), gap=float(gap), converged=bool(converged),
            val_ic=best["v_ic"], best_stage=bs, n_iter=best["n_iter"], final_val_ic=best["final_ic"],
            cfg=cfg, nfeat=int(s["F"]), sharpe=C.sharpe_row(ev["sharpe"], ev.get("ci")),
            curve=[(int(a), float(b)) for a, b in cv],
            regimes={k: (None if v is None else dict(ic=v["ic_pooled"],
                     net8_pooled=v["sharpe"][8.0]["pooled"], net8_btc=v["sharpe"][8.0]["btc"]))
                     for k, v in regs.items()})

    runtime = time.time() - t0
    P(f"\n[cap_gbm] runtime {runtime:.1f}s")
    results["runtime"] = runtime
    json.dump(results, open(_T / "_cap_gbm_results.json", "w"), indent=2, default=float)
    (_T / "_cap_gbm_log.txt").write_text("\n".join(LOG))
    print("[cap_gbm] wrote _cap_gbm_results.json", flush=True)


if __name__ == "__main__":
    main()
