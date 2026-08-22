#!/usr/bin/env python3
"""TASK_819 capacity ladder — ARM 1: RIDGE-RICH (linear capacity control).

Asks whether GIVING THE LINEAR MODEL MORE CAPACITY (extra lags, vol-of-vol,
nonlinear trend transforms, pairwise interactions) beats the plain 14-feature
ridge (IC≈0.03). This is the linear ceiling: if even a rich linear model can't
move OOS IC, the bottleneck is not linear capacity.

PLUS the leak detector: shuffled-label ridge-rich -> OOS IC must be ≈0.

Also emits the canonical STANDARD-ridge + TSMOM references that every arm shares,
so cap_aggregate can cross-check they are identical across the four scripts.

Outputs: _cap_ridge_results.json  (consumed by cap_aggregate.py). CPU only.
"""
import json
import time
from pathlib import Path
import numpy as np

import cap_common as C

_T = Path(__file__).resolve().parent
LOG = []


def P(s=""):
    print(s, flush=True)
    LOG.append(s)


def main():
    t0 = time.time()
    close, ts, logc, r = C.load_daily()
    Td, A = logc.shape
    mu, sd = C.per_asset_stats(r, 1, C.VAL_START_DAY)
    P(f"[cap_ridge] daily panel Td={Td} A={A}  train<{C.VAL_START_DAY} val<{C.TEST_START_DAY} test>={C.TEST_START_DAY}")

    results = {"model": "ridge_rich"}
    for H in (1, 5):
        P(f"\n{'='*78}\n### HORIZON H={H}d\n{'='*78}")
        feat_std = C.build_features(logc, r, mu, sd, H, rich=False)
        feat_rich = C.build_features(logc, r, mu, sd, H, rich=True)
        P(f"[H{H}] features: standard={len(feat_std['names'])}  rich={len(feat_rich['names'])}")
        P(f"[H{H}] samples train={int(feat_std['tr'].sum())*A} val={int(feat_std['va'].sum())*A} test={int(feat_std['te'].sum())*A}")

        rd = C.ridge_baseline(feat_std, ts, H)
        rr = C.ridge_baseline(feat_rich, ts, H)
        tm = C.tsmom_baseline(logc, ts, H)

        for tag, d in [("STANDARD-ridge", rd), ("RICH-ridge", rr)]:
            P(f"\n[H{H}] {tag} (alpha={d['alpha']:g}, nfeat={d['nfeat']}):")
            P(f"      OOS IC pooled={d['oos_ic']:+.4f}  BTC={d['oos_ic_btc']:+.4f}  R2={d['oos_r2']:+.4f}  "
              f"train_IC={d['ic_train']:+.4f}  gap={d['gap']:+.4f}")
            for cbp in C.COSTS_BP:
                s = d["sharpe"][cbp]
                P(f"      net@{int(cbp):2d}bp  pooled={s['pooled']:+.2f}  BTC={s['btc']:+.2f}")
        P(f"\n[H{H}] TSMOM(sign 30d): OOS IC pooled={tm['oos_ic']:+.4f}  BTC={tm['oos_ic_btc']:+.4f}  "
          f"net@8bp pooled={tm['sharpe'][8.0]['pooled']:+.2f}")

        # ---- shuffled-label leak detector (rich features) ----
        s = C.split_xy(feat_rich)
        rng = np.random.RandomState(819)
        ysh = s["ytr"].copy()
        rng.shuffle(ysh)
        w = C.ridge_fit(s["Xtr"], ysh, 10.0)
        ev_sh, r2_sh, _ = C.eval_model_from_flat(s["Xte"] @ w, feat_rich, ts, H)
        leak_ok = abs(ev_sh["ic_pooled"]) < 0.02
        P(f"\n[H{H}] SELF-TEST shuffled-label ridge-rich: OOS IC={ev_sh['ic_pooled']:+.4f}  (leak-free<0.02: {leak_ok})")

        results[H] = dict(
            ridge_std=_pack(rd), ridge_rich=_pack(rr), tsmom=_pack_tsmom(tm),
            shuffle_ic=float(ev_sh["ic_pooled"]), leak_ok=bool(leak_ok))

    runtime = time.time() - t0
    P(f"\n[cap_ridge] runtime {runtime:.1f}s")
    results["runtime"] = runtime
    json.dump(results, open(_T / "_cap_ridge_results.json", "w"), indent=2, default=float)
    (_T / "_cap_ridge_log.txt").write_text("\n".join(LOG))
    print("[cap_ridge] wrote _cap_ridge_results.json", flush=True)


def _pack(d):
    return dict(name=d["name"], oos_ic=d["oos_ic"], oos_ic_btc=d["oos_ic_btc"],
                oos_r2=d["oos_r2"], ic_train=d["ic_train"], gap=d["gap"], alpha=d["alpha"],
                val_ic=d["val_ic"], nfeat=d["nfeat"], sharpe=C.sharpe_row(d["sharpe"], d.get("ci")),
                regimes={k: (None if v is None else dict(ic=v["ic_pooled"],
                         net8_pooled=v["sharpe"][8.0]["pooled"], net8_btc=v["sharpe"][8.0]["btc"]))
                         for k, v in d["regimes"].items()})


def _pack_tsmom(d):
    return dict(name=d["name"], oos_ic=d["oos_ic"], oos_ic_btc=d["oos_ic_btc"],
                sharpe=C.sharpe_row(d["sharpe"], d.get("ci")),
                regimes={k: (None if v is None else dict(ic=v["ic_pooled"],
                         net8_pooled=v["sharpe"][8.0]["pooled"], net8_btc=v["sharpe"][8.0]["btc"]))
                         for k, v in d["regimes"].items()})


if __name__ == "__main__":
    main()
