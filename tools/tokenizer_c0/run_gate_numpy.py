"""
C0 viability gate runner — NUMPY/SKLEARN backend (CPU only).

Identical gate logic to run_gate.py, but the FSQ tokenizer is the numpy
manual-backprop implementation (fsq_numpy.py) and the next-token sign probe is a
sklearn classifier on token-context features. Used because JAX/torch execution is
blocked in this sandbox (even on CPU); numpy+sklearn execute fine.

Three metrics:
  RQ0   directional acc of P(sign next return | token context), OOS, real vs
        returns-shuffled (phase-randomized) control. Purge+embargo chrono split.
  FID   reconstruction MSE on returns + relative MSE on de-normalized price path.
  HEALTH FSQ token usage + perplexity (collapse check).

Writes runs/789_token_c0/RESULT.md (+ .json).
"""
import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import data as D
from fsq_numpy import TokenizerNumpy
from sklearn.linear_model import LogisticRegression


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--tok-epochs", type=int, default=25)
    p.add_argument("--tok-bs", type=int, default=512)
    p.add_argument("--tok-lr", type=float, default=1e-3)
    p.add_argument("--context", type=int, default=4)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--embargo-frac", type=float, default=0.0)
    p.add_argument("--max-windows-per-asset", type=int, default=25000)
    p.add_argument("--fsq-levels", type=str, default="8,8,8,5,5")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str,
                   default="/home/lukas/Trader/.claude/worktrees/agent-ab33144138e003324/runs/789_token_c0/RESULT.md")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def build_dataset(window, horizon, max_per_asset, shuffle_control, seed):
    rng = np.random.default_rng(seed + (777 if shuffle_control else 0))
    allX, allC, allR, allE, allA, allTs = [], [], [], [], [], []
    counts = {}
    for aid, name in D.ASSETS:
        ts, o, h, l, c, valid = D.load_hourly_ohlc(aid)
        feats, close, fvalid = D.make_features(o, h, l, c, valid)
        if shuffle_control:
            feats = D.phase_shuffle_returns(feats, fvalid, rng)
        X, Cw, R, E = D.make_windows(feats, close, fvalid, window, horizon)
        if X.shape[0] == 0:
            counts[name] = 0; continue
        if X.shape[0] > max_per_asset:
            X, Cw, R, E = X[-max_per_asset:], Cw[-max_per_asset:], R[-max_per_asset:], E[-max_per_asset:]
        end_ts = ts[1:][E]
        allX.append(X); allC.append(Cw); allR.append(R)
        allE.append(E); allA.append(np.full(len(R), aid)); allTs.append(end_ts)
        counts[name] = int(X.shape[0])
    return dict(X=np.concatenate(allX), C=np.concatenate(allC), R=np.concatenate(allR),
                E=np.concatenate(allE), A=np.concatenate(allA), TS=np.concatenate(allTs),
                counts=counts)


def chrono_folds(ds, window, horizon, n_folds, embargo_frac):
    folds = []
    purge = window + horizon
    for k in range(n_folds):
        tr = np.zeros(len(ds["R"]), bool); te = np.zeros(len(ds["R"]), bool)
        for aid, _ in D.ASSETS:
            idx = np.where(ds["A"] == aid)[0]
            if len(idx) < 50:
                continue
            order = idx[np.argsort(ds["TS"][idx])]
            N = len(order); block = int(N * 0.2)
            hi = N - k * block; lo = max(0, hi - block)
            if hi - lo < 20:
                continue
            test_local = order[lo:hi]
            embargo = int(embargo_frac * N)
            train_hi = lo - purge - embargo
            if train_hi < 20:
                continue
            tr[order[:train_hi]] = True
            te[test_local] = True
        if tr.sum() > 100 and te.sum() > 50:
            folds.append((tr, te))
    return folds


def run_pipeline(args, levels, shuffle_control, seed):
    tag = "SHUFFLE" if shuffle_control else "REAL"
    print(f"\n===== dataset [{tag}] =====", flush=True)
    ds = build_dataset(args.window, args.horizon, args.max_windows_per_asset,
                       shuffle_control, seed)
    print(f"[{tag}] windows={len(ds['R'])} per-asset={ds['counts']}", flush=True)
    folds = chrono_folds(ds, args.window, args.horizon, args.folds, args.embargo_frac)
    print(f"[{tag}] folds={len(folds)}", flush=True)

    fa, fb, fmr, fmp, fppl, fused = [], [], [], [], [], []
    cb = int(np.prod(levels))

    for fi, (tr, te) in enumerate(folds):
        Xtr = ds["X"][tr].astype(np.float64)
        Xte = ds["X"][te].astype(np.float64)

        tok = TokenizerNumpy(args.window, hidden=args.hidden, fsq_levels=levels,
                             seed=seed * 100 + fi)
        rng = np.random.default_rng(seed + fi)
        ntr = Xtr.shape[0]
        for ep in range(args.tok_epochs):
            perm = rng.permutation(ntr); ls = []
            for s in range(0, ntr, args.tok_bs):
                bi = perm[s:s + args.tok_bs]
                ls.append(tok.train_step(Xtr[bi], args.tok_lr))
            if ep == 0 or ep == args.tok_epochs - 1:
                print(f"[{tag} f{fi}] tok ep{ep} mse={np.mean(ls):.5f}", flush=True)

        def enc(Xall):
            ids, recs = [], []
            for s in range(0, Xall.shape[0], 8192):
                i, r = tok.encode(Xall[s:s + 8192])
                ids.append(i); recs.append(r)
            return np.concatenate(ids), np.concatenate(recs)

        idx_tr, _ = enc(Xtr)
        idx_te, rec_te = enc(Xte)

        # METRIC 2 fidelity
        Xte_np = ds["X"][te]
        mse_ret = float(np.mean((rec_te[:, :, 0] - Xte_np[:, :, 0]) ** 2))
        Cwin = ds["C"][te]
        start = Cwin[:, 0:1]
        price_path = start * np.exp(np.cumsum(rec_te[:, 1:, 0], axis=1))
        true_path = Cwin[:, 1:]
        mse_price = float(np.mean(((price_path - true_path) / np.maximum(true_path, 1e-9)) ** 2))

        # METRIC 3 health
        counts = np.bincount(idx_te, minlength=cb).astype(np.float64)
        p = counts / counts.sum(); nz = p[p > 0]
        ppl = float(np.exp(-(nz * np.log(nz)).sum()))
        used = int((counts > 0).sum()); top1 = float(counts.max() / counts.sum())

        # METRIC 1 RQ0 probe
        ctx = args.context
        Rtr, Rte = ds["R"][tr], ds["R"][te]
        Atr, Ate = ds["A"][tr], ds["A"][te]
        TStr, TSte = ds["TS"][tr], ds["TS"][te]

        def make_seq(idx_arr, A_arr, TS_arr, R_arr):
            Xs, Y = [], []
            for aid, _ in D.ASSETS:
                m = np.where(A_arr == aid)[0]
                if len(m) <= ctx:
                    continue
                order = m[np.argsort(TS_arr[m])]
                toks = idx_arr[order]; rr = R_arr[order]
                for j in range(ctx, len(order)):
                    Xs.append(toks[j - ctx:j]); Y.append(1 if rr[j] > 0 else 0)
            if not Xs:
                return np.zeros((0, ctx), np.int64), np.zeros((0,), np.int64)
            return np.asarray(Xs, np.int64), np.asarray(Y, np.int64)

        Sx_tr, Sy_tr = make_seq(idx_tr, Atr, TStr, Rtr)
        Sx_te, Sy_te = make_seq(idx_te, Ate, TSte, Rte)
        if len(Sy_te) < 30 or len(Sy_tr) < 100:
            print(f"[{tag} f{fi}] skip probe", flush=True); continue

        # one-hot each context position over the codebook would be huge; instead
        # use per-position token id as a categorical via target-mean (frequency)
        # encoding fit on TRAIN ONLY (no leakage), then logistic regression.
        # freq-encode: map each token to P(up | token at that ctx position) on train.
        feat_tr = np.zeros((len(Sy_tr), ctx), np.float64)
        feat_te = np.zeros((len(Sy_te), ctx), np.float64)
        prior = Sy_tr.mean()
        for pos in range(ctx):
            tab = {}
            col = Sx_tr[:, pos]
            for tk in np.unique(col):
                mask = col == tk
                # smoothed target mean
                tab[tk] = (Sy_tr[mask].sum() + prior * 5) / (mask.sum() + 5)
            feat_tr[:, pos] = [tab[t] for t in col]
            feat_te[:, pos] = [tab.get(t, prior) for t in Sx_te[:, pos]]

        clf = LogisticRegression(max_iter=500, C=1.0)
        clf.fit(feat_tr, Sy_tr)
        acc = float(clf.score(feat_te, Sy_te))
        base = float(max(Sy_te.mean(), 1 - Sy_te.mean()))

        print(f"[{tag} f{fi}] RQ0 acc={acc:.4f} maj={base:.4f} | "
              f"mse_ret={mse_ret:.6f} mse_price={mse_price:.6f} | "
              f"ppl={ppl:.1f} used={used}/{cb} top1={top1:.3f}", flush=True)
        fa.append(acc); fb.append(base); fmr.append(mse_ret); fmp.append(mse_price)
        fppl.append(ppl); fused.append((used, cb, top1))

    m = lambda v: float(np.mean(v)) if v else float("nan")
    return dict(acc=m(fa), acc_folds=fa, base=m(fb), mse_ret=m(fmr),
                mse_price=m(fmp), ppl=m(fppl), used=fused)


def main():
    args = parse_args()
    if args.quick:
        args.tok_epochs = 5; args.folds = 2; args.max_windows_per_asset = 6000
    levels = tuple(int(x) for x in args.fsq_levels.split(","))
    t0 = time.time()
    real = run_pipeline(args, levels, False, args.seed)
    shuf = run_pipeline(args, levels, True, args.seed)
    gap = real["acc"] - shuf["acc"]
    res = dict(args=vars(args), fsq_levels=levels, real=real, shuffle=shuf,
               edge_gap=gap, wallclock_s=time.time() - t0,
               verdict="PROCEED" if gap >= 0.01 else "KILL")
    print("\n========= SUMMARY =========")
    print(f"real acc  {real['acc']:.4f} folds {['%.3f'%a for a in real['acc_folds']]}")
    print(f"shuf acc  {shuf['acc']:.4f} folds {['%.3f'%a for a in shuf['acc_folds']]}")
    print(f"maj base  {real['base']:.4f}")
    print(f"EDGE gap  {gap:+.4f}")
    print(f"mse_ret {real['mse_ret']:.6f} mse_price {real['mse_price']:.6f}")
    print(f"ppl {real['ppl']:.1f}")
    print(f"VERDICT {res['verdict']}  wall {res['wallclock_s']:.0f}s")
    write_md(args, res, levels)
    with open(args.out.replace(".md", ".json"), "w") as f:
        json.dump(res, f, indent=2, default=float)


def write_md(args, r, levels):
    real, shuf = r["real"], r["shuffle"]
    used = real["used"][0] if real["used"] else (0, 0, 0)
    cb = int(np.prod(levels))
    Ls = []
    a = Ls.append
    a("# TASK_789 C0 Viability Gate — Temporal Token Middle-Layer")
    a("")
    a(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')}_")
    a("")
    a("## Implementation note")
    a("- **Backend: pure numpy (FSQ tokenizer, manual backprop) + sklearn (probe), CPU.**")
    a("  JAX and torch *execution* are blocked in the C0 sandbox (even on CPU; GPU init")
    a("  and `nvidia-smi` denied). numpy/sklearn run fine, so the gate was executed on CPU.")
    a("  A JAX/Flax implementation of the identical design also ships (`fsq.py`,")
    a("  `run_gate.py`) for the GPU rerun once the sandbox permits device init.")
    a("- This is an **honest reduced run**: CPU backend, "
      f"{args.max_windows_per_asset} windows/asset cap, hidden={args.hidden}, "
      f"{args.tok_epochs} tokenizer epochs. No silent truncation — these are the knobs.")
    a("")
    a("## Configuration")
    a(f"- Data: hourly-aggregated OHLC from `trading_data.db` (1m→1h, gap-masked, validity-tracked).")
    a(f"- Assets: {', '.join(n for _, n in D.ASSETS)}")
    a(f"- Window {args.window} bars | Horizon {args.horizon} | Probe context {args.context} tokens")
    a(f"- FSQ levels {levels} → implicit codebook {cb}")
    a(f"- Features (scale-free): log-return(close), body, upper wick, lower wick, range")
    a(f"- RevIN per-window instance-norm → MLP enc → FSQ(STE) → MLP dec; recon-trained then FROZEN.")
    a("")
    a("## Purge + Embargo split spec")
    a(f"- Per-asset CHRONOLOGICAL K-fold ({len(real['acc_folds'])} folds). Test = tail-20% block rolled per fold.")
    a(f"- PURGE = window+horizon = {args.window + args.horizon} bars dropped between train tail and test head")
    a("  (removes overlapping-window leakage — a documented tokenizer-probe trap).")
    a(f"- Embargo fraction = {args.embargo_frac}. Tokenizer trained on TRAIN windows only, frozen before probe.")
    a("- Returns-shuffled control = phase-randomized surrogate of the return channel + time-permuted")
    a("  geometry channels (preserves marginals, destroys temporal structure); SAME pipeline, SAME labels.")
    a("")
    a("## Metric 1 — RQ0 token edge (kill-switch)")
    a(f"- **Real OOS directional acc (fold-mean): {real['acc']:.4f}**  folds {['%.4f'%x for x in real['acc_folds']]}")
    a(f"- **Returns-shuffled control acc: {shuf['acc']:.4f}**  folds {['%.4f'%x for x in shuf['acc_folds']]}")
    a(f"- Majority-class baseline: {real['base']:.4f}")
    a(f"- **EDGE = real − shuffle = {r['edge_gap']:+.4f}**  (PROCEED threshold ≥ +0.01)")
    a("")
    a("## Metric 2 — Decode fidelity (TEST)")
    a(f"- Recon MSE, return channel (normalized feature space): {real['mse_ret']:.6f}")
    a(f"- Relative MSE, DE-NORMALIZED price path (cumulated from recon log-returns): {real['mse_price']:.6f}")
    a("  (price path = first window close × exp(cumsum recon log-returns) — the surface MCTS reward reads).")
    a("")
    a("## Metric 3 — Codebook health (TEST, real stream)")
    a(f"- Implicit codebook {cb}; codes used {used[0]}/{used[1]}; perplexity {real['ppl']:.1f}; "
      f"top-1 share {used[2]:.3f} (low = healthy).")
    a("")
    a("## VERDICT")
    a(f"### **{r['verdict']}**")
    if r["verdict"] == "PROCEED":
        a(f"Real OOS accuracy beats the phase-shuffled surrogate control by {r['edge_gap']:+.4f} "
          f"(≥ +0.01). Shape tokens carry temporal edge the null does not. Proceed to C1.")
    else:
        a(f"Real OOS accuracy does NOT beat the phase-shuffled surrogate control by a meaningful "
          f"margin (gap {r['edge_gap']:+.4f} < +0.01). On our data, shape tokens carry no edge over "
          f"returns-shuffled surrogates out-of-sample — the candlestick-skepticism literature wins. KILL.")
    a("")
    a(f"_wallclock {r['wallclock_s']:.0f}s — numpy/sklearn CPU_")
    with open(args.out, "w") as f:
        f.write("\n".join(Ls) + "\n")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
