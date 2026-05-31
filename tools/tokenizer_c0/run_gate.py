"""
C0 viability gate runner (TASK_789).

Pipeline:
  1. Load hourly OHLC for the chosen assets (data.py), build scale-free features.
  2. Make ALL-valid windows with a purge+embargo chronological split.
  3. Train the FSQ tokenizer by reconstruction on TRAIN windows only; freeze.
  4. Tokenize train+test; train a tiny next-token sign model on TRAIN tokens.
  5. Metrics:
       RQ0  -> directional accuracy of P(sign(next return) | token context) on
               TEST, vs a returns-shuffled control run through the SAME pipeline.
       FID  -> reconstruction MSE on returns + on de-normalized price paths (TEST).
       HEALTH -> FSQ token usage histogram + perplexity (TEST).

  Honest CV: the train/test split is chronological PER ASSET with a purge gap of
  (window + horizon) bars removed between train and test so no training window
  overlaps the test span. We additionally run K chronological folds and average.

Output: writes runs/789_token_c0/RESULT.md and prints a verdict.
"""
import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import data as D


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--tok-epochs", type=int, default=15)
    p.add_argument("--tok-bs", type=int, default=1024)
    p.add_argument("--tok-lr", type=float, default=1e-3)
    p.add_argument("--probe-epochs", type=int, default=40)
    p.add_argument("--probe-bs", type=int, default=2048)
    p.add_argument("--context", type=int, default=4, help="num prior tokens as probe context")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--embargo-frac", type=float, default=0.0,
                   help="extra embargo as fraction of split, on top of window+horizon purge")
    p.add_argument("--max-windows-per-asset", type=int, default=40000)
    p.add_argument("--fsq-levels", type=str, default="8,8,8,5,5")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mem-fraction", type=float, default=0.5)
    p.add_argument("--out", type=str, default="/home/lukas/Trader/.claude/worktrees/agent-ab33144138e003324/runs/789_token_c0/RESULT.md")
    p.add_argument("--quick", action="store_true", help="tiny smoke run")
    return p.parse_args()


def build_dataset(window, horizon, max_per_asset, shuffle_control, seed):
    """Returns concatenated X, closes, fut_ret, plus per-row asset id and a
    global end-time (for chronological splitting) and an asset-local order."""
    rng = np.random.default_rng(seed + (777 if shuffle_control else 0))
    allX, allC, allR, allEnd, allAsset, allTs = [], [], [], [], [], []
    per_asset_counts = {}
    for aid, name in D.ASSETS:
        ts, o, h, l, c, valid = D.load_hourly_ohlc(aid)
        feats, close, fvalid = D.make_features(o, h, l, c, valid)
        if shuffle_control:
            feats = D.phase_shuffle_returns(feats, fvalid, rng)
        X, Cw, R, E = D.make_windows(feats, close, fvalid, window, horizon)
        if X.shape[0] == 0:
            per_asset_counts[name] = 0
            continue
        if X.shape[0] > max_per_asset:
            # keep the most RECENT max_per_asset windows (contiguous, preserves time order)
            X, Cw, R, E = X[-max_per_asset:], Cw[-max_per_asset:], R[-max_per_asset:], E[-max_per_asset:]
        end_ts = ts[1:][E]  # timestamp of last bar in window
        allX.append(X); allC.append(Cw); allR.append(R)
        allEnd.append(E); allAsset.append(np.full(len(R), aid)); allTs.append(end_ts)
        per_asset_counts[name] = int(X.shape[0])
    X = np.concatenate(allX); Cw = np.concatenate(allC); R = np.concatenate(allR)
    E = np.concatenate(allEnd); A = np.concatenate(allAsset); TS = np.concatenate(allTs)
    return dict(X=X, C=Cw, R=R, E=E, A=A, TS=TS, counts=per_asset_counts)


def chrono_folds(ds, window, horizon, n_folds, embargo_frac):
    """Per-asset chronological K-fold. For each asset we sort by end-time and
    cut the tail block as test, with a purge gap of (window+horizon)+embargo
    bars removed from the end of train. Returns list of (train_mask, test_mask).
    We implement an expanding-window style: fold k uses the k-th tail slice as
    test and everything strictly before (minus purge) as train.
    """
    folds = []
    purge = window + horizon
    for k in range(n_folds):
        tr = np.zeros(len(ds["R"]), dtype=bool)
        te = np.zeros(len(ds["R"]), dtype=bool)
        for aid, _ in D.ASSETS:
            idx = np.where(ds["A"] == aid)[0]
            if len(idx) < 50:
                continue
            order = idx[np.argsort(ds["TS"][idx])]
            N = len(order)
            # test block = fold slice from the tail region
            test_frac = 0.2
            block = int(N * test_frac)
            # fold k test window: [N - (k+1)*block, N - k*block)
            hi = N - k * block
            lo = max(0, hi - block)
            if hi - lo < 20:
                continue
            test_local = order[lo:hi]
            embargo = int(embargo_frac * N)
            train_hi = lo - purge - embargo
            if train_hi < 20:
                continue
            train_local = order[:train_hi]
            tr[train_local] = True
            te[test_local] = True
        if tr.sum() > 100 and te.sum() > 50:
            folds.append((tr, te))
    return folds


def main():
    args = parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(args.mem_fraction))
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if args.quick:
        args.tok_epochs = 3; args.probe_epochs = 8; args.folds = 2
        args.max_windows_per_asset = 6000

    import jax, jax.numpy as jnp
    import optax
    from flax.training import train_state
    sys.path.insert(0, os.path.dirname(__file__))
    from fsq import OHLCTokenizer

    print("JAX devices:", jax.devices(), flush=True)
    levels = tuple(int(x) for x in args.fsq_levels.split(","))
    t0 = time.time()

    results = {"args": vars(args), "fsq_levels": levels}

    def run_pipeline(shuffle_control, seed):
        tag = "SHUFFLE" if shuffle_control else "REAL"
        print(f"\n===== building dataset [{tag}] =====", flush=True)
        ds = build_dataset(args.window, args.horizon, args.max_windows_per_asset,
                           shuffle_control, seed)
        print(f"[{tag}] total windows: {len(ds['R'])}  per-asset: {ds['counts']}", flush=True)
        folds = chrono_folds(ds, args.window, args.horizon, args.folds, args.embargo_frac)
        print(f"[{tag}] folds: {len(folds)}", flush=True)

        fold_acc, fold_base, fold_mse_ret, fold_mse_price, fold_ppl, fold_used = [], [], [], [], [], []

        for fi, (tr, te) in enumerate(folds):
            Xtr = jnp.asarray(ds["X"][tr]); Xte = jnp.asarray(ds["X"][te])
            # ---- train tokenizer on TRAIN windows by reconstruction ----
            tok = OHLCTokenizer(window=args.window, fsq_levels=levels)
            key = jax.random.PRNGKey(seed * 100 + fi)
            params = tok.init(key, Xtr[:4])
            tx = optax.adam(args.tok_lr)
            state = train_state.TrainState.create(apply_fn=tok.apply, params=params, tx=tx)

            @jax.jit
            def tok_step(state, xb):
                def loss_fn(p):
                    out = state.apply_fn(p, xb)
                    return jnp.mean((out["recon_n"] - out["xn"]) ** 2)
                loss, grads = jax.value_and_grad(loss_fn)(state.params)
                return state.apply_gradients(grads=grads), loss

            ntr = Xtr.shape[0]
            for ep in range(args.tok_epochs):
                perm = np.random.default_rng(seed + ep).permutation(ntr)
                ep_loss = 0.0; nb = 0
                for s in range(0, ntr, args.tok_bs):
                    bidx = perm[s:s + args.tok_bs]
                    state, l = tok_step(state, Xtr[bidx])
                    ep_loss += float(l); nb += 1
                if ep == 0 or ep == args.tok_epochs - 1:
                    print(f"[{tag} f{fi}] tok ep{ep} recon_n_mse={ep_loss/max(nb,1):.5f}", flush=True)

            # ---- freeze: tokenize train + test ----
            @jax.jit
            def encode(xb):
                out = state.apply_fn(state.params, xb)
                return out["idx"], out["recon"], out["std"], out["mean"]

            def batched_encode(Xall):
                idxs, recs = [], []
                for s in range(0, Xall.shape[0], 4096):
                    i, r, _, _ = encode(Xall[s:s + 4096])
                    idxs.append(np.asarray(i)); recs.append(np.asarray(r))
                return np.concatenate(idxs), np.concatenate(recs)

            idx_tr, _ = batched_encode(Xtr)
            idx_te, rec_te = batched_encode(Xte)

            # ===== METRIC 2: decode fidelity (TEST) =====
            Xte_np = ds["X"][te]
            # return channel = feature 0 (log return of close)
            mse_ret = float(np.mean((rec_te[:, :, 0] - Xte_np[:, :, 0]) ** 2))
            # reconstruct de-normalized PRICE path from reconstructed log-returns.
            # window close path: start from the FIRST close of the window, then
            # multiply by exp(cumsum(recon logret)) for bars 1..window-1.
            Cwin = ds["C"][te]  # [N, window] true closes
            start = Cwin[:, 0:1]
            rec_logret = rec_te[:, 1:, 0]  # reconstructed logrets for bars 1..W-1
            price_path = start * np.exp(np.cumsum(rec_logret, axis=1))
            true_path = Cwin[:, 1:]
            # scale-free price error: relative
            mse_price = float(np.mean(((price_path - true_path) / np.maximum(true_path, 1e-9)) ** 2))

            # ===== METRIC 3: codebook health (TEST tokens) =====
            cb = int(np.prod(levels))
            counts = np.bincount(idx_te, minlength=cb).astype(np.float64)
            p = counts / counts.sum()
            nz = p[p > 0]
            ppl = float(np.exp(-np.sum(nz * np.log(nz))))
            used = int((counts > 0).sum())
            top1 = float(counts.max() / counts.sum())

            # ===== METRIC 1: RQ0 token edge =====
            # tiny next-token sign probe: context = last `ctx` tokens predict
            # sign(future return). Map token id -> embedding, mean-pool ctx,
            # logistic head. Train on TRAIN, eval directional acc on TEST.
            ctx = args.context
            Rtr = ds["R"][tr]; Rte = ds["R"][te]
            Atr = ds["A"][tr]; Ate = ds["A"][te]
            TStr = ds["TS"][tr]; TSte = ds["TS"][te]

            def make_seq(idx_arr, A_arr, TS_arr, R_arr):
                """For each row, gather the ctx tokens immediately preceding it
                IN TIME within the same asset (contiguous hourly windows)."""
                Xseq, Y = [], []
                for aid, _ in D.ASSETS:
                    m = np.where(A_arr == aid)[0]
                    if len(m) <= ctx:
                        continue
                    order = m[np.argsort(TS_arr[m])]
                    toks = idx_arr[order]
                    rr = R_arr[order]
                    for j in range(ctx, len(order)):
                        Xseq.append(toks[j - ctx:j])
                        Y.append(1.0 if rr[j] > 0 else 0.0)
                if not Xseq:
                    return np.zeros((0, ctx), np.int64), np.zeros((0,), np.float32)
                return np.asarray(Xseq, np.int64), np.asarray(Y, np.float32)

            Sx_tr, Sy_tr = make_seq(idx_tr, Atr, TStr, Rtr)
            Sx_te, Sy_te = make_seq(idx_te, Ate, TSte, Rte)
            if len(Sy_te) < 30 or len(Sy_tr) < 100:
                print(f"[{tag} f{fi}] skip probe (too few seq)", flush=True)
                continue

            import flax.linen as nn
            class Probe(nn.Module):
                vocab: int; ctx: int; emb: int = 16
                @nn.compact
                def __call__(self, t):
                    e = nn.Embed(self.vocab, self.emb)(t)  # [B,ctx,emb]
                    h = e.reshape(t.shape[0], -1)
                    h = nn.gelu(nn.Dense(32)(h))
                    return nn.Dense(1)(h)[:, 0]

            probe = Probe(vocab=cb, ctx=ctx)
            pk = jax.random.PRNGKey(seed * 13 + fi)
            pparams = probe.init(pk, jnp.asarray(Sx_tr[:4]))
            ptx = optax.adam(1e-3)
            pstate = train_state.TrainState.create(apply_fn=probe.apply, params=pparams, tx=ptx)

            @jax.jit
            def p_step(st, xb, yb):
                def lf(p):
                    logit = st.apply_fn(p, xb)
                    return optax.sigmoid_binary_cross_entropy(logit, yb).mean()
                l, g = jax.value_and_grad(lf)(st.params)
                return st.apply_gradients(grads=g), l

            Sx_tr_j = jnp.asarray(Sx_tr); Sy_tr_j = jnp.asarray(Sy_tr)
            npr = Sx_tr.shape[0]
            for ep in range(args.probe_epochs):
                perm = np.random.default_rng(1000 + ep).permutation(npr)
                for s in range(0, npr, args.probe_bs):
                    bi = perm[s:s + args.probe_bs]
                    pstate, _ = p_step(pstate, Sx_tr_j[bi], Sy_tr_j[bi])

            @jax.jit
            def p_pred(st, xb):
                return (st.apply_fn(st.params, xb) > 0).astype(jnp.float32)

            preds = []
            for s in range(0, Sx_te.shape[0], 8192):
                preds.append(np.asarray(p_pred(pstate, jnp.asarray(Sx_te[s:s + 8192]))))
            preds = np.concatenate(preds)
            acc = float((preds == Sy_te).mean())
            base = float(max(Sy_te.mean(), 1 - Sy_te.mean()))  # majority-class baseline

            print(f"[{tag} f{fi}] RQ0 acc={acc:.4f} maj_base={base:.4f} "
                  f"| mse_ret={mse_ret:.5f} mse_price={mse_price:.6f} "
                  f"| ppl={ppl:.1f} used={used}/{cb} top1={top1:.3f}", flush=True)

            fold_acc.append(acc); fold_base.append(base)
            fold_mse_ret.append(mse_ret); fold_mse_price.append(mse_price)
            fold_ppl.append(ppl); fold_used.append((used, cb, top1))

        return dict(
            acc=float(np.mean(fold_acc)) if fold_acc else float("nan"),
            acc_folds=fold_acc,
            base=float(np.mean(fold_base)) if fold_base else float("nan"),
            mse_ret=float(np.mean(fold_mse_ret)) if fold_mse_ret else float("nan"),
            mse_price=float(np.mean(fold_mse_price)) if fold_mse_price else float("nan"),
            ppl=float(np.mean(fold_ppl)) if fold_ppl else float("nan"),
            used=fold_used,
        )

    real = run_pipeline(shuffle_control=False, seed=args.seed)
    shuf = run_pipeline(shuffle_control=True, seed=args.seed)

    gap = real["acc"] - shuf["acc"]
    results["real"] = real
    results["shuffle"] = shuf
    results["edge_gap"] = gap
    results["wallclock_s"] = time.time() - t0

    verdict = "PROCEED" if gap >= 0.01 else "KILL"
    results["verdict"] = verdict

    print("\n================ SUMMARY ================", flush=True)
    print(f"RQ0 real acc   = {real['acc']:.4f}  (folds {['%.3f'%a for a in real['acc_folds']]})")
    print(f"RQ0 shuf acc   = {shuf['acc']:.4f}  (folds {['%.3f'%a for a in shuf['acc_folds']]})")
    print(f"RQ0 maj base   = {real['base']:.4f}")
    print(f"EDGE gap       = {gap:+.4f}")
    print(f"Decode mse_ret = {real['mse_ret']:.5f}  mse_price(rel) = {real['mse_price']:.6f}")
    print(f"Codebook ppl   = {real['ppl']:.1f}  (real)")
    print(f"VERDICT        = {verdict}")
    print(f"wallclock      = {results['wallclock_s']:.0f}s")

    write_result_md(args, results)
    with open(args.out.replace(".md", ".json"), "w") as f:
        json.dump(results, f, indent=2, default=float)


def write_result_md(args, r):
    real, shuf = r["real"], r["shuffle"]
    used_real = real["used"][0] if real["used"] else (0, 0, 0)
    lines = []
    L = lines.append
    L("# TASK_789 C0 Viability Gate — Temporal Token Middle-Layer")
    L("")
    L(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} — impl: JAX/Flax — single RTX 3090_")
    L("")
    L("## Configuration")
    L(f"- Data: hourly-aggregated OHLC from `trading_data.db` (1m→1h, gaps masked).")
    L(f"- Assets: {', '.join(n for _, n in D.ASSETS)}")
    L(f"- Window: {args.window} bars | Horizon: {args.horizon} bar(s) | Probe context: {args.context} tokens")
    L(f"- FSQ levels: {r['fsq_levels']} → implicit codebook = {int(np.prod(r['fsq_levels']))}")
    L(f"- Features (scale-free): log-return(close), body, upper wick, lower wick, range")
    L(f"- Tokenizer: RevIN per-window instance-norm → MLP enc → FSQ → MLP dec (recon-trained, frozen)")
    L("")
    L("## Purge + Embargo Split Spec")
    L(f"- Per-asset CHRONOLOGICAL K-fold ({len(real['acc_folds'])} folds reported).")
    L(f"- Test = tail 20% block (rolled back per fold). Train = strictly-earlier bars.")
    L(f"- PURGE gap = window+horizon = {args.window + args.horizon} bars dropped between "
      f"train tail and test head (kills overlapping-window leakage).")
    L(f"- Embargo fraction = {args.embargo_frac} (extra gap as fraction of asset series).")
    L(f"- Tokenizer trained on TRAIN windows ONLY, frozen before probe/eval.")
    L("")
    L("## Metric 1 — RQ0 token edge (THE kill-switch)")
    L(f"- **Real** directional acc (OOS, fold-mean): **{real['acc']:.4f}**  folds={['%.4f'%a for a in real['acc_folds']]}")
    L(f"- **Returns-shuffled control** acc (same pipeline, phase-randomized): **{shuf['acc']:.4f}**  folds={['%.4f'%a for a in shuf['acc_folds']]}")
    L(f"- Majority-class baseline: {real['base']:.4f}")
    L(f"- **EDGE = real − shuffle = {r['edge_gap']:+.4f}**")
    L("")
    L("## Metric 2 — Decode fidelity (TEST)")
    L(f"- Reconstruction MSE on return channel (normalized feature space): {real['mse_ret']:.6f}")
    L(f"- Relative MSE on DE-NORMALIZED price path (cumulated from recon log-returns): {real['mse_price']:.6f}")
    L("  - (price path rebuilt from the FIRST window close × exp(cumsum recon log-returns);")
    L("     this is the surface the eventual MCTS reward would read.)")
    L("")
    L("## Metric 3 — Codebook health (TEST, real stream)")
    L(f"- Implicit codebook size: {int(np.prod(r['fsq_levels']))}")
    L(f"- Codes used: {used_real[0]} / {used_real[1]}")
    L(f"- Perplexity (fold-mean): {real['ppl']:.1f}")
    L(f"- Top-1 code share (fold 0): {used_real[2]:.3f}  (low = healthy, no collapse)")
    L("")
    L("## VERDICT")
    L(f"### **{r['verdict']}**")
    if r["verdict"] == "PROCEED":
        L(f"Real OOS directional accuracy beats the returns-shuffled control by "
          f"{r['edge_gap']:+.4f} (≥ +0.01 threshold). Shape tokens carry temporal edge "
          f"the surrogate-null does not. Proceed to C1.")
    else:
        L(f"Real OOS directional accuracy does NOT beat the returns-shuffled control "
          f"by a meaningful margin (gap {r['edge_gap']:+.4f} < +0.01). The candlestick-"
          f"skepticism literature wins on our data at this resolution: shape tokens carry "
          f"no edge over phase-randomized surrogates out-of-sample. KILL the branch.")
    L("")
    L(f"_wallclock: {r['wallclock_s']:.0f}s_")
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
