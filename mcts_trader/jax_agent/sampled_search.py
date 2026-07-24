"""TASK_828 Stage 2 — Sampled-MuZero MCTS over joint action-VECTORS.

The scalar mctx search (``mctx_search.py``, CANON) holds a single categorical
over the 320 joint (asset, action) indices — it cannot represent the 5^n_assets
joint action-vector space a portfolio rebalance lives in. This module adds a
NEW, flag-gated search (Sampled MuZero, Hubert et al. 2021, arXiv:2104.06303):

  1. At the root, SAMPLE K candidate action-VECTORS from the factored policy
     head's per-asset marginals (independent per-asset categorical draws). K is
     a config parameter (default 16). Duplicates are allowed (i.i.d. sampling).
  2. Treat those K sampled vectors as the node's action set — a reduced MDP with
     a {0..K-1} action space. Run mctx's Gumbel-MuZero search / Sequential
     Halving + qtransform machinery over the K candidates (REUSED unchanged from
     canon; ``mctx_search.py`` is not imported or modified here → the canon
     scalar path is trivially bit-identical).
  3. Expand the tree with the SAMPLED VECTOR (not a scalar index): the
     recurrent_fn gathers candidate k's vector and calls the model's
     ``recurrent_inference_vector`` (dynamics_vector embeds the (n_assets,)
     vector — muzero_model.py). Reward/value therefore condition on the whole
     rebalance.
  4. CORRECT visit counts for the sampling distribution (§ _empirical_prior):
     the intractable joint prior π(a)=Πᵢπᵢ(aᵢ) is replaced by the EMPIRICAL
     sample-based prior β̂(a)=n_a/K, expressed per-slot as a uniform 1/K prior
     (log-uniform prior_logits). Because the K slots were drawn i.i.d. from β,
     duplicate vectors accumulate n_a/K → the search operates with an effective
     prior ≈ β WITHOUT ever evaluating the joint. This is THE Sampled-MuZero
     correction (Hubert 2021 §3, PUCT/empirical-prior variant).

Returns, per batch element: the selected action-VECTOR (B, n_assets) ∈ {0..4},
the K candidate vectors (B, K, n_assets), and the visit distribution over the
sampled set (B, K) — the raw material for the Stage-3 factored policy target.

Design calls flagged for the verify panel / operator (see task Factory notes):
  * DEPTH / tree reuse: by DEFAULT interior nodes reuse the ROOT's K sampled
    vectors as their action set (mctx requires a fixed NUMBER of actions across
    depth). With num_simulations modest the search is dominated by root
    Sequential-Halving over the sampled set (Sampled-MuZero root planning),
    which is the load-bearing behavior here.
    TASK_838: ``per_node_resampling=True`` (flag, default OFF = bit-identical
    root-reuse) resamples K vectors from the factored policy at EACH expanded
    interior node's latent — the mctx embedding pytree carries
    ``{'latent', 'cand_vecs'}`` so slot k at a node maps to the k-th vector
    sampled AT that node (fixed K per node keeps shapes static). The
    Sampled-MuZero correction stays exact PER NODE: each node's K slots are
    drawn i.i.d. from that node's own proposal β, so the sample-based prior
    β̂(a)=n_a/K is again the uniform-1/K slot prior (Hubert et al. 2021,
    arXiv:2104.06303 — the correction is local to the node that sampled).
    Interior rng: deterministic derivation from the expand rng mctx passes to
    recurrent_fn — ``fold_in(rng, 838)`` then split into (dither, sample); no
    top-level rng consumption changes, ON or OFF. Interior book/mask: the
    latent does not carry a book, so the per-node proposal uses the HOLD-book
    (book=None → hold_bias on the HOLD slot, DEC-3 v1 semantics) and no
    action_mask (interior vectors are planning candidates, never executed).
  * PRIOR: uniform-per-slot empirical prior at ALL depths (the samples ARE the
    prior; the model policy enters only via the sampling proposal). Alternative:
    importance-weighted per-candidate logp(vec) interior priors — not used, to
    keep the correction single-sourced and unbiased.
  * EMBEDDING: joint one-hot → single Dense (VectorActionEmbedding). See
    muzero_model.py for the rationale.
  * TASK_851 — ``leaf_eval={'model','true-env'}`` (default 'model' = every branch above,
    canon bit-identical). LEDGER 849/leaf measured the search leaf v̂(g(s,a)) ACTION-BLIND
    (+0.03 corr vs true one-step reward, vs r̂'s +0.29) — with a flat v̂, maximizing r̂+γv̂
    sees only the fee cost of trading and search rationally converges to HOLD/churn. E1/831
    proved the fix with the SAME proposal machinery: a TRUE-ENV R-step rollout leaf scored
    +5.29 net@21 (vs the model leaf's −6.92). ``leaf_eval='true-env'`` WIRES that proven
    operator (``experiments/831_search_probe/search_probe.py::build_search_step`` /
    ``eval_candidate`` — reused verbatim via ``_true_env_leaf_values``, not reinvented) into
    this shipped Gumbel search path:
      - candidate generation (``_build_proposal`` / ``_policy_prior_candidates``) is
        UNTOUCHED — only the per-candidate LEAF reward/value changes.
      - for each of the K root candidates, step a COPY of the true env forward with that
        candidate (r0), then ``leaf_rollout_r`` (R, default 3, 831x/C's "+5.29" depth; R=1
        supported, "already beats the engine ref") further TRUE-env steps under HOLD, giving
        V = Σ_{t=1}^{R} γ^(t-1)·r_t; if ``leaf_vhat_tail`` (default ON) add γ^R·v̂ at the
        horizon, where v̂ comes from the MODEL's own latent advanced by the IDENTICAL action
        sequence (candidate, then R holds) — no observation re-encoding needed. R=0 degenerates
        to V = v̂ (tail ON) or V = 0 (tail OFF, so Q = r0 exactly — "pure truncated return").
      - all K candidates' (r0, V) are precomputed ONCE per decision, vmapped over K
        (cost guard, task scope 3) through the SAME ``batch_step_vector`` self-play/eval
        already pay for; ``recurrent_fn`` becomes an O(1) table lookup by candidate index.
      - the mctx call is made with ``max_depth=1``: since the K candidates' values are FIXED
        per decision (root-planning, matching 831 exactly — no path-dependent recursion),
        letting the search "go deeper" than depth 1 would just re-derive an arbitrary
        composite of two UNRELATED one-step-ahead evaluations (mctx's own doc: "a leaf can be
        expanded multiple times" under a depth cutoff — i.e. re-evaluates the SAME node,
        never compounds). This keeps the true-env branch semantically identical to 831's
        flat argmax while still running the REAL Gumbel/Sequential-Halving selection +
        visit-count statistics (the "wiring in", not a bypass).
      - determinism/scoping: the rollout reads the SAME panel rows/prices the decision is
        being made on, bounded to the R-step horizon — identical in kind to E1's probe (a
        sanctioned in-simulator planning lookahead, not train/test leakage; no lookahead
        beyond R).
      - incompatible with ``per_node_resampling=True`` (rejected in ``__init__``): there is
        no "interior node" left to resample at depth 1, and per-node resampling's own
        rationale (fresh candidates per node) doesn't apply to a flattened root-planning
        search. ``root_value`` (the model's raw prior decode) is UNCHANGED by this flag —
        distilling the true-env leaf back into v̂ is a separate, later task (D19 opt (c)).

CPU / fixture-only for tests. Never GPU-hours in the factory loop.
"""

import jax
import jax.numpy as jnp
import mctx
from typing import Callable, Tuple, Any
import chex


class SampledMuZeroVectorSearch:
    """Sampled-MuZero MCTS over joint per-asset action-vectors (flag-gated).

    Drop-in-adjacent to ``MuZeroMCTXSearch`` but returns an action VECTOR
    (B, n_assets) rather than a scalar action index. Requires a model built with
    ``use_vector_action=True`` (so ``recurrent_inference_vector`` /
    ``dynamics_vector`` exist).
    """

    def __init__(
        self,
        model_apply_fn: Callable,
        n_assets: int = 64,
        n_actions_per_asset: int = 5,
        num_samples: int = 16,          # K — candidate action-vectors per root
        num_simulations: int = 32,
        max_num_considered_actions: int = 16,
        discount: float = 0.997,
        gumbel_scale: float = 1.0,
        value_scale: float = 1.0,
        value_shift: float = 0.0,
        # TASK_828 Stage 3: dithered-book proposal knobs (D-2/D-3).
        hold_bias: float = 0.0,             # additive bias on the book (HOLD) slot
        dither_scale: float = 1.0,          # Gumbel dither scale on the proposal
        proposal_scheme: str = 'dithered-book',  # 'dithered-book' | 'iid-beta' | 'policy-prior'
        # TASK_847: policy-prior proposal knobs (used ONLY when
        # proposal_scheme == 'policy-prior'; ignored by every other scheme so
        # the default paths stay bit-identical — see _build_candidates).
        proposal_temp: float = 1.0,         # temperature on the per-asset marginals
        explore_slots: int = 2,             # N dither-machinery slots retained (of K)
        # TASK_854 (D21): scalar weight on the LEARNED policy prior in the Gumbel
        # candidate-scoring / proposal path. The candidate set is drawn by Gumbel-max
        # from `prior_weight·policy_logits + book/hold bias + dither·g` (dithered-book)
        # or from `prior_weight·policy_logits` (iid / policy-prior slots). Default 1.0 =
        # today, BIT-IDENTICAL (the scaling branch is not even traced). 0.0 = pure-Q =
        # the E1 selection: with the learned prior removed the candidate scoring is
        # invariant to policy_logits (E1's dithered-book fed zero policy, only the HOLD
        # bias). See LEDGER 832/E5-inversion2 + DECISIONS D21 for why this is the LAST
        # learned component in the true-env-leaf loop. NOTE (verify panel): the empirical
        # Sampled-MuZero prior fed to mctx's RootFnOutput is uniform (1/K) at ALL depths
        # (`_empirical_prior`), so the learned prior enters ONLY through candidate
        # generation, never through mctx's internal `logits + g + σ(Q)` root selection —
        # scaling mctx's uniform prior_logits would be a no-op (exactly the `--value-scale`
        # inert-flag hazard this task also fixes). This knob therefore gates the learned
        # policy where it actually lives: the proposal that scores/produces the candidates.
        prior_weight: float = 1.0,
        # TASK_838: resample K vectors at each expanded interior node (default
        # OFF = root-reuse, bit-identical to the shipped Stage-2/3 behavior).
        per_node_resampling: bool = False,
        # TASK_851: search LEAF evaluator. 'model' (default) = every branch above,
        # canon bit-identical. 'true-env' replaces the per-candidate leaf reward/value
        # with a true-env rollout (module docstring TASK_851 section has the full design).
        leaf_eval: str = 'model',
        leaf_rollout_r: int = 3,          # R further true-env HOLD steps past the candidate
        leaf_vhat_tail: bool = True,      # + discounted v̂ at the R-step horizon (true-env only)
    ):
        self.model_apply_fn = model_apply_fn
        self.n_assets = n_assets
        self.n_actions_per_asset = n_actions_per_asset
        self.num_samples = num_samples          # K (static)
        self.num_simulations = num_simulations
        # Sequential Halving cannot consider more actions than we sampled.
        self.max_num_considered_actions = min(max_num_considered_actions,
                                              num_samples)
        self.discount = discount
        self.gumbel_scale = gumbel_scale
        self.value_scale = value_scale
        self.value_shift = value_shift
        # Proposal (sampling-distribution) knobs — static, captured at trace.
        self.hold_bias = hold_bias
        self.dither_scale = dither_scale
        self.proposal_scheme = proposal_scheme
        # TASK_847 policy-prior knobs (static; ignored unless scheme is policy-prior).
        self.proposal_temp = proposal_temp
        self.explore_slots = explore_slots
        # TASK_854 (D21): learned-prior weight in the candidate scoring/proposal (static
        # Python float captured at trace; == 1.0 leaves the proposal graph untouched).
        self.prior_weight = prior_weight
        if proposal_scheme == 'policy-prior':
            # Fail-loud on misconfiguration (candidate 0 is always the argmax vector).
            if not (0 <= explore_slots <= num_samples - 1):
                raise ValueError(
                    f"policy-prior needs 0 <= explore_slots <= num_samples-1 "
                    f"(got explore_slots={explore_slots}, num_samples={num_samples}); "
                    f"candidate 0 is always the policy-argmax vector.")
            if proposal_temp <= 0:
                raise ValueError(
                    f"policy-prior needs proposal_temp > 0 (got {proposal_temp}); "
                    f"temp<=0 would silently collapse the sampled slots to the argmax.")
            if per_node_resampling:
                # Interior policy-prior is OUT OF SCOPE for TASK_847: the interior
                # recurrent_fn expands via _build_proposal, which under a
                # 'policy-prior' scheme falls through to raw masked marginals (no
                # argmax anchor, no dither) — a silent-wrong hybrid. Reject the
                # combo rather than plumb an unspecified interior proposal.
                raise ValueError(
                    "proposal_scheme='policy-prior' is incompatible with "
                    "per_node_resampling=True (interior policy-prior is out of "
                    "scope for TASK_847); use per_node_resampling=False.")
        # TASK_838 — static Python bool, captured at construction/trace time:
        # OFF ⇒ the per-node branches below are never traced (XLA DCE like every
        # prior stage) and the rng stream is untouched.
        self.per_node_resampling = per_node_resampling

        # TASK_851 — validated + captured at construction (static Python values; the
        # untaken leaf_eval branch in _search is never traced, same XLA-DCE discipline).
        if leaf_eval not in ('model', 'true-env'):
            raise ValueError(f"leaf_eval must be 'model' or 'true-env', got {leaf_eval!r}")
        if leaf_eval == 'true-env':
            if leaf_rollout_r < 0:
                raise ValueError(f"leaf_rollout_r must be >= 0, got {leaf_rollout_r}")
            if per_node_resampling:
                raise ValueError(
                    "leaf_eval='true-env' is incompatible with per_node_resampling=True: "
                    "the true-env leaf flattens the search to depth 1 (root-planning over "
                    "the K candidates — module docstring TASK_851 section) and per-node "
                    "resampling has no interior node left to resample at depth 1. Use "
                    "per_node_resampling=False (the default).")
        self.leaf_eval = leaf_eval
        self.leaf_rollout_r = leaf_rollout_r
        self.leaf_vhat_tail = leaf_vhat_tail

        # TASK_854 scope 2 (D21): fail-loud on the INERT `value_scale` combination rather
        # than silently accepting a knob with no effect (the verification hazard that burned
        # a GPU run — LEDGER 832/E5-inversion2). `value_scale` scales the MODEL's categorical
        # value decode (`_categorical_to_scalar`). In the true-env leaf the per-candidate Q =
        # r0 + γ·V is a TRUE-env rollout (`_true_env_leaf_values`) and only the optional v̂
        # TAIL routes through the model value decode — so with the tail OFF, value_scale
        # touches nothing that reaches candidate selection (byte-identical, as the probe
        # found). It stays truthfully wired everywhere it DOES bite (the model leaf, and the
        # true-env leaf WITH tail); only the provably-inert combination is rejected.
        if (leaf_eval == 'true-env' and not leaf_vhat_tail
                and float(value_scale) != 1.0):
            raise ValueError(
                f"value_scale={value_scale} has no effect with leaf_eval='true-env' and "
                f"leaf_vhat_tail=False: the true-env leaf Q is a pure true-env rollout, so "
                f"value_scale (which only scales the MODEL value decode) never reaches "
                f"candidate selection. Enable leaf_vhat_tail to route it through the v̂ tail, "
                f"or drop --value-scale (leave it 1.0). Fail-loud per TASK_854 scope 2.")

        # TASK_855 scope 1: `value_shift` shares the IDENTICAL silent-inert hazard as
        # `value_scale` above — it is an additive shift on the MODEL value decode
        # (`_categorical_to_scalar`: raw_value*value_scale + value_shift; inert default 0.0),
        # so with the true-env leaf and no v̂ tail it never reaches candidate selection either.
        # Fail-loud on that provably-inert combination; truthfully wired everywhere it DOES bite.
        if (leaf_eval == 'true-env' and not leaf_vhat_tail
                and float(value_shift) != 0.0):
            raise ValueError(
                f"value_shift={value_shift} has no effect with leaf_eval='true-env' and "
                f"leaf_vhat_tail=False: the true-env leaf Q is a pure true-env rollout, so "
                f"value_shift (which only shifts the MODEL value decode) never reaches "
                f"candidate selection. Enable leaf_vhat_tail to route it through the v̂ tail, "
                f"or drop --value-shift (leave it 0.0). Fail-loud per TASK_855 scope 1.")

        # env_config (TASK_851, true-env only) is a hashable chex.dataclass — static, like
        # every other jit wrapper in this codebase (e.g. jit_batch_step_vector). Canon calls
        # never pass it (stays None ⇒ one compiled variant, unaffected).
        self.search_fn = jax.jit(self._search, static_argnames=['env_config'])

    # ---- value/reward categorical → scalar (mirrors mctx_search.py canon) ----

    def _categorical_to_scalar(self, logits: jnp.ndarray) -> jnp.ndarray:
        from . import get_value_support
        support = get_value_support()
        probs = jax.nn.softmax(logits, axis=-1)
        raw_value = jnp.sum(probs * support, axis=-1)
        return raw_value * self.value_scale + self.value_shift

    def _reward_to_scalar(self, logits: jnp.ndarray) -> jnp.ndarray:
        from . import get_value_support
        support = get_value_support()
        probs = jax.nn.softmax(logits, axis=-1)
        return jnp.sum(probs * support, axis=-1)

    # ---- sampling + Sampled-MuZero visit-count correction --------------------

    def _sample_action_vectors(
        self, rng: chex.PRNGKey, policy_logits: jnp.ndarray
    ) -> jnp.ndarray:
        """Draw K action-VECTORS from the factored policy's per-asset marginals.

        policy_logits: (B, n_assets*A) — the flattened factored policy head, laid
        out [asset, action] (index = asset*A + action, matching the env decode
        asset=idx//A, action=idx%A). Reshape → (B, n_assets, A); each per-asset
        row is an independent categorical marginal. K i.i.d. joint samples are
        drawn by sampling every asset independently.

        Returns (B, K, n_assets) int32 in {0..A-1}.
        """
        B = policy_logits.shape[0]
        pa_logits = policy_logits.reshape(B, self.n_assets, self.n_actions_per_asset)
        # (K, B, n_assets) independent per-asset categorical draws → transpose.
        samples = jax.random.categorical(
            rng, pa_logits, axis=-1, shape=(self.num_samples, B, self.n_assets)
        )
        return jnp.transpose(samples, (1, 0, 2)).astype(jnp.int32)  # (B, K, n_assets)

    def _empirical_prior(
        self, sampled_vecs: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Sampled-MuZero sample-based prior + per-candidate empirical mass β̂.

        The proposal is β = the factored policy π. K vectors are drawn i.i.d.
        from β. The Sampled-MuZero correction replaces the intractable joint
        prior π(a)=Πᵢπᵢ(aᵢ) with the empirical frequency β̂(a)=n_a/K. In the
        STATIC-shape (fixed-K) representation this is a uniform 1/K prior over
        the K slots (log-uniform prior_logits); identical vectors then accumulate
        n_a/K mass naturally inside mctx's visit allocation — recovering β̂
        WITHOUT evaluating the joint.

        Returns:
            prior_logits: (B, K) — log-uniform (all zeros ⇒ softmax = 1/K). This
                is what the search consumes as the corrected prior.
            beta_hat:     (B, K) — per-slot per-candidate empirical mass n_a/K
                (n_a = number of the K slots whose vector equals this slot's).
                Exposed for diagnostics + the Stage-3 training-target correction;
                unit-tested against hand-worked tiny examples.
        """
        B, K, _ = sampled_vecs.shape
        prior_logits = jnp.zeros((B, K), dtype=jnp.float32)  # uniform 1/K
        # eq[b,i,j] = (vector at slot i == vector at slot j)
        eq = jnp.all(sampled_vecs[:, :, None, :] == sampled_vecs[:, None, :, :],
                     axis=-1)                                  # (B, K, K)
        counts = jnp.sum(eq.astype(jnp.int32), axis=-1)        # (B, K): n_a per slot
        beta_hat = counts.astype(jnp.float32) / float(K)       # (B, K)
        return prior_logits, beta_hat

    # ---- TASK_851: true-env leaf evaluator (reuses the 831 operator verbatim) ------------

    def _model_latent_step(self, params, latent, vec):
        """One ``recurrent_inference_vector`` call, decoded to (next_latent, value_scalar).

        Used ONLY by the true-env leaf's optional v̂ tail (``leaf_vhat_tail``): advances the
        MODEL's own latent by the SAME action the true env just took, so the tail bootstrap
        needs no observation re-encoding from the rolled-out env state.
        """
        out = self.model_apply_fn(
            {'params': params}, latent, vec, mode='recurrent_inference_vector')
        return out['latent_state'], self._categorical_to_scalar(out['value'])

    def _true_env_leaf_values(
        self, params, env_state, prices, env_config, latent, sampled_vecs
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """TASK_851 — per-candidate (reward, value) under the true-env leaf.

        Ports ``experiments/831_search_probe/search_probe.py``'s ``build_search_step`` /
        ``eval_candidate`` (the E1 +5.29 operator, LEDGER 849/leaf) — reused, not reinvented —
        generalized from a single env instance to the shipped BATCH-of-B search, and extended
        with the optional discounted v̂ tail (a TASK_851 addition; not part of the 831 probe).

        For candidate k, with R = ``self.leaf_rollout_r`` and γ = ``self.discount``::

            V_k = Σ_{t=1}^{R} γ^(t-1)·r_t                     [true-env HOLD rollout]
                  + γ^R · v̂(s_{1+R})                          [only if leaf_vhat_tail]
            leaf_reward_k = r0                 leaf_value_k = V_k

        The caller feeds ``(leaf_reward_k, leaf_value_k)`` to ``RecurrentFnOutput``; mctx's
        own backup then forms ``Q_k = r0 + γ·V_k`` — exactly 831's ``eval_candidate`` formula
        (generalized with the optional tail). R=0 ⇒ V_k = v̂(s_1) (tail ON) or 0 (tail OFF, so
        Q_k = r0 exactly — the "pure truncated return" property).

        Vectorized over the K candidates via ``jax.vmap`` (cost guard, task scope 3): ONE
        vmapped K-candidate × R-step true-env rollout computed ONCE per decision, through the
        same ``batch_step_vector`` self-play/eval already pay for.

        Args:
            env_state: TradingState, batch B — the REAL decision state the search is deciding
                on (a JAX pytree; "stepping a copy" is automatic/functional, no explicit copy).
            prices: (T, n_assets) for the SHARED-price batch, or (B, T, n_assets) when each
                env carries its OWN price series (``--unique-data-per-env``, self_play.py's
                ``unique_prices_mode``) — the SAME panel the decision is being made on (no
                lookahead beyond the R-step horizon). Dispatches to ``batch_step_vector`` vs
                ``batch_step_vector_unique`` on ``prices.ndim`` exactly as self_play.py's real
                env step already does (fix pass, correctness-judge finding: unique-price runs
                previously crashed with an opaque broadcast-shape error).
            env_config: EnvConfig — the SAME fee/slippage tier as the live step.
            latent: (B, state_dim) — root model latent (read only if leaf_vhat_tail).
            sampled_vecs: (B, K, n_assets) int32 — the K candidate vectors (UNCHANGED
                generation; TASK_851 never touches how candidates are proposed).

        Returns:
            leaf_reward: (B, K) float32 — r0 per candidate.
            leaf_value:  (B, K) float32 — V per candidate (formula above).
        """
        from mcts_trader.env.jax_trading_env import (
            batch_step_vector, batch_step_vector_unique,
        )
        B, K, A = sampled_vecs.shape
        R = self.leaf_rollout_r
        discount = self.discount
        hold_vecs = jnp.ones((B, A), dtype=jnp.int32)

        # Fix pass (correctness-judge MAJOR): ``prices.ndim`` is a STATIC/structural
        # property (fixed at trace time, not a traced value), so a plain Python dispatch
        # here mirrors self_play.py:699-708's ``unique_prices_mode`` exactly — never a
        # lax.cond, no per-call branching cost.
        unique_prices_mode = prices.ndim == 3
        step_fn = batch_step_vector_unique if unique_prices_mode else batch_step_vector

        def eval_one_candidate(vec_k):
            # vec_k: (B, n_assets) — candidate k's vector for every batch element.
            s1, _, r0, _ = step_fn(env_state, vec_k, prices, env_config)

            if R > 0:
                def hold_step(carry_state, _):
                    ns, _, r, _ = step_fn(carry_state, hold_vecs, prices, env_config)
                    return ns, r
                _, rs = jax.lax.scan(hold_step, s1, xs=None, length=R)   # rs: (R, B)
                disc = discount ** jnp.arange(R)                          # (R,)
                v_trunc = jnp.sum(disc[:, None] * rs, axis=0)             # (B,)
            else:
                v_trunc = jnp.zeros_like(r0)

            if self.leaf_vhat_tail:
                lat, vhat = self._model_latent_step(params, latent, vec_k)
                for _ in range(R):    # R is a static python int — unrolled, cheap (small R)
                    lat, vhat = self._model_latent_step(params, lat, hold_vecs)
                v_total = v_trunc + (discount ** R) * vhat
            else:
                v_total = v_trunc
            return r0, v_total

        leaf_reward_kb, leaf_value_kb = jax.vmap(
            eval_one_candidate, in_axes=1, out_axes=0)(sampled_vecs)     # (K, B) each
        return (jnp.transpose(leaf_reward_kb, (1, 0)),
                jnp.transpose(leaf_value_kb, (1, 0)))                    # (B, K) each

    # ---- search --------------------------------------------------------------

    def _build_proposal(
        self,
        policy_logits: jnp.ndarray,
        action_mask: jnp.ndarray,
        book: jnp.ndarray,
        dither_rng: chex.PRNGKey,
        scheme: str = None,
    ) -> jnp.ndarray:
        """TASK_828 Stage 3: dithered-book proposal (D-2/D-3).

        Start from the factored policy logits (asset-major (B, n_assets*A)),
        reshape → (B, n_assets, A), then:
          * add a constant ``hold_bias`` to the book slot per asset (v1: HOLD=1,
            DEC-3 HOLD-maintains-position; HOLD is always legal),
          * add independent Gumbel dither (× ``dither_scale``) per (asset, slot),
          * mask illegal per-asset slots to a large negative so softmax(·) → ~0.
        The K candidate vectors are then sampled from ``softmax(proposal, -1)`` per
        asset. The Sampled-MuZero visit correction is proposal-independent (the
        samples ARE drawn from this proposal), so ``_empirical_prior`` stays
        uniform-1/K.

        ``scheme`` defaults to ``self.proposal_scheme`` (default-path calls pass
        nothing ⇒ the traced graph is unchanged). TASK_847 passes an explicit
        ``'dithered-book'`` to reuse the dither machinery for the policy-prior
        exploration slots regardless of ``self.proposal_scheme``.
        """
        scheme = self.proposal_scheme if scheme is None else scheme
        B = policy_logits.shape[0]
        n_assets, A = self.n_assets, self.n_actions_per_asset
        proposal = policy_logits.reshape(B, n_assets, A)
        # TASK_854 (D21): scale the LEARNED policy prior. Static Python guard ⇒ at the
        # default 1.0 the graph is untouched (bit-identical); 0.0 zeroes the policy so the
        # candidate scoring below is invariant to policy_logits (E1's dithered-book fed
        # zero policy). Applied BEFORE the book/HOLD bias + dither so those structural
        # terms are unaffected by prior_weight (they are not the learned component).
        if self.prior_weight != 1.0:
            proposal = self.prior_weight * proposal

        if scheme == 'dithered-book':
            if book is not None:
                # Bias the maintain-action slot per asset (v1 book = HOLD=1).
                book_oh = jax.nn.one_hot(book.astype(jnp.int32), A,
                                         dtype=proposal.dtype)   # (B, n_assets, A)
                proposal = proposal + self.hold_bias * book_oh
            else:
                # No explicit book → bias the HOLD slot (index 1) directly.
                proposal = proposal.at[..., 1].add(self.hold_bias)
            proposal = proposal + self.dither_scale * jax.random.gumbel(
                dither_rng, proposal.shape, dtype=proposal.dtype)
        # 'iid-beta' (or any other scheme) → sample from the raw factored marginals
        # (no book bias / dither); mask still applies below.

        if action_mask is not None:
            mask_bool = action_mask.reshape(B, n_assets, A).astype(bool)
            proposal = jnp.where(mask_bool, proposal, -1e9)
        return proposal

    def _policy_prior_candidates(
        self,
        policy_logits: jnp.ndarray,
        action_mask: jnp.ndarray,
        book: jnp.ndarray,
        sample_rng: chex.PRNGKey,
        dither_rng: chex.PRNGKey,
    ) -> jnp.ndarray:
        """TASK_847: policy-prior candidate set for the vector search.

        The dithered-book proposal is POLICY-BLIND — it perturbs the current
        book, so the policy's preferred action vector is never among the K
        candidates and the prior cannot express itself through search (LEDGER
        832/E5-exit/erosion/autopsy). This scheme lays the K candidates out so
        the learned prior is always representable, per batch element:

          slot 0          : the policy-argmax vector (per-asset argmax of the
                            masked policy marginals) — GUARANTEED every root;
          slots 1..K-1-N  : i.i.d. per-asset draws from the marginals at
                            temperature ``proposal_temp``;
          slots K-N..K-1  : N exploration slots drawn from the existing
                            dithered-book machinery (``hold_bias`` + Gumbel
                            dither), retained so the search keeps a little
                            book-anchored exploration.

        The legality mask is applied to every slot. K and N are static Python
        ints (captured at trace) so shapes are fixed. ``_empirical_prior`` stays
        uniform-1/K (the slot prior mctx consumes) and duplicate vectors (e.g.
        the argmax also drawn by a sample or explore slot) accumulate their
        n_a/K empirical mass naturally — that is the dedup: identical candidates
        are counted, not dropped (dropping would break the fixed-K static shape
        AND the visit accounting).

        CAVEAT (β̂ interpretation): unlike the i.i.d. schemes, the K slots here
        are a MIXTURE (deterministic argmax + tempered-π samples + dither), so
        the per-candidate ``beta_hat`` returned by ``_empirical_prior`` is a
        biased mixture frequency, NOT an unbiased estimate of the factored
        policy π. The mctx visit mechanics are still correct (uniform-1/K slot
        prior), but a Stage-3 factored-policy target built on ``beta_hat`` under
        this scheme would be mis-specified — read B territory, out of scope here.

        Returns (B, K, n_assets) int32 in {0..A-1}.
        """
        B = policy_logits.shape[0]
        n_assets, A = self.n_assets, self.n_actions_per_asset
        K, N = self.num_samples, self.explore_slots

        logits = policy_logits.reshape(B, n_assets, A)
        # TASK_854 (D21): scale the learned prior BEFORE masking, matching _build_proposal
        # (static guard ⇒ 1.0 = bit-identical). At 0.0 both the argmax slot and the tempered
        # samples below draw from the masked-uniform marginals — the learned prior no longer
        # steers the policy-prior candidate set. (The N explore slots reuse _build_proposal,
        # which applies the same scaling — so the whole policy-prior set is gated coherently.)
        if self.prior_weight != 1.0:
            logits = self.prior_weight * logits
        if action_mask is not None:
            mask_bool = action_mask.reshape(B, n_assets, A).astype(bool)
            logits = jnp.where(mask_bool, logits, -1e9)

        # slot 0 — per-asset argmax of the (masked) policy marginals.
        argmax_vec = jnp.argmax(logits, axis=-1).astype(jnp.int32)     # (B, n_assets)
        slots = [argmax_vec[:, None, :]]                               # (B, 1, n_assets)

        # slots 1..K-1-N — i.i.d. per-asset draws at temperature.
        n_sampled = K - 1 - N
        if n_sampled > 0:
            temp = max(float(self.proposal_temp), 1e-6)
            s = jax.random.categorical(
                sample_rng, logits / temp, axis=-1,
                shape=(n_sampled, B, n_assets))
            slots.append(jnp.transpose(s, (1, 0, 2)).astype(jnp.int32))

        # slots K-N..K-1 — exploration via the dithered-book machinery (reused
        # unconditionally, independent of self.proposal_scheme).
        if N > 0:
            prop_rng, samp_rng = jax.random.split(dither_rng)
            explore_proposal = self._build_proposal(
                policy_logits, action_mask, book, prop_rng, scheme='dithered-book')
            e = jax.random.categorical(
                samp_rng, explore_proposal, axis=-1,
                shape=(N, B, n_assets))
            slots.append(jnp.transpose(e, (1, 0, 2)).astype(jnp.int32))

        return jnp.concatenate(slots, axis=1)                         # (B, K, n_assets)

    def build_root_candidates(
        self,
        policy_logits: jnp.ndarray,
        action_mask: jnp.ndarray = None,
        book: jnp.ndarray = None,
        rng: chex.PRNGKey = None,
    ) -> jnp.ndarray:
        """TASK_847 test/diagnostic hook: build the root K-candidate set straight
        from policy logits (no model inference), honoring ``proposal_scheme``.

        For 'policy-prior' this is exactly the in-search generator; for every
        other scheme it mirrors the in-search sample path. Not on the jitted
        search path — used by the selftest candidate-set property test and for
        offline inspection.
        """
        if rng is None:
            rng = jax.random.PRNGKey(0)
        sample_rng, dither_rng = jax.random.split(rng)
        if self.proposal_scheme == 'policy-prior':
            return self._policy_prior_candidates(
                policy_logits, action_mask, book, sample_rng, dither_rng)
        proposal = self._build_proposal(policy_logits, action_mask, book, dither_rng)
        samples = jax.random.categorical(
            sample_rng, proposal, axis=-1,
            shape=(self.num_samples, policy_logits.shape[0], self.n_assets))
        return jnp.transpose(samples, (1, 0, 2)).astype(jnp.int32)

    def _search(
        self,
        params: dict,
        rng: chex.PRNGKey,
        observation: jnp.ndarray,
        action_mask: jnp.ndarray = None,
        book: jnp.ndarray = None,
        deterministic: bool = False,
        # TASK_851: true-env leaf only. None (canon default everywhere) ⇒ zero-leaf-pytree
        # args, no tensors added to the trace — the 'model' branch is bit-identical.
        env_state=None,
        prices: jnp.ndarray = None,
        env_config=None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
        """Run Sampled-MuZero vector search.

        Args:
            action_mask: (B, n_assets*A) legal-action mask (asset-major), or None.
            book: (B, n_assets) int per-asset maintain-action (v1: HOLD=1), or None.
            deterministic: True → select the applied vector via visit-argmax
                (eval); False → the gumbel-selected candidate (exploration).
            env_state, prices, env_config: TASK_851 — required iff
                ``self.leaf_eval == 'true-env'`` (the true-env leaf needs to step a copy of
                the real env forward). Ignored by the 'model' leaf (default, canon).

        Returns:
            action_vec:   (B, n_assets) int32 — selected candidate vector
            sampled_vecs: (B, K, n_assets) int32 — the K candidate vectors
            visit_probs:  (B, K) — visit distribution over the sampled set
            search_stats: dict
        """
        sample_rng, dither_rng, search_rng = jax.random.split(rng, 3)

        # 1. Root inference → factored policy, value, latent.
        root_out = self.model_apply_fn(
            {'params': params}, observation, method='initial_inference'
        )
        policy_logits = root_out['policy_logits']            # (B, n_assets*A)
        latent = root_out['latent_state']                    # (B, state_dim)
        root_value = self._categorical_to_scalar(root_out['value'])  # (B,)

        # 2. Build the K candidate action-vectors. Static string dispatch on
        #    ``proposal_scheme`` (captured at trace ⇒ the untaken branch is never
        #    traced): the non-'policy-prior' branch is the shipped Stage-2/3 code
        #    VERBATIM (bit-identity pinned by tools/tests/test_pernode_parity.py),
        #    'policy-prior' (TASK_847) builds the structured argmax+samples+explore
        #    candidate set so the learned prior is representable in the search.
        if self.proposal_scheme == 'policy-prior':
            sampled_vecs = self._policy_prior_candidates(
                policy_logits, action_mask, book, sample_rng, dither_rng)  # (B,K,A_n)
        else:
            # Build the dithered-book proposal and sample K candidate vectors from it.
            proposal = self._build_proposal(policy_logits, action_mask, book, dither_rng)
            samples = jax.random.categorical(
                sample_rng, proposal, axis=-1,
                shape=(self.num_samples, observation.shape[0], self.n_assets),
            )
            sampled_vecs = jnp.transpose(samples, (1, 0, 2)).astype(jnp.int32)  # (B,K,A_n)

        # 3. Sampled-MuZero corrected prior over the K slots (uniform 1/K).
        prior_logits, beta_hat = self._empirical_prior(sampled_vecs)  # (B,K),(B,K)

        B = observation.shape[0]

        # TASK_851: static dispatch on ``leaf_eval`` (Python str captured at construction ⇒
        # decided at trace time; the untaken branch is never traced). 'true-env' takes
        # priority — it's mutually exclusive with per_node_resampling (enforced in
        # __init__) and flattens the search to depth 1 (module docstring TASK_851 section).
        # The 'model' branches below (TASK_838's two sub-branches) are UNCHANGED.
        if self.leaf_eval == 'true-env':
            if env_state is None or prices is None or env_config is None:
                raise ValueError(
                    "leaf_eval='true-env' requires env_state, prices, and env_config (the "
                    "true-env leaf steps a copy of the real env forward — TASK_851); got "
                    f"env_state={'set' if env_state is not None else None}, "
                    f"prices={'set' if prices is not None else None}, "
                    f"env_config={'set' if env_config is not None else None}.")
            leaf_reward, leaf_value = self._true_env_leaf_values(
                params, env_state, prices, env_config, latent, sampled_vecs)
            root = mctx.RootFnOutput(
                prior_logits=prior_logits,
                value=root_value,
                embedding=latent,
            )

            def recurrent_fn(params_, rng_, action, embedding):
                # Pure table lookup: the K candidates' (reward, value) were fully
                # precomputed above (root-planning, TASK_851) — no per-node model/env
                # calls here. ``embedding`` is passed through unread (max_depth=1 below
                # means it is never consumed by a deeper expansion).
                idx = action.astype(jnp.int32)[:, None]                       # (B,1)
                reward = jnp.take_along_axis(leaf_reward, idx, axis=1)[:, 0]  # (B,)
                value = jnp.take_along_axis(leaf_value, idx, axis=1)[:, 0]    # (B,)
                interior_prior = jnp.zeros((B, self.num_samples), dtype=jnp.float32)
                recurrent_output = mctx.RecurrentFnOutput(
                    reward=reward,
                    discount=jnp.full_like(reward, self.discount),
                    prior_logits=interior_prior,
                    value=value,
                )
                return recurrent_output, embedding
        elif self.per_node_resampling:
            # ON: the mctx embedding pytree carries, per node, the latent AND
            # the K vectors sampled AT that node, so slot k at a node maps to
            # that node's own k-th candidate (at the root: the root's samples).
            root = mctx.RootFnOutput(
                prior_logits=prior_logits,
                value=root_value,
                embedding={'latent': latent, 'cand_vecs': sampled_vecs},
            )

            # GUARD (TASK_838): per_node_resampling correctness requires
            # max_depth unset/== num_simulations on the mctx call below — a
            # smaller max_depth makes mctx RE-EXPAND already-visited nodes at
            # the depth cutoff, which would re-roll that node's cand_vecs while
            # its children's slot statistics still refer to the OLD samples.
            def recurrent_fn(params_, rng_, action, embedding):
                # Gather the chosen vector from the PARENT node's own set.
                idx = action.astype(jnp.int32)[:, None, None]      # (B,1,1)
                vec = jnp.take_along_axis(
                    embedding['cand_vecs'], idx, axis=1)[:, 0, :]  # (B,A_n)
                out = self.model_apply_fn(
                    {'params': params_}, embedding['latent'], vec,
                    mode='recurrent_inference_vector'
                )
                reward = self._reward_to_scalar(out['reward'])
                value = self._categorical_to_scalar(out['value'])
                # Resample K vectors from the factored policy at the CHILD's
                # latent — same dithered-book machinery as the root; interior
                # book = HOLD-book, no interior mask (module docstring).
                # Deterministic per-node rng: mctx hands recurrent_fn a fresh
                # expand rng per simulation; fold_in(·, 838) namespaces it,
                # then split → (dither, sample). Seed-reproducible, and the
                # top-level rng stream is untouched.
                node_rng = jax.random.fold_in(rng_, 838)
                node_dither_rng, node_sample_rng = jax.random.split(node_rng)
                child_proposal = self._build_proposal(
                    out['policy_logits'], None, None, node_dither_rng)
                child_samples = jax.random.categorical(
                    node_sample_rng, child_proposal, axis=-1,
                    shape=(self.num_samples, B, self.n_assets),
                )
                child_vecs = jnp.transpose(
                    child_samples, (1, 0, 2)).astype(jnp.int32)    # (B,K,A_n)
                # β=π holds PER NODE: the child's K slots are drawn i.i.d.
                # from the child's OWN proposal, so the Sampled-MuZero
                # sample-based prior β̂(a)=n_a/K is again the uniform-1/K slot
                # prior at this node (Hubert et al. 2021, arXiv:2104.06303 §3
                # — the correction is local to the node that sampled).
                interior_prior = jnp.zeros((B, self.num_samples), dtype=jnp.float32)
                recurrent_output = mctx.RecurrentFnOutput(
                    reward=reward,
                    discount=jnp.full_like(reward, self.discount),
                    prior_logits=interior_prior,
                    value=value,
                )
                return recurrent_output, {'latent': out['latent_state'],
                                          'cand_vecs': child_vecs}
        else:
            root = mctx.RootFnOutput(
                prior_logits=prior_logits,
                value=root_value,
                embedding=latent,
            )

            def recurrent_fn(params_, rng_, action, embedding):
                # action: (B,) index into the K sampled candidates. Gather each
                # batch element's chosen action-VECTOR and expand via the vector
                # dynamics path (embeds the vector, not a scalar index).
                idx = action.astype(jnp.int32)[:, None, None]          # (B,1,1)
                vec = jnp.take_along_axis(sampled_vecs, idx, axis=1)[:, 0, :]  # (B,A_n)
                out = self.model_apply_fn(
                    {'params': params_}, embedding, vec,
                    mode='recurrent_inference_vector'
                )
                reward = self._reward_to_scalar(out['reward'])
                value = self._categorical_to_scalar(out['value'])
                # Interior prior over the (reused) K sampled slots = same empirical
                # uniform prior (documented simplification — see module docstring).
                interior_prior = jnp.zeros((B, self.num_samples), dtype=jnp.float32)
                recurrent_output = mctx.RecurrentFnOutput(
                    reward=reward,
                    discount=jnp.full_like(reward, self.discount),
                    prior_logits=interior_prior,
                    value=value,
                )
                return recurrent_output, out['latent_state']

        # 4. Gumbel-MuZero search / Sequential Halving over the K candidates —
        #    the same gumbel + qtransform machinery as the canon scalar search.
        #    GUARD (TASK_838): per_node_resampling correctness requires max_depth
        #    unset/== num_simulations; a smaller max_depth re-expands visited
        #    nodes and orphans their children's slot statistics (today: no
        #    max_depth is passed for that branch → mctx defaults it to
        #    num_simulations — safe).
        #    TASK_851: leaf_eval='true-env' passes max_depth=1 DELIBERATELY (module
        #    docstring) — the K candidates' values are precomputed root-planning
        #    (path-independent), so mctx re-EXPANDS (re-evaluates) an already-visited
        #    depth-1 node past the cutoff rather than compounding into a spurious
        #    depth-2 node (mctx `_src/search.py`: "a leaf can be expanded multiple
        #    times" under max_depth — never a correctness issue, since our
        #    recurrent_fn is a pure, embedding-independent table lookup).
        policy_output = mctx.gumbel_muzero_policy(
            params=params,
            rng_key=search_rng,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=self.num_simulations,
            max_num_considered_actions=self.max_num_considered_actions,
            qtransform=mctx.qtransform_by_parent_and_siblings,
            gumbel_scale=self.gumbel_scale,
            max_depth=(1 if self.leaf_eval == 'true-env' else None),
        )

        gumbel_idx = policy_output.action                    # (B,) exploration pick
        visit_probs = policy_output.action_weights           # (B, K)

        # deterministic=True (eval) → select via visit-count argmax (carry-note 3:
        # the default gumbel-selected action is exploration-dominated). Traced bool
        # → jnp.where (no lax.cond in a jitted/scan context).
        argmax_idx = jnp.argmax(visit_probs, axis=-1).astype(gumbel_idx.dtype)  # (B,)
        selected_idx = jnp.where(deterministic, argmax_idx, gumbel_idx)         # (B,)

        # 5. Map the selected candidate index back to its action VECTOR.
        sel = selected_idx.astype(jnp.int32)[:, None, None]  # (B,1,1)
        action_vec = jnp.take_along_axis(sampled_vecs, sel, axis=1)[:, 0, :]  # (B,A_n)

        search_stats = {
            'root_value': root_value,
            'beta_hat': beta_hat,
            'selected_idx': selected_idx,
            'visit_entropy': -jnp.sum(
                visit_probs * jnp.log(jnp.maximum(visit_probs, 1e-10)), axis=-1),
        }
        if self.per_node_resampling:
            # TASK_838 ON-only diagnostics (static branch ⇒ the OFF stats dict
            # is byte-identical to canon): per-node candidate sets, per-node
            # slot priors and visit counts straight from the search tree.
            tree = policy_output.search_tree
            search_stats['tree_cand_vecs'] = tree.embeddings['cand_vecs']      # (B,N,K,A_n)
            search_stats['tree_children_prior_logits'] = tree.children_prior_logits  # (B,N,K)
            search_stats['tree_node_visits'] = tree.node_visits                # (B,N)
        return action_vec, sampled_vecs, visit_probs, search_stats

    def __call__(
        self,
        params: dict,
        rng: chex.PRNGKey,
        observation: jnp.ndarray,
        action_mask: jnp.ndarray = None,
        book: jnp.ndarray = None,
        deterministic: bool = False,
        # TASK_851: true-env leaf only (see _search's docstring).
        env_state=None,
        prices: jnp.ndarray = None,
        env_config=None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
        """JIT-compiled Sampled-MuZero vector search."""
        return self.search_fn(params, rng, observation, action_mask, book, deterministic,
                              env_state=env_state, prices=prices, env_config=env_config)

    def raw_search(
        self,
        params: dict,
        rng: chex.PRNGKey,
        observation: jnp.ndarray,
        action_mask: jnp.ndarray = None,
        book: jnp.ndarray = None,
        deterministic: bool = False,
        # TASK_851: true-env leaf only (see _search's docstring).
        env_state=None,
        prices: jnp.ndarray = None,
        env_config=None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
        """Non-JIT search (for use inside an outer lax.scan / debugging)."""
        return self._search(params, rng, observation, action_mask, book, deterministic,
                            env_state=env_state, prices=prices, env_config=env_config)


def create_sampled_vector_search(model, **kwargs):
    """Factory: build a SampledMuZeroVectorSearch from a MuZeroNetworks model.

    The model MUST be built with use_vector_action=True. All constructor knobs
    (incl. TASK_838 ``per_node_resampling``, TASK_851 ``leaf_eval``/
    ``leaf_rollout_r``/``leaf_vhat_tail``, and TASK_854 ``prior_weight``, all
    default OFF/canon) pass through **kwargs.
    """
    return SampledMuZeroVectorSearch(model_apply_fn=model.apply, **kwargs)
