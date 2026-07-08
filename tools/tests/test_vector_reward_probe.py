"""TASK_828 Stage 3 §4c — E3 reward-vs-action probe HARNESS (CPU, UNTRAINED model).

Validates that the vector-action reward head is an INSTRUMENT that measures
d(reward)/d(action). On a fresh ``use_vector_action=True`` model:

  * one ``initial_inference`` → a fixed latent,
  * M distinct random action VECTORS pushed through ``recurrent_inference_vector``,
  * reward logits decoded → scalar rewards.

Asserts: all rewards finite; SENSITIVITY = variance of reward across M *distinct*
vectors is > 0 and dominates the CONTROL = variance across the *same* vector ×M
(≈0); and a single-slot flip (HOLD→BUY on one asset) moves the reward.

This proves the HARNESS measures action-sensitivity. It does NOT assert the real E3
DEC-1 reward-attribution threshold on a mid/late-training model — that is the
SCHEDULED GPU run, out of factory scope.

Run: JAX_PLATFORMS=cpu pytest tools/tests/test_vector_reward_probe.py -q
"""
import os
# CPU / fixture-only; never GPU in the loop, never trading_data.db.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import sys

import numpy as np
import jax
import jax.numpy as jnp

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_SEED = 0
_N_ASSETS = 64
_A = 5          # n_actions_per_asset (0=sell,1=hold,2=buy,3=limit_sell,4=limit_buy)
_M = 64         # number of probe action-vectors


def _build_untrained_vector_model():
    """Fresh use_vector_action=True model + freshly-initialised params (untrained)."""
    from mcts_trader.training.config import TrainingConfig
    from mcts_trader.training.setup import setup_model_and_trainer

    config = TrainingConfig(
        name="reward_probe", synthetic_level=0, seed=_SEED,
        parallel_envs=2, steps=16, epochs=1, sims=8, batch_size=32,
        model_size="small",
    )
    config.use_jax_env = True
    config.vector_action = True
    model, trainer = setup_model_and_trainer(config, _N_ASSETS)
    rng = jax.random.PRNGKey(_SEED)
    dummy_obs = jnp.ones((1, 27, _N_ASSETS, _N_ASSETS))
    trainer.state = trainer.init_train_state(rng, dummy_obs)
    assert model.use_vector_action, "model must be use_vector_action=True"
    return model, trainer.state.params


def _decode_reward(reward_logits):
    """Categorical reward logits (…, support_size) → scalar expected reward (…,)."""
    from mcts_trader.jax_agent import get_value_support
    support = get_value_support()
    probs = jax.nn.softmax(reward_logits, axis=-1)
    return jnp.sum(probs * support, axis=-1)


def run_probe():
    model, params = _build_untrained_vector_model()

    obs = jnp.ones((1, 27, _N_ASSETS, _N_ASSETS))
    init_out = model.apply({'params': params}, obs, mode='initial_inference')
    latent = init_out['latent_state']              # (1, state_dim) — FIXED latent
    latent_M = jnp.repeat(latent, _M, axis=0)      # (M, state_dim)

    # M DISTINCT random action vectors in {0..A-1}.
    rng = jax.random.PRNGKey(_SEED + 1)
    distinct = jax.random.randint(rng, (_M, _N_ASSETS), 0, _A).astype(jnp.int32)
    rec_d = model.apply({'params': params}, latent_M, distinct,
                        mode='recurrent_inference_vector')
    r_distinct = np.asarray(_decode_reward(rec_d['reward']))

    # CONTROL: the SAME vector ×M → reward variance must be ~0 (deterministic head).
    same = jnp.tile(distinct[0:1], (_M, 1))
    rec_c = model.apply({'params': params}, latent_M, same,
                        mode='recurrent_inference_vector')
    r_control = np.asarray(_decode_reward(rec_c['reward']))

    # Single-slot flip: all-HOLD base vs HOLD with asset-0 flipped to BUY.
    base = jnp.ones((1, _N_ASSETS), dtype=jnp.int32)       # HOLD=1 everywhere
    flipped = base.at[0, 0].set(2)                          # BUY=2 on asset 0
    r_base = float(np.asarray(_decode_reward(
        model.apply({'params': params}, latent, base,
                    mode='recurrent_inference_vector')['reward']))[0])
    r_flip = float(np.asarray(_decode_reward(
        model.apply({'params': params}, latent, flipped,
                    mode='recurrent_inference_vector')['reward']))[0])

    return {
        'r_distinct': r_distinct,
        'r_control': r_control,
        'sensitivity': float(np.var(r_distinct)),
        'control': float(np.var(r_control)),
        'r_base': r_base,
        'r_flip': r_flip,
    }


def test_reward_probe_measures_action_sensitivity():
    res = run_probe()
    assert np.all(np.isfinite(res['r_distinct'])), "distinct-vector rewards not finite"
    assert np.all(np.isfinite(res['r_control'])), "control rewards not finite"

    sens = res['sensitivity']
    ctrl = res['control']
    ratio = sens / max(ctrl, 1e-12)
    d_flip = abs(res['r_flip'] - res['r_base'])

    # Emit the numbers the gate wants recorded.
    print(f"\n[reward-probe] sensitivity_var={sens:.6e}  control_var={ctrl:.6e}  ratio={ratio:.6e}")
    print(f"[reward-probe] single-slot flip: r_base={res['r_base']:.6f}  "
          f"r_flip={res['r_flip']:.6f}  |delta|={d_flip:.6e}")

    # SENSITIVITY: reward genuinely varies with the action vector.
    assert sens > 0.0, "reward is action-INSENSITIVE (var over distinct vectors == 0)"
    # CONTROL must sit strictly below sensitivity, and sensitivity must DOMINATE it
    # by a wide margin (not an absolute threshold — that's the scheduled E3 run).
    assert ctrl < sens, f"control var ({ctrl:.3e}) not below sensitivity var ({sens:.3e})"
    assert ratio > 100.0, f"sensitivity does not dominate control (ratio={ratio:.3e})"
    # A single per-asset slot flip must move the reward (d(reward)/d(action) != 0).
    assert d_flip > 0.0, "single-slot flip (HOLD->BUY on one asset) did not move reward"


if __name__ == "__main__":
    r = run_probe()
    _ratio = r['sensitivity'] / max(r['control'], 1e-12)
    print(f"sensitivity_var={r['sensitivity']:.6e}  control_var={r['control']:.6e}  ratio={_ratio:.6e}")
    print(f"single-slot flip |delta|={abs(r['r_flip'] - r['r_base']):.6e}")
    print("PASS" if (r['sensitivity'] > 0 and _ratio > 100.0) else "FAIL")
