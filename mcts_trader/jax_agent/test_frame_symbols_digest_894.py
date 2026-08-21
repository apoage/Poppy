#!/usr/bin/env python3
"""TASK_894 — unit suite for the SAVE-TIME half of the frame_symbols digest: the checkpoint_io
primitive (`frame_symbols_digest`) and `trainer.save_checkpoint`'s new `frame_symbols` argument.

Separate from `test_checkpoint_io.py` (TASK_888/889's own suite, the checkpoint LOAD path) and
from `experiments/835_stage4_eval/test_frame_symbols_digest_894.py` (the CERTIFICATION-side
suite: `_certification_verdict`'s new conjunct, the retro-stamp tool, and the end-to-end
reversed-frame_symbols forgery through imprint.py + eval_vector.py). This file only exercises
what a checkpoint SAVE writes.

Gates covered here:
  (digest) `checkpoint_io.frame_symbols_digest` is order-sensitive (a reversed list digests
      differently) and deterministic (the same ordered list digests identically every time,
      including across a fresh Python list built the same way).
  (save-absent) `trainer.save_checkpoint` with NO `frame_symbols` argument (the default) writes
      NO `frame_symbols_sha256` key into EITHER `trainer_metadata.json` or the per-leaf
      `trainer_metadata_<step>.json` — never `null`, absent — and is BYTE-IDENTICAL to a call
      that passes `frame_symbols=None` explicitly (acceptance ii's "no pinned frame at save
      time" case, proven directly at the save site rather than inferred).
  (save-present) `trainer.save_checkpoint(frame_symbols=[...])` writes
      `checkpoint_io.frame_symbols_digest([...])` under `checkpoint_io.FRAME_SYMBOLS_SHA256_FIELD`
      into BOTH sidecars, alongside (not replacing) every architecture field save_checkpoint
      already wrote.
  (save-order) the SAME symbol set in a DIFFERENT order writes a DIFFERENT recorded digest.

CPU-only, no GPU, no `trading_data.db`, no fixture DB — every checkpoint here is a throwaway
tiny model saved to a tempdir and removed at the end.

Run: ``python mcts_trader/jax_agent/test_frame_symbols_digest_894.py`` (exit 0 = PASS).
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import json
import shutil
import sys
import tempfile

import jax
import jax.numpy as jnp

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from mcts_trader.training.config import TrainingConfig
from mcts_trader.training.setup import setup_model_and_trainer
from mcts_trader.jax_agent import checkpoint_io as cio

failures = []


def _check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        failures.append(f"{name}: {detail}")


def _tiny_trainer(n_assets=2, seed=0):
    """Mirrors test_checkpoint_io.py's `_tiny_trainer` exactly — the smallest trainer this
    program's own construction path will build. Plumbing tests only, no training/search."""
    cfg = TrainingConfig(name="test_894_digest", model_size="small", vector_action=True,
                         seed=seed)
    model, trainer = setup_model_and_trainer(cfg, n_assets)
    dummy_obs = jnp.ones((1, 27, n_assets, n_assets))
    trainer.init_train_state(jax.random.PRNGKey(seed), dummy_obs)
    return trainer


FRAME_A = ["XBT/USD", "ETH/USD", "LTC/USD", "XMR/USD"]
FRAME_A_REVERSED = list(reversed(FRAME_A))


def test_digest_order_sensitive_and_deterministic():
    d1 = cio.frame_symbols_digest(FRAME_A)
    d2 = cio.frame_symbols_digest(list(FRAME_A))          # a FRESH list, same order
    d_rev = cio.frame_symbols_digest(FRAME_A_REVERSED)
    _check("digest is deterministic across independently-built equal-order lists", d1 == d2,
          f"{d1!r} != {d2!r}")
    _check("digest is ORDER-SENSITIVE (reversed list digests differently)", d1 != d_rev,
          f"reversed list produced the SAME digest {d1!r} — the whole point of this field is "
          f"to protect ORDER, which param_fingerprint cannot see")
    _check("digest is a 64-char lowercase hex sha256", isinstance(d1, str) and len(d1) == 64
          and all(c in "0123456789abcdef" for c in d1), repr(d1))


def _leaf_dir(run_dir):
    leaves = [d for d in os.listdir(run_dir) if d.startswith("checkpoint_")]
    assert len(leaves) == 1, f"expected exactly one checkpoint_* leaf in {run_dir}: {leaves}"
    return os.path.join(run_dir, leaves[0])


def test_save_absent_is_byte_identical_and_omits_key():
    """acceptance (ii)'s save-site half: no pinned frame at save time => the key is ABSENT
    (never null), and NOT passing the argument at all is indistinguishable from passing
    `frame_symbols=None` explicitly — the default a pre-894 caller (train_jax_fast.py, distill.py,
    every probe script) keeps using unconditionally."""
    work = tempfile.mkdtemp(prefix="t894_save_absent_")
    try:
        trainer_default = _tiny_trainer(seed=1)
        trainer_explicit = _tiny_trainer(seed=1)
        ckpt_default = os.path.join(work, "ckpt_default")
        ckpt_explicit = os.path.join(work, "ckpt_explicit")
        trainer_default.save_checkpoint(ckpt_default, overwrite=True)
        trainer_explicit.save_checkpoint(ckpt_explicit, overwrite=True, frame_symbols=None)
        with open(os.path.join(ckpt_default, "trainer_metadata.json")) as fh:
            meta_default = json.load(fh)
        with open(os.path.join(ckpt_explicit, "trainer_metadata.json")) as fh:
            meta_explicit = json.load(fh)
        _check("no frame_symbols_sha256 key when the argument is omitted",
              cio.FRAME_SYMBOLS_SHA256_FIELD not in meta_default, repr(meta_default))
        _check("no frame_symbols_sha256 key when frame_symbols=None is passed explicitly",
              cio.FRAME_SYMBOLS_SHA256_FIELD not in meta_explicit, repr(meta_explicit))
        _check("omitted vs explicit-None metadata documents are IDENTICAL (byte-for-byte, "
              "modulo the unrelated param_fingerprint/step_count of two independently-built "
              "trainers)",
              {k: v for k, v in meta_default.items() if k != "param_fingerprint"}
              == {k: v for k, v in meta_explicit.items() if k != "param_fingerprint"},
              f"{meta_default}\nvs\n{meta_explicit}")
        # The per-leaf sidecar carries the SAME record (trainer.py writes one `metadata` dict to
        # both files) — checked once, on the default-argument save.
        leaf_meta_path = cio.leaf_metadata_path_for(ckpt_default, trainer_default.state.step_count)
        with open(leaf_meta_path) as fh:
            leaf_meta = json.load(fh)
        _check("no frame_symbols_sha256 key in the per-leaf trainer_metadata_<step>.json either",
              cio.FRAME_SYMBOLS_SHA256_FIELD not in leaf_meta, repr(leaf_meta))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_save_present_writes_correct_digest_in_both_sidecars():
    """acceptance (i)/(ii)'s save-site positive half: a pinned frame at save time writes the
    EXACT digest `checkpoint_io.frame_symbols_digest` computes, in BOTH the shared and per-leaf
    sidecars, alongside (not instead of) the architecture fields save_checkpoint already wrote."""
    work = tempfile.mkdtemp(prefix="t894_save_present_")
    try:
        trainer = _tiny_trainer(seed=2)
        ckpt = os.path.join(work, "ckpt_pinned")
        trainer.save_checkpoint(ckpt, overwrite=True, frame_symbols=FRAME_A)
        expect = cio.frame_symbols_digest(FRAME_A)
        with open(os.path.join(ckpt, "trainer_metadata.json")) as fh:
            meta = json.load(fh)
        leaf_meta_path = cio.leaf_metadata_path_for(ckpt, trainer.state.step_count)
        with open(leaf_meta_path) as fh:
            leaf_meta = json.load(fh)
        _check("trainer_metadata.json records the exact expected digest",
              meta.get(cio.FRAME_SYMBOLS_SHA256_FIELD) == expect,
              f"{meta.get(cio.FRAME_SYMBOLS_SHA256_FIELD)!r} != {expect!r}")
        _check("per-leaf trainer_metadata_<step>.json records the SAME digest",
              leaf_meta.get(cio.FRAME_SYMBOLS_SHA256_FIELD) == expect,
              f"{leaf_meta.get(cio.FRAME_SYMBOLS_SHA256_FIELD)!r} != {expect!r}")
        # Alongside, not instead of: every architecture field from BEFORE this task still there.
        _check("architecture fields (h_cycles/l_cycles/state_dim/vector_action/param_fingerprint)"
              " are all still present",
              all(k in meta for k in ("h_cycles", "l_cycles", "state_dim", "vector_action",
                                      "param_fingerprint")),
              repr(meta))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_save_reversed_order_writes_different_digest():
    """The SAME symbol SET in reversed order records a DIFFERENT frame_symbols_sha256 — proof
    that the save site is genuinely order-sensitive, not merely set-membership-sensitive."""
    work = tempfile.mkdtemp(prefix="t894_save_reversed_")
    try:
        trainer_a = _tiny_trainer(seed=3)
        trainer_b = _tiny_trainer(seed=3)
        ckpt_a = os.path.join(work, "ckpt_a")
        ckpt_b = os.path.join(work, "ckpt_b")
        trainer_a.save_checkpoint(ckpt_a, overwrite=True, frame_symbols=FRAME_A)
        trainer_b.save_checkpoint(ckpt_b, overwrite=True, frame_symbols=FRAME_A_REVERSED)
        with open(os.path.join(ckpt_a, "trainer_metadata.json")) as fh:
            sha_a = json.load(fh)[cio.FRAME_SYMBOLS_SHA256_FIELD]
        with open(os.path.join(ckpt_b, "trainer_metadata.json")) as fh:
            sha_b = json.load(fh)[cio.FRAME_SYMBOLS_SHA256_FIELD]
        _check("reversed-order save records a DIFFERENT frame_symbols_sha256", sha_a != sha_b,
              f"both recorded {sha_a!r}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    tests = [test_digest_order_sensitive_and_deterministic,
            test_save_absent_is_byte_identical_and_omits_key,
            test_save_present_writes_correct_digest_in_both_sidecars,
            test_save_reversed_order_writes_different_digest]
    for t in tests:
        print(f"[894] {t.__name__} ...")
        t()
    print("=" * 84)
    if failures:
        print(f"[894] FAIL — {len(failures)} failure(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[894] ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
