"""TASK_876 — WRITE-SIDE behaviour this row ships.

Scope note (deliberate): TASK_876 ships the score EMITTER and two storage-policy defaults,
with ZERO deleter surface. ``tools/ckpt_gc.py`` is byte-identical to HEAD and is NOT under
test here. The shipped GC is epoch-keyed and cannot resolve ``checkpoint_<step>`` or
``checkpoint_epoch_<N>`` rows; making it path-first is TASK_881. See
``append_checkpoint_score``'s docstring — these tests pin the WRITER, not the reader.

Covered:
  1. emission is unconditional and APPENDS across a simulated restart (never truncates)
  2. None / NaN / Inf -> ``score: null`` + ``score_invalid``; a real ``0.0`` stays ``0.0``
  3. ``--ckpt-retention score`` REFUSES; ``recency`` is the default
  4. ``save_checkpoint`` defaults to ``overwrite=False``; ``_verify_checkpoint_on_disk``
     raises ``CheckpointVanishedError`` on a vanished / empty / zero-byte checkpoint
  5. FIX-1: ``_attempt_checkpoint_save`` swallows ONLY the three designed outcomes; any
     unrecognised exception propagates so the run exits nonzero
"""

import ast
import importlib
import inspect
import json
import math
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from mcts_trader.training import safe_checkpoint as sc  # noqa: E402


def _rows(run_dir):
    with open(os.path.join(str(run_dir), "scores.jsonl")) as fh:
        return [json.loads(l) for l in fh if l.strip()]


# ---------------------------------------------------------------- 1. unconditional append

def test_emission_is_unconditional_and_appends_across_restart(tmp_path):
    """Every save gets a row, and a restarted process appends instead of truncating."""
    run = tmp_path / "run"

    # run dir does not exist yet -> the emitter must still produce a row (unconditional)
    sc.append_checkpoint_score(run, 1, 0.5, "m", path=str(run / "snapshot_ep1"), step=100)
    sc.append_checkpoint_score(run, 2, -0.25, "m", path=str(run / "snapshot_ep2"), step=200)
    assert len(_rows(run)) == 2

    first_session = _rows(run)[0]["session"]

    # Simulate a RESTART: a fresh import of the module mints a new process session id and
    # must open scores.jsonl in append mode, not truncate it.
    sc2 = importlib.reload(sc)
    assert sc2._RUN_SESSION_ID != first_session, "reload must mint a new session id"
    sc2.append_checkpoint_score(run, 1, 0.75, "m", path=str(run / "snapshot_ep1"), step=100)

    rows = _rows(run)
    assert len(rows) == 3, "restart truncated scores.jsonl instead of appending"
    assert [r["epoch"] for r in rows] == [1, 2, 1]
    assert rows[0]["score"] == 0.5 and rows[1]["score"] == -0.25
    assert rows[0]["session"] == first_session
    assert rows[2]["session"] != first_session

    # path/step are recorded (the forward-compatible fields TASK_881's reader will use)
    assert rows[0]["path"] == os.path.abspath(str(run / "snapshot_ep1"))
    assert rows[0]["step"] == 100

    importlib.reload(sc)  # leave the module in a clean state for the other tests


def test_every_save_site_in_train_jax_fast_emits_a_row():
    """Guard the 'unconditional at EVERY save' claim at the call sites.

    Counts ``append_checkpoint_score`` calls in train_jax_fast.py: SafeCheckpointManager
    epoch dir + its golden sibling (FIX-2), periodic, snapshot, final = 5.
    """
    src = open(os.path.join(REPO, "train_jax_fast.py")).read()
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "append_checkpoint_score"]
    assert len(calls) == 5, (
        f"expected 5 emission sites (safe-epoch, safe-GOLDEN, periodic, snapshot, final), "
        f"found {len(calls)} — a save path lost its score row")
    # FIX-2: the golden checkpoint must be one of them.
    assert "golden_checkpoint" in src and "golden checkpoint (first SafeCheckpointManager save" in src


# ------------------------------------------------------- 2. invalid scores are not fabricated

@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), float("-inf")])
def test_invalid_score_becomes_null_plus_score_invalid(tmp_path, bad):
    run = tmp_path / "run"
    sc.append_checkpoint_score(run, 7, bad, "m", path=str(run / "snapshot_ep7"), step=7)
    row = _rows(run)[0]
    assert row["score"] is None, f"{bad!r} must not be written as a usable score"
    assert "score_invalid" in row, f"{bad!r} must be recorded as invalid for the audit trail"
    assert row["epoch"] == 7


def test_real_zero_is_kept_as_zero(tmp_path):
    """0.0 is a MEASUREMENT, not a missing value — it must survive as 0.0."""
    run = tmp_path / "run"
    sc.append_checkpoint_score(run, 3, 0.0, "m", path=str(run / "snapshot_ep3"), step=3)
    row = _rows(run)[0]
    assert row["score"] == 0.0
    assert isinstance(row["score"], float)
    assert row["score"] is not None
    assert "score_invalid" not in row


# ------------------------------------------------------------------ 3. --ckpt-retention

def test_ckpt_retention_defaults_to_recency():
    from mcts_trader.training.config import TrainingConfig
    assert TrainingConfig.ckpt_retention == 'recency'

    from mcts_trader.training.cli import parse_args
    old = sys.argv
    try:
        sys.argv = ['train_jax_fast.py']
        assert parse_args().ckpt_retention == 'recency'
    finally:
        sys.argv = old


def test_ckpt_retention_score_refuses():
    """'score' has no implementation in this build (TASK_881) — it must FAIL LOUDLY."""
    from mcts_trader.training.cli import parse_args
    old = sys.argv
    try:
        sys.argv = ['train_jax_fast.py', '--ckpt-retention', 'score']
        with pytest.raises(SystemExit) as ei:
            parse_args()
        assert ei.value.code != 0
    finally:
        sys.argv = old


# ------------------------------------------------- 4. save_checkpoint defaults + verification

def test_save_checkpoint_defaults_to_overwrite_false():
    from mcts_trader.jax_agent.trainer import MuZeroTrainer
    sig = inspect.signature(MuZeroTrainer.save_checkpoint)
    assert sig.parameters['overwrite'].default is False, (
        "overwrite=True on the training save path mass-deletes every checkpoint NEWER than "
        "the one being written (flax _remove_invalid_ckpts) — the D32 ruling is False")
    assert sig.parameters['keep'].default == 3


def test_verify_checkpoint_on_disk_rejects_vanished_empty_and_zero_byte(tmp_path):
    from mcts_trader.jax_agent.trainer import (
        CheckpointVanishedError, _verify_checkpoint_on_disk)

    with pytest.raises(CheckpointVanishedError):
        _verify_checkpoint_on_disk(None, str(tmp_path), 5)

    missing = tmp_path / "checkpoint_5"                      # flax returned it, never wrote it
    with pytest.raises(CheckpointVanishedError):
        _verify_checkpoint_on_disk(str(missing), str(tmp_path), 5)

    empty_dir = tmp_path / "checkpoint_6"
    empty_dir.mkdir()
    with pytest.raises(CheckpointVanishedError):
        _verify_checkpoint_on_disk(str(empty_dir), str(tmp_path), 6)

    zero_file = tmp_path / "checkpoint_7"
    zero_file.write_bytes(b"")
    with pytest.raises(CheckpointVanishedError):
        _verify_checkpoint_on_disk(str(zero_file), str(tmp_path), 7)

    # the happy paths must still pass through unchanged
    good_dir = tmp_path / "checkpoint_8"
    good_dir.mkdir()
    (good_dir / "blob").write_bytes(b"x")
    assert _verify_checkpoint_on_disk(str(good_dir), str(tmp_path), 8) == str(good_dir)

    good_file = tmp_path / "checkpoint_9"
    good_file.write_bytes(b"x")
    assert _verify_checkpoint_on_disk(str(good_file), str(tmp_path), 9) == str(good_file)


# ----------------------------------------------------- 5. FIX-1: no swallowing of the unknown

class _Boom:
    """Minimal trainer stand-in: save_checkpoint raises whatever it was handed."""

    def __init__(self, exc=None, returns=None):
        self._exc, self._returns = exc, returns
        self.calls = 0

    def save_checkpoint(self, path, keep=3):
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        return self._returns


def _attempt():
    import train_jax_fast
    return train_jax_fast._attempt_checkpoint_save


@pytest.mark.parametrize("exc", [
    OSError(28, "No space left on device"),
    PermissionError(13, "Permission denied"),
    RuntimeError("orbax internal failure"),
    ValueError("some unrelated value error"),
])
def test_unrecognized_save_exception_is_not_swallowed(exc):
    """The whole point of FIX-1: dojo_manager gates on returncode, so an unknown failure
    must take the run down instead of being logged and reported as a success."""
    with pytest.raises(type(exc)):
        _attempt()(_Boom(exc=exc), "/tmp/does-not-matter", 3, 'PERIODIC SAVE', 1,
                   "/tmp/does-not-matter")


def test_step_collision_is_logged_and_continues():
    written, msgs = _attempt()(
        _Boom(exc=ValueError("Destination /x/checkpoint_5 already exists")),
        "/x", 3, 'PERIODIC SAVE', 1, "/x")
    assert written is None
    assert msgs and 'REFUSED' in msgs[0]


def test_vanished_checkpoint_is_logged_and_continues():
    from mcts_trader.jax_agent.trainer import CheckpointVanishedError
    written, msgs = _attempt()(
        _Boom(exc=CheckpointVanishedError("/x/checkpoint_5", "/x", 5, "gone")),
        "/x", 3, 'PERIODIC SAVE', 1, "/x")
    assert written is None
    assert msgs and 'VANISHED' in msgs[0]


def test_sidecar_failure_still_scores_the_real_checkpoint():
    from mcts_trader.jax_agent.trainer import CheckpointSidecarError
    written, msgs = _attempt()(
        _Boom(exc=CheckpointSidecarError("/x/checkpoint_5", "popart_state.npz", OSError("ENOSPC"))),
        "/x", 3, 'PERIODIC SAVE', 1, "/x")
    assert written == "/x/checkpoint_5", "a real on-disk checkpoint must still be scored"
    assert msgs and 'IS on disk' in msgs[0]


def test_successful_save_returns_the_written_path():
    written, msgs = _attempt()(_Boom(returns="/x/checkpoint_5"), "/x", 3, 'PERIODIC SAVE', 1, "/x")
    assert written == "/x/checkpoint_5"
    assert msgs == []


def test_final_save_failure_sets_a_nonzero_exit_instead_of_raising_from_finally():
    """FIX-1's other half, pinned structurally.

    The final save lives inside ``finally:`` where re-raising would REPLACE the exception
    that actually stopped training. So it records a flag and exits nonzero AFTER the
    try/finally. This asserts on ``main``'s source because the behaviour only manifests at
    process exit; the behavioural half (propagation) is covered by the tests above and the
    end-to-end read-only-dir repro is in the TASK_876 verification log.
    """
    import train_jax_fast
    src = inspect.getsource(train_jax_fast.main)
    assert 'final_ckpt_save_failed = False' in src, "the exit-code flag was removed"
    assert 'final_ckpt_save_failed = True' in src, "the finally-block handler no longer records"
    exit_block = 'if final_ckpt_save_failed:\n        sys.exit(1)'
    assert exit_block in src, "an unrecognised final-save failure no longer exits nonzero"
    # the exit must be OUTSIDE (after) the finally block, or it would clobber the real
    # exception that stopped training.
    assert src.index(exit_block) > src.index('    finally:')
    assert 'TRAINING FAILED: FINAL CHECKPOINT NOT SAVED' in src, (
        "a failed final save must not print TRAINING COMPLETE!")
