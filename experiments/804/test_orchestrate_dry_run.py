#!/usr/bin/env python3
"""TASK_804-B4 — dry-run audit tests for the token-arms orchestrator.

These tests ARE the acceptance gate.  They run the FULL queue with stub
commands (no GPU, no server, no tmux) and assert the five required
behaviors:

  1. 2-wide packing + seed-major order
  2. watchdog firing (stale-inject -> kill -> queue recovers)
  3. eval-on-land (each finished seed dispatches its eval before the end)
  4. per-run archival (config + deploy hash + vocab sha present)
  5. stop/extend (KILL path and EXTEND path in separate invocations)

Run from anywhere:  pytest experiments/804/test_orchestrate_dry_run.py -v
(each test chdir's into a fresh tmp dir so runs/804_token_arms is isolated).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orchestrate_token_arms as orch  # noqa: E402


# --------------------------------------------------------------------------- #
def _run(tmp_path, monkeypatch, extra_args):
    """chdir into an isolated tmp dir, run main() with the given args, return
    the runs root path."""
    monkeypatch.chdir(tmp_path)
    base = ["--dry-run", "--train-stub-sec", "1.0", "--eval-stub-sec", "0.5",
            "--tick-sec", "0.1"]
    rc = orch.main(base + extra_args)
    assert rc == 0
    return tmp_path / "runs" / "804_token_arms"


# --------------------------------------------------------------------------- #
def test_seed_major_queue_order():
    """Behavior 1a: queue is seed-major (raw_s7, real_s7, rand_s7, then s13...)."""
    o = orch.Orchestrator(orch.parse_args(["--dry-run"]))
    q = o.build_queue([7, 13, 23])
    tags = [r.tag for r in q]
    assert tags == [
        "raw_seed7", "real_fe_seed7", "rand_fe_seed7",
        "raw_seed13", "real_fe_seed13", "rand_fe_seed13",
        "raw_seed23", "real_fe_seed23", "rand_fe_seed23",
    ]


def test_two_wide_and_eval_on_land(tmp_path, monkeypatch, capsys):
    """Behavior 1: never >2 concurrent trains.  Behavior 3: eval-on-land —
    every train gets an eval dispatched, and all 9 evals land."""
    root = _run(tmp_path, monkeypatch, ["--scenario", "neutral"])
    out = capsys.readouterr().out

    # max concurrent trains never exceeded 2.
    assert "max concurrent trains: 2 (<=2 ok=True)" in out or \
           "max concurrent trains: 1" in out  # tiny stubs may serialize

    # eval-on-land: an EVAL-ON-LAND dispatch line per arm/seed, 9 total.
    dispatched = out.count("EVAL-ON-LAND dispatch")
    assert dispatched == 9, f"expected 9 eval dispatches, saw {dispatched}"
    landed = out.count("EVAL LANDED")
    assert landed == 9, f"expected 9 evals landed, saw {landed}"

    # all 9 train run dirs + 9 eval dirs exist.
    for seed in (7, 13, 23):
        for arm in ("raw", "real_fe", "rand_fe"):
            assert (root / f"{arm}_seed{seed}" / "phase2_stageB_best.pkl").exists()
            assert (root / f"eval_{arm}_seed{seed}" / "benchmark.json").exists()


def test_two_wide_invariant_strict(tmp_path, monkeypatch):
    """Behavior 1 (strict): instrument the orchestrator and assert the live
    concurrent-train count never exceeds 2 at any launch."""
    monkeypatch.chdir(tmp_path)
    args = orch.parse_args(["--dry-run", "--train-stub-sec", "1.0",
                            "--eval-stub-sec", "0.5", "--tick-sec", "0.05",
                            "--scenario", "neutral"])
    o = orch.Orchestrator(args)
    seen = []
    real_launch = o._launch_train

    def spy(run, gpu):
        real_launch(run, gpu)
        n = sum(1 for s in o.train_slots.values() if s is not None)
        seen.append(n)

    o._launch_train = spy  # type: ignore
    assert o.run() == 0
    assert max(seen) <= 2, f"2-wide violated: peaked at {max(seen)}"


def test_watchdog_fires_and_recovers(tmp_path, monkeypatch, capsys):
    """Behavior 2: inject a stale log on raw_seed7 -> watchdog kills it,
    flags it, frees the slot, and the queue recovers (later seeds still run)."""
    root = _run(tmp_path, monkeypatch,
                ["--scenario", "neutral", "--inject-stale",
                 "--watchdog-stale-sec", "0.5", "--watchdog-poll-sec", "0.1"])
    out = capsys.readouterr().out

    assert "WATCHDOG KILL raw_seed7" in out, "watchdog did not kill stale run"
    assert "project_seed7_stageb_wedge" in out, "wedge flag not raised"
    assert "SLOT FREED (killed) GPU" in out, "slot not freed after kill"

    # Queue recovered: seed-13 and seed-23 trains still completed.
    assert (root / "raw_seed13" / "phase2_stageB_best.pkl").exists()
    assert (root / "raw_seed23" / "phase2_stageB_best.pkl").exists()
    # The killed run produced NO best ckpt (it was hung then killed).
    assert not (root / "raw_seed7" / "phase2_stageB_best.pkl").exists()


def test_per_run_archival(tmp_path, monkeypatch):
    """Behavior 4: each run dir has resolved_config.json + DEPLOY_HASH;
    token arms carry VOCAB_SHA256; rand_fe carries PERM_SEED."""
    root = _run(tmp_path, monkeypatch, ["--scenario", "neutral"])
    for seed in (7, 13, 23):
        for arm in ("raw", "real_fe", "rand_fe"):
            rd = root / f"{arm}_seed{seed}"
            cfg = json.loads((rd / "resolved_config.json").read_text())
            assert cfg["deploy_hash"] == "57d05ac"
            assert cfg["arm"] == arm and cfg["seed"] == seed
            assert (rd / "DEPLOY_HASH").read_text().strip() == "57d05ac"
            if arm == "raw":
                assert cfg["vocab_file"] is None
                assert not (rd / "VOCAB_SHA256").exists()
                assert not (rd / "PERM_SEED").exists()
            else:
                assert cfg["vocab_file"] is not None
                assert (rd / "VOCAB_SHA256").exists()
                # vocab files don't exist in this isolated dir -> MISSING sha,
                # but the sha field is still recorded (audit-traceable).
                assert cfg["vocab_sha256"] is not None
            if arm == "rand_fe":
                assert cfg["perm_seed"] == 804
                assert (rd / "PERM_SEED").read_text().strip() == "804"


def test_stop_path_kill(tmp_path, monkeypatch, capsys):
    """Behavior 5a: KILL path — real_fe collapses, campaign is killed and no
    extra (extend) seeds are launched."""
    root = _run(tmp_path, monkeypatch, ["--scenario", "kill"])
    out = capsys.readouterr().out
    assert "STOP/KILL" in out
    assert "CAMPAIGN KILLED" in out
    assert "VERDICT: CAMPAIGN KILLED" in out
    # No extend seeds were queued.
    assert "EXTEND" not in out or "EXTEND TRIGGERED" not in out
    assert not (root / "raw_seed31").exists()
    # RESULT.md records the kill.
    assert "CAMPAIGN KILLED" in (root / "RESULT.md").read_text()


def test_stop_path_rand_ge_real(tmp_path, monkeypatch, capsys):
    """Behavior 5a (variant): random >= real consistently -> KILL."""
    _run(tmp_path, monkeypatch, ["--scenario", "rand_ge_real"])
    out = capsys.readouterr().out
    assert "STOP/KILL" in out
    assert "random arm >= real arm" in out


def test_extend_path(tmp_path, monkeypatch, capsys):
    """Behavior 5b: EXTEND path — real_fe 3-seed mean >= +1pp, no seed < -3pp
    -> seeds 31/42 are queued and run."""
    root = _run(tmp_path, monkeypatch, ["--scenario", "extend"])
    out = capsys.readouterr().out
    assert "EXTEND: real_fe 3-seed mean" in out
    assert "EXTEND queue appended" in out
    assert "VERDICT: EXTEND to seeds (31, 42)" in out
    # Extension seeds actually trained + evaluated.
    for seed in (31, 42):
        assert (root / f"raw_seed{seed}" / "phase2_stageB_best.pkl").exists()
        assert (root / f"eval_real_fe_seed{seed}" / "benchmark.json").exists()
    # No campaign kill in the extend scenario.
    assert "CAMPAIGN KILLED" not in out


def test_gpu0_never_referenced():
    """GPU contract: GPU 0 untouchable — never appears in train or eval slots."""
    assert 0 not in orch.TRAIN_GPUS
    assert orch.EVAL_GPU != 0
    assert orch.TRAIN_GPUS == (1, 2)
    assert orch.EVAL_GPU == 3


def test_real_mode_emits_safe_run_and_tmux(tmp_path, monkeypatch, capsys):
    """Real mode (not dry-run) must EMIT safe_run + tmux lines, not spawn GPU
    jobs.  We can't run a full real campaign (no GPU), but the command
    builders + wrappers must produce the contracted shape."""
    argv = orch.build_train_cmd("real_fe", 7, Path("runs/804_token_arms/real_fe_seed7"), 1)
    assert "train_phase2.py" in argv
    assert "--seed" in argv and "7" in argv
    assert "--use-token-frontend" in argv
    assert "--token-frontend-vocab-file" in argv
    wrapped = orch.wrap_safe_run("804_train_real_fe_seed7", 1, "40G", argv)
    assert wrapped[0].endswith("safe_run.sh")
    assert "--gpu" in wrapped and "1" in wrapped
    assert "--gpu-frac" in wrapped and "0.7" in wrapped
    line = orch.tmux_launch_line("804_train_real_fe_seed7", wrapped,
                                 Path("runs/804_token_arms/real_fe_seed7/train.log"))
    assert line.startswith("tmux new-window")
    assert "source trader_env/bin/activate" in line


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
