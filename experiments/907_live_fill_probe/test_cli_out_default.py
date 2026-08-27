"""Gate: `--out`'s default must NOT land on a tracked path — a named prior defect class in
this repo (a default `--out` on a tracked artifact is a write nobody thinks of as an edit).
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cli  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_default_out_path_is_under_the_gitignored_runs_dir():
    out = Path(cli._default_out_path(now=1000))
    assert out.name == "shadow_1000.md"
    assert out.parent == _REPO_ROOT / "runs" / "907_live_fill_probe"


def test_default_out_path_is_actually_git_ignored():
    out = cli._default_out_path(now=1000)
    result = subprocess.run(["git", "check-ignore", "-q", out], cwd=str(_REPO_ROOT))
    # exit code 0 means the path IS ignored — this is what we require
    assert result.returncode == 0, f"{out} is NOT gitignored — default --out would land tracked"


def test_default_out_path_is_not_under_tasks_docs_or_lore():
    out = str(cli._default_out_path(now=1000))
    for forbidden in ("/tasks/", "/docs/", "/lore/", "/.dark-factory/"):
        assert forbidden not in out


def test_cli_flags_are_not_abbreviation_reachable():
    """MINOR 2: `allow_abbrev=False`. `--capture` must not be reachable as `--cap`, and no flag
    may be silently prefix-matched — the sibling 881D row shipped exactly that defect class."""
    import pytest
    p = cli.build_argparser()
    for truncated in ("--cap", "--captur", "--ou", "--pair", "--gap-detect-threshold"):
        with pytest.raises(SystemExit):
            p.parse_args([truncated, "x"])
    # the explicit spellings still work
    ns = p.parse_args(["--capture", "a.jsonl", "--out", "o.md", "--pairs", "BTC/USD"])
    assert ns.capture == ["a.jsonl"] and ns.out == "o.md" and ns.pairs == "BTC/USD"


def test_recorder_flags_are_not_abbreviation_reachable():
    import pytest
    from recorder import build_argparser as recorder_argparser
    p = recorder_argparser()
    for truncated in ("--pair", "--ou", "--capture-i", "--heartbeat-timeout"):
        with pytest.raises(SystemExit):
            p.parse_args(["--pairs", "BTC/USD", "--out", "o.jsonl", truncated, "1"])
    ns = p.parse_args(["--pairs", "BTC/USD", "--out", "o.jsonl"])
    assert ns.pairs == "BTC/USD" and ns.out == "o.jsonl"
