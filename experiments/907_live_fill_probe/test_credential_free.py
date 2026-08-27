"""Confirmation, structurally enforced: phase 1 reads/logs/references no credential at all,
and the whole probe (phase 1 + phase 2 rails) never imports the private/authenticated Kraken
REST path. Grepped rather than asserted in prose, so a future edit that adds a credential
reference here fails CI rather than fails silently.
"""
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
FORBIDDEN = ("KRAKEN_API_KEY", "KRAKEN_API_SECRET", "_private_request", "api_secret", "api_key")

PHASE1_FILES = ("recorder.py", "capture.py", "shadow_engine.py", "reference_fill_model.py",
                 "estimands.py", "render.py", "cli.py")
ALL_SOURCE_FILES = PHASE1_FILES + ("phase2_rails.py",)


def test_phase1_files_never_reference_any_credential():
    for fname in PHASE1_FILES:
        text = (_THIS_DIR / fname).read_text()
        for token in FORBIDDEN:
            assert token not in text, f"{fname} references {token!r} — phase 1 must not"


def test_phase2_rails_never_reads_a_credential_either():
    text = (_THIS_DIR / "phase2_rails.py").read_text()
    for token in FORBIDDEN:
        assert token not in text, f"phase2_rails.py references {token!r}"


def test_no_file_here_imports_kraken_client_private_machinery():
    for fname in ALL_SOURCE_FILES:
        text = (_THIS_DIR / fname).read_text()
        assert "kraken_client" not in text, f"{fname} references kraken_client.py"
