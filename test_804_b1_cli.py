"""TASK_804-B1 audit gates.

Verifies the four token-frontend CLI overrides added to ``train_phase2_cli.py``
+ ``train_phase2.py::main()``:

  E1 (canon byte-identical / kill check)
      Parsing WITHOUT any new flag must leave a fresh ``Phase2Config`` at its
      defaults for all four token-frontend fields — i.e. the override block is a
      no-op when the flags are absent. This kills the "flag added but never lands
      in config" silent-no-configure failure AND the inverse (override fires when
      it shouldn't). We assert the whole config dict is unchanged vs a pristine
      Phase2Config, not just the four fields.

  Round-trip
      Each of the four flags, when passed, actually lands in ``Phase2Config``.

The override logic in ``main()`` is a handful of ``if args.X is not None``
lines; rather than import the heavy ``train_phase2.main`` (JAX, mctx, ...), we
replicate exactly those lines in ``_apply_token_frontend_overrides`` below and
keep it byte-for-byte in sync with the committed block. ``parse_args`` itself is
imported and exercised for real.
"""

import sys
import dataclasses

from train_phase2_cli import parse_args
from phase2_config import Phase2Config


# --- Mirror of the TASK_804-B1 override block in train_phase2.py::main() -------
# Keep these lines byte-identical to the committed override block.
def _apply_token_frontend_overrides(config, args):
    if args.use_token_frontend is not None:
        config.use_token_frontend = bool(args.use_token_frontend)
    if args.token_frontend_vocab_file is not None:
        config.token_frontend_vocab_file = args.token_frontend_vocab_file
    if args.token_frontend_language is not None:
        config.token_frontend_language = args.token_frontend_language
    if args.token_frontend_cache_dir is not None:
        config.token_frontend_cache_dir = args.token_frontend_cache_dir
    return config


# parse_args() requires --checkpoint and --stage; supply minimal valid values.
_BASE_ARGV = ['train_phase2.py', '--checkpoint', '/tmp/none', '--stage', 'A']


def _parse(extra):
    argv = _BASE_ARGV + list(extra)
    old = sys.argv
    try:
        sys.argv = argv
        return parse_args()
    finally:
        sys.argv = old


_TOKEN_FIELDS = (
    'use_token_frontend',
    'token_frontend_vocab_file',
    'token_frontend_language',
    'token_frontend_cache_dir',
)


def test_e1_canon_byte_identical_no_flags():
    """E1: absent flags -> config byte-identical to a pristine Phase2Config."""
    args = _parse([])
    # All four new args default to None (= inherit) when the flag is absent.
    for f in _TOKEN_FIELDS:
        assert getattr(args, f) is None, f"arg {f} should default to None, got {getattr(args, f)!r}"

    pristine = Phase2Config()
    configured = _apply_token_frontend_overrides(Phase2Config(), args)

    # Four token-frontend fields untouched at their canon defaults.
    assert configured.use_token_frontend is False
    assert configured.token_frontend_vocab_file == ''
    assert configured.token_frontend_language == 'L1_bpe'
    assert configured.token_frontend_cache_dir == ''

    # Whole config dict unchanged (no collateral mutation).
    assert dataclasses.asdict(configured) == dataclasses.asdict(pristine)


def test_roundtrip_use_token_frontend():
    args = _parse(['--use-token-frontend'])
    cfg = _apply_token_frontend_overrides(Phase2Config(), args)
    assert cfg.use_token_frontend is True


def test_roundtrip_no_use_token_frontend_forces_off():
    """BooleanOptionalAction --no- variant forces OFF over an inherited True."""
    args = _parse(['--no-use-token-frontend'])
    cfg = Phase2Config()
    cfg.use_token_frontend = True  # simulate inherited-from-ckpt True
    _apply_token_frontend_overrides(cfg, args)
    assert cfg.use_token_frontend is False


def test_roundtrip_vocab_file():
    args = _parse(['--token-frontend-vocab-file', 'runs/799/X_vocab.json'])
    cfg = _apply_token_frontend_overrides(Phase2Config(), args)
    assert cfg.token_frontend_vocab_file == 'runs/799/X_vocab.json'


def test_roundtrip_language():
    args = _parse(['--token-frontend-language', 'L8_haar'])
    cfg = _apply_token_frontend_overrides(Phase2Config(), args)
    assert cfg.token_frontend_language == 'L8_haar'


def test_roundtrip_cache_dir():
    args = _parse(['--token-frontend-cache-dir', '/data/tok_cache'])
    cfg = _apply_token_frontend_overrides(Phase2Config(), args)
    assert cfg.token_frontend_cache_dir == '/data/tok_cache'


if __name__ == '__main__':
    test_e1_canon_byte_identical_no_flags()
    print('E1 PASS: no-flags config byte-identical to pristine Phase2Config')
    test_roundtrip_use_token_frontend()
    test_roundtrip_no_use_token_frontend_forces_off()
    test_roundtrip_vocab_file()
    test_roundtrip_language()
    test_roundtrip_cache_dir()
    print('Round-trip PASS: all four flags land in Phase2Config')
    print('ALL TASK_804-B1 GATES PASSED')
