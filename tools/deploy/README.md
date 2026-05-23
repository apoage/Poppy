# Deploy — workstation ↔ gpu-server

Two-direction sync between this workstation and `gpu-server` (operator@<redacted-ip>:22).
Workstation is source of truth for code; server is execution-only.

## Transports

| Direction | Mechanism | Contents |
|-----------|-----------|----------|
| ws → srv  | git push `training` branch | code, agent context, experiment specs |
| ws → srv  | rsync                       | Bella checkpoints (`checkpoints/diff_teacher/`) |
| ws → srv  | `sync-db.sh`                | `trading_data.db` (**integrity-gated**, manual server-side swap) |
| srv → ws  | rsync                       | `runs/`, `logs/` (training output) |

`sync-db.sh` exists because the 2026-05-12 corruption taught us that a sha256
of the *compressed* file does not catch SQLite page-level corruption — zstd
faithfully compresses a broken DB. The script gates the sync on a source-side
`PRAGMA integrity_check`; the server-side decompress + re-verify is a manual
step the operator runs only when training is idle (the script prints the
exact command). Use this script — never raw `zstd → rsync` on the DB.

The `training` branch is auto-built from the workstation **working tree** (uncommitted
state included). Master commit cadence is independent.

## First-time setup

```bash
tools/deploy/bootstrap-server.sh        # init git + config on server, sanity checks
tools/deploy/deploy.sh --dry-run        # preview first deploy
tools/deploy/deploy.sh                  # actual first deploy
# Then build the venv on server — see "Server Python setup" below.
```

## Server Python setup (one-time, after first deploy)

**Status (2026-05-10):** gpu-server runs Ubuntu 20.04, default `python3` is 3.8.10
only. JAX and most modern ML deps need ≥ 3.10. We do NOT use system Python.

**Recommended: uv** (no sudo, fast, manages Python + venv in one tool).

We install from `requirements.frozen.txt` (a `pip freeze` capture of the
workstation venv) using `--no-deps`. This sidesteps dependency resolution
entirely — workstation already proved this exact package set works together, so
the server reproduces it byte-for-byte (modulo recompiles) without any resolver
involvement. Python version on server must match workstation (3.10).

**OS glibc gap caveat**: workstation runs Ubuntu 25.10 (glibc ~2.40), server
runs Ubuntu 20.04 (glibc 2.31). `pip freeze` doesn't carry binaries, so server
fetches its own wheels per package, tagged for its glibc. manylinux_2_28
wheels work on glibc ≥ 2.28, so 20.04 is fine for the vast majority of the
stack. If a specific package has only manylinux_2_34+ wheels (rare), it will
either fail to install or fall back to a source build (which may need
build-essential — server agent will surface this).

```bash
ssh gpu-server
# Install uv (userspace, ~/.local/bin)
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL                                  # reload PATH

# Install Python 3.10 (matches workstation; prebuilt binary, no compile)
uv python install 3.10

# Create the venv inside the repo so deploy.sh's exclude list catches it
cd ~/Trader
uv venv --python 3.10 trader_env
source trader_env/bin/activate

# Install workstation's exact package set, no resolution
uv pip install --no-deps -r requirements.frozen.txt

# Smoke test
python -c "import jax; print(jax.devices())"
```

**Why --no-deps**: previous attempts to resolve `requirements.lock.txt` hit
multiple internal conflicts (LightZero pinning old gymnasium, tensorflow-metadata
pinning newer protobuf than the lock allows). Workstation's installed environment
is internally inconsistent by strict-resolver standards but operationally works.
We honor what works rather than re-solve.

**Workflow for adding deps** (workstation-mastered, per SERVER.md):
1. `pip install <pkg>` on workstation
2. Verify it works
3. `pip freeze > requirements.frozen.txt` (overwrite)
4. `tools/deploy/deploy.sh --code-only` to ship
5. Server agent re-runs `uv pip install --no-deps -r requirements.frozen.txt`

**Alternatives (documented, not used):**

- **deadsnakes apt** — system-wide Python via PPA. Needs sudo password
  (interactive `ssh -t`). Couples Python lifecycle to apt updates.
  ```bash
  ssh -t gpu-server 'sudo add-apt-repository ppa:deadsnakes/ppa && \
    sudo apt update && sudo apt install python3.11 python3.11-venv python3.11-dev'
  ```
- **pyenv** — builds from source, needs `build-essential libssl-dev libffi-dev ...`
  via sudo. Slow. Avoid unless uv has a problem on this host.

**Why uv won here:** no sudo (server's sudo needs password), no compile, no system
pollution; same tool can manage workstation env later if we want unification.

## Daily flow

```bash
# Edit on workstation. Master may stay uncommitted indefinitely.

tools/deploy/deploy.sh                  # ship current state to server
ssh gpu-server 'cd Trader && source trader_env/bin/activate && python train_jax_fast.py ...'
tools/deploy/pull-results.sh            # bring runs/ + logs/ home
```

## Flags

`deploy.sh`:
- `--dry-run`     — preview only, no commits / pushes / rsync writes
- `--code-only`   — skip Bella checkpoint rsync (fast iteration on code)
- `--memory`      — also sync `~/.claude/projects/-home-lukas-Trader/` to server

`pull-results.sh`:
- `--dry-run`     — preview only
- `path1 path2`   — pull specific subdirs instead of `runs/` and `logs/`

## What gets pushed

Include = everything in working tree EXCEPT what's in `.training-exclude`.
The exclude list owns the curation — edit there if something's wrong.

Notable exclusions:
- `lore/`, `tasks/`, `bloom/`, `yarrow/`, `docs/`, `_old/`
- `checkpoints/`, `runs/`, `logs/`, `wandb/`, `output/`, `trader_env/`, `data/`, `*.db`
- Top-level `.md` files except agent context (`CLAUDE.md`, `AGENTS.md`, `README.md`,
  `SERVER.md`)

## Server agent

Server has no `lore/`, `tasks/`, `bloom/`. The root `CLAUDE.md` should detect this via
the absence of `.env-workstation` (a marker file gitignored on workstation) and switch
to server-agent mode. See `SERVER.md` for the server agent's contract.

## Recovery

- "I want to restart server from scratch": `ssh gpu-server 'rm -rf ~/Trader/.git'`,
  re-run bootstrap, redeploy.
- "I broke the training worktree on workstation": `git worktree remove --force
  .git-training-worktree`, then `deploy.sh` recreates it.
- "I need to pull a specific run": `tools/deploy/pull-results.sh runs/<id>`
