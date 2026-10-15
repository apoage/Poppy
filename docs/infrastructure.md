# Infrastructure

Redacted for the public release. The original described the private hosts this project ran on:

- `workstation` — 1x RTX 3090, orchestration and planning.
- `gpu-server` — multi-RTX-3090 training/verification host, reached over a private VPN.
- a small VPS that ran an unattended Kraken public-OHLCV puller (`tools/kraken_pull_vps/`).

Host addresses, users, ports, VPN topology and VPS service configuration are intentionally omitted.
The market database (`trading_data.db`, Kraken OHLCV) and the verification fixture
(`fixtures/trading_fixture.db`, built by `fixtures/make_fixture.py`) are not included.
