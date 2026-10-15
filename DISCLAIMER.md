# Disclaimer

**This repository is a research archive. It is not financial advice, not investment advice,
and not a trading product.**

- **Nothing here makes money.** The record in `README.md`, `GRAVEYARD.md` and
  `experiments/LEDGER.md` documents that no strategy in this repository is profitable after
  realistic costs. Positive numbers that appear anywhere in the history were later shown to be
  bugs, artifacts, in-sample fits, or edges that decayed.
- **The live-trading code is unmaintained and known unprofitable.** `real_trade_jax.py`,
  `paper_trade_jax.py`, `config/real_trade_config.py`, `mcts_trader/data/kraken_client.py` and
  related scripts were written early, are not kept in sync with the rest of the code, and have
  not been verified against the current exchange API. They can place real orders with real
  money if given credentials. Do not do that.
- **No fitness for any purpose.** Code is provided "as is" under the MIT License and
  documentation under CC BY 4.0, both of which disclaim all warranties and liability. You are
  solely responsible for anything you run, deploy, or trade.
- **Crypto trading is high risk.** Leverage, shorting, margin calls, exchange outages, fees,
  slippage and borrow costs can lose more than you put in.
- **Numbers are research measurements**, made on specific data windows with specific cost
  assumptions and instruments that were themselves wrong at times (see the framework-epoch
  caveats in `experiments/LEDGER.md`). They are not forecasts.
- **No affiliation.** This project is not affiliated with, endorsed by, or sponsored by Kraken
  or any AI or model provider named in it. No market data is redistributed here; obtaining
  data is subject to the data provider's own terms.
