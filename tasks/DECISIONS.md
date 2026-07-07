# DECISIONS — the single blocked-on-human surface

One row per call. Recommendation FIRST — the human should be able to answer with one word.
OPEN rows are walls: the factory never guesses past one; the planner seeds rows here instead of
scattering open questions across task files. Answered rows move to DECIDED with the date.

## OPEN

| # | Decision | Recommendation | Context |
|---|----------|----------------|---------|
| D1 | Do TASK_824/825/826 stay active alongside 828, or is 828 the sole focus? | 828 sole factory focus; 824 runs as a scheduled server job outside the loop | `tasks/QUEUE.md` gated rows |
| D2 | TASK_827 (reward-head fix): fully dead, or fallback if 828 stalls? | Dead — superseded; its substance folds into 828 Stage 3 targets | handoff 2026-07-06 |
| D3 | Keep the TASK_820 paper loop (systemd timer 02:21) running? | Keep — zero-cost forward evidence on the linear book | `tools/systemd/trader-paper.*` |
| D4 | Iris-DEX: pursue the node-less hosted-RPC path (TASK_826) or fully shelve? | Shelve until 828 Stage 4 verdict | `tasks/TASK_826_ONCHAIN_SIGNAL_PROBE.md` |
| D5 | Prune round 2: remove Foxglove (train_easy.py, easy_trainer.py, tools/orchestration, tools/DSe) and/or Bella (diff_teacher/)? | Keep Bella (synthetic-market generator is promising for 828 training data); retire Foxglove (oracle-as-teacher path, superseded by 828) | repo prune 2026-07-07 kept both pending this call |

## DECIDED

| # | Decision | Outcome | Date |
|---|----------|---------|------|
| D0 | Deploy the linear CTA to the VPS (TASK_823)? | REJECTED — too trivial to deploy; VPS stays a services host | 2026-07-06 |
