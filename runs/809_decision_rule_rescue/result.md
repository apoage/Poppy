# TASK_809 — Decision-rule rescue — RESULT

> **RETRACTED-AT-SOURCE 2026-08-19 (planner, operator-ratified; pathfinder audit
> `docs/pathfinder_audit_2026-08-19.md` §A).** The "VERDICT (realistic only): 2 of 864 realistic
> cells pass" and the +5.08 headline below are an **estimator artifact**: per-bar returns
> annualized ×√525600 over OVERLAPPING holds (the "809 trap" TASK_819's discipline section cites).
> The retraction was recorded in `lore/notes_2026-06-25.md` but never propagated here — this
> banner closes that. **The corrected per-decision number was never computed, so the cells below
> are neither validated nor falsified** — the strategy died of venue economics and a broken
> estimator, not of evidence. The honest re-measurement is **TASK_897**
> (`tasks/TASK_897_CORRECTED_809_REMEASURE.md`); until it reports, quote NOTHING from this file
> except as history. Also note: the table's durable rows are the OPT corner the run itself
> excluded from its verdict — the realistic sweep's per-cell data was not preserved.

## Models
- **Arm A**: conviction-percentile floor (eligible = top (1-τ) of the cross-section by pred) + min-hold N bars (held names stay ≥N). τ∈(0.0, 0.75, 0.9), N∈(0, 60). Turnover MEASURED (target <~0.1).
- **Arm B (maker)**: Kraken maker = POSITIVE fee (NOT a rebate). Net cost/side = maker_fee + adv_sel, BOTH ≥0 and swept (fee∈{0,8,16}bp, adv∈{0,5,10}bp). Limit at close(d), fills iff bar d+1 low≤limit/high≥limit (execution fact, no decision look-ahead, G12); adverse selection = fill-conditioning + adv_sel penalty. Free corner (fee=0∧adv=0) FLAGGED OPT, excluded from verdict.
- Inherited (806/808): linear_xs signal, split, taker 21bp, per-bar Sharpe×√525600, block-bootstrap CI, gate (Sharpe≥1.0 & PnL>0 & CI lo>0). Signal-IC {1: 0.15105257300384026, 5: 0.14115985613795426, 20: 0.059754718949893586}.
- Universe 64, grid 987904 rows. RSS peak **5.17GB**, runtime 4668s.

## Hard checks
1. **Arm A turnover ≪ 1.0?** gated min turnover = **0.0325** (all configs); at aggressive conviction/min-hold (τ≥0.9 or N≥60) min = **0.0325** (808 fixed ≈ 1.0; target <~0.1).
2. **Arm B net cost positive across realistic sweep?** maker net cost/side = fee+adv ≥ 0 by construction; free corner flagged OPT.

## Gate
WORKS iff post-cost Sharpe ≥ 1.0 AND PnL>0 AND CI lo>0 — and the cell is REALISTIC (maker free corner excluded).

**VERDICT (realistic only): 2 of 864 realistic cells pass.**
Best realistic: gated_maker H=1 k=16 long_short τ=0.75 N=60 fee=0bp adv=5bp → post Sharpe **+5.08** CI[+2.77,+7.63] turnover 0.0868 fill 0.73 PnL +1.3127.

## Cells (ranked by post-cost Sharpe; OPT = optimistic free corner, excluded)
| arm | H | k | var | τ | N | fee bp | adv bp | turnover | fill% | Sh pre | Sh post | CI | PnL | maxDD | flag |
|---|--:|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|---|--:|--:|:--:|
| gated_maker | 1 | 16 | long_short | 0 | 0 | 0 | 0 | 1.1100 | 76 | +229.95 | +229.95 | [+218.51,+241.63] | +101582019953475584000.0000 | -0.0659 | OPT |
| gated_maker | 1 | 16 | long_short | 0.75 | 0 | 0 | 0 | 1.7397 | 77 | +204.11 | +204.11 | [+191.87,+215.92] | +26013110554518253568.0000 | -0.0682 | OPT |
| gated_maker | 1 | 32 | long_short | 0.75 | 0 | 0 | 0 | 0.8698 | 77 | +204.11 | +204.11 | [+191.87,+215.92] | +5180951253.0024 | -0.0345 | OPT |
| gated_maker | 1 | 32 | long_short | 0 | 0 | 0 | 0 | 0.9938 | 76 | +192.96 | +192.96 | [+179.51,+206.36] | +65995728250952.0547 | -0.0460 | OPT |
| gated_maker | 1 | 16 | long_short | 0.9 | 0 | 0 | 0 | 0.7769 | 77 | +184.65 | +184.65 | [+173.96,+194.72] | +74219367476.7798 | -0.0589 | OPT |
| gated_maker | 1 | 32 | long_short | 0.9 | 0 | 0 | 0 | 0.3884 | 77 | +184.65 | +184.65 | [+173.96,+194.72] | +274050.0530 | -0.0298 | OPT |
| gated_maker | 5 | 16 | long_short | 0 | 0 | 0 | 0 | 1.1907 | 78 | +173.45 | +173.45 | [+162.19,+184.44] | +11134607315537136.0000 | -0.0741 | OPT |
| gated_maker | 5 | 16 | long_short | 0.75 | 0 | 0 | 0 | 1.8464 | 79 | +142.15 | +142.15 | [+131.08,+152.97] | +819104029732844.0000 | -0.1026 | OPT |
| gated_maker | 5 | 32 | long_short | 0.75 | 0 | 0 | 0 | 0.9232 | 79 | +142.15 | +142.15 | [+131.08,+152.97] | +29152390.9839 | -0.0524 | OPT |
| gated_maker | 5 | 16 | long_short | 0.9 | 0 | 0 | 0 | 0.8542 | 79 | +137.19 | +137.19 | [+128.05,+145.69] | +349800993.9843 | -0.0541 | OPT |
| gated_maker | 5 | 32 | long_short | 0.9 | 0 | 0 | 0 | 0.4271 | 79 | +137.19 | +137.19 | [+128.05,+145.69] | +18823.1195 | -0.0273 | OPT |
| gated_maker | 5 | 32 | long_short | 0 | 0 | 0 | 0 | 1.0634 | 78 | +136.03 | +136.03 | [+124.34,+147.58] | +17363805974.6220 | -0.0723 | OPT |
| gated_maker | 1 | 16 | long_only | 0.9 | 0 | 0 | 0 | 0.3892 | 77 | +100.48 | +100.48 | [+90.97,+109.96] | +410611.2440 | -0.0963 | OPT |
| gated_maker | 1 | 32 | long_only | 0.9 | 0 | 0 | 0 | 0.1946 | 77 | +100.48 | +100.48 | [+90.97,+109.96] | +643.0787 | -0.0491 | OPT |
| gated_maker | 1 | 16 | long_only | 0 | 0 | 0 | 0 | 0.5546 | 76 | +94.93 | +94.93 | [+85.93,+104.08] | +19095318981.5370 | -0.2002 | OPT |
| gated_maker | 1 | 16 | long_only | 0.75 | 0 | 0 | 0 | 0.8698 | 77 | +87.84 | +87.84 | [+78.89,+97.06] | +7340859596.8361 | -0.2088 | OPT |
| gated_maker | 1 | 32 | long_only | 0.75 | 0 | 0 | 0 | 0.4349 | 77 | +87.84 | +87.84 | [+78.89,+97.06] | +87471.5619 | -0.1094 | OPT |
| gated_maker | 20 | 16 | long_short | 0 | 0 | 0 | 0 | 1.3759 | 80 | +87.62 | +87.62 | [+81.84,+93.19] | +178361374.4367 | -0.1089 | OPT |
| gated_maker | 1 | 32 | long_only | 0 | 0 | 0 | 0 | 0.4968 | 76 | +80.22 | +80.22 | [+73.04,+87.88] | +10032629.3666 | -0.1424 | OPT |
| gated_maker | 1 | 32 | long_short | 0 | 60 | 0 | 0 | 0.1430 | 74 | +77.78 | +77.78 | [+73.82,+81.77] | +6824.9534 | -0.0403 | OPT |
| gated_maker | 5 | 16 | long_only | 0.9 | 0 | 0 | 0 | 0.4277 | 79 | +75.39 | +75.39 | [+67.61,+83.30] | +28603.8684 | -0.0991 | OPT |
| gated_maker | 5 | 32 | long_only | 0.9 | 0 | 0 | 0 | 0.2139 | 79 | +75.39 | +75.39 | [+67.61,+83.30] | +169.0951 | -0.0506 | OPT |
| gated_maker | 5 | 16 | long_only | 0 | 0 | 0 | 0 | 0.5958 | 78 | +73.74 | +73.74 | [+65.83,+82.04] | +203411931.6132 | -0.2491 | OPT |
| gated_maker | 5 | 16 | long_only | 0.75 | 0 | 0 | 0 | 0.9234 | 79 | +63.96 | +63.96 | [+56.31,+71.86] | +48888498.4763 | -0.2571 | OPT |
| gated_maker | 5 | 32 | long_only | 0.75 | 0 | 0 | 0 | 0.4617 | 79 | +63.96 | +63.96 | [+56.31,+71.86] | +7158.5052 | -0.1367 | OPT |
| gated_maker | 5 | 32 | long_short | 0 | 60 | 0 | 0 | 0.1429 | 76 | +61.66 | +61.66 | [+56.77,+66.28] | +1386.7276 | -0.0881 | OPT |
| gated_maker | 1 | 16 | long_short | 0 | 60 | 0 | 0 | 0.0863 | 73 | +60.46 | +60.46 | [+57.36,+63.68] | +27162.1187 | -0.0500 | OPT |
| gated_maker | 1 | 16 | long_short | 0.75 | 60 | 0 | 0 | 0.0868 | 73 | +60.02 | +60.02 | [+56.88,+63.10] | +29092.2191 | -0.0514 | OPT |
| gated_maker | 1 | 32 | long_short | 0.75 | 60 | 0 | 0 | 0.1089 | 74 | +59.71 | +59.71 | [+55.15,+64.53] | +2585.2045 | -0.0747 | OPT |
| gated_maker | 20 | 32 | long_short | 0 | 0 | 0 | 0 | 1.1823 | 79 | +59.63 | +59.63 | [+54.21,+64.71] | +48479.9935 | -0.0817 | OPT |
| gated_maker | 20 | 16 | long_short | 0.75 | 0 | 0 | 0 | 2.0536 | 80 | +58.31 | +58.31 | [+52.75,+63.46] | +1930037.1669 | -0.1600 | OPT |
| gated_maker | 20 | 32 | long_short | 0.75 | 0 | 0 | 0 | 1.0268 | 80 | +58.31 | +58.31 | [+52.75,+63.46] | +1414.8468 | -0.0828 | OPT |
| gated_maker | 5 | 32 | long_only | 0 | 0 | 0 | 0 | 0.5315 | 78 | +57.75 | +57.75 | [+51.03,+64.66] | +169804.4802 | -0.1834 | OPT |
| gated_maker | 1 | 16 | long_short | 0.9 | 60 | 0 | 0 | 0.0863 | 73 | +56.25 | +56.25 | [+52.96,+59.53] | +14420.2876 | -0.0608 | OPT |
| gated_maker | 20 | 16 | long_short | 0.9 | 0 | 0 | 0 | 0.9742 | 81 | +55.26 | +55.26 | [+49.87,+60.36] | +2635.2026 | -0.0731 | OPT |
| gated_maker | 20 | 32 | long_short | 0.9 | 0 | 0 | 0 | 0.4871 | 81 | +55.26 | +55.26 | [+49.87,+60.36] | +50.6643 | -0.0371 | OPT |
| gated_maker | 5 | 16 | long_short | 0 | 60 | 0 | 0 | 0.0867 | 76 | +53.84 | +53.84 | [+50.67,+56.68] | +8528.8612 | -0.0513 | OPT |
| gated_maker | 5 | 16 | long_short | 0.75 | 60 | 0 | 0 | 0.0872 | 76 | +51.35 | +51.35 | [+48.45,+54.36] | +6962.8436 | -0.0584 | OPT |
| gated_maker | 5 | 16 | long_short | 0.9 | 60 | 0 | 0 | 0.0869 | 77 | +49.70 | +49.70 | [+46.55,+52.72] | +5183.1580 | -0.0756 | OPT |
| gated_maker | 5 | 32 | long_short | 0.75 | 60 | 0 | 0 | 0.1096 | 76 | +48.88 | +48.88 | [+43.63,+54.10] | +750.3316 | -0.1167 | OPT |
| gated_maker | 1 | 32 | long_short | 0.9 | 60 | 0 | 0 | 0.0788 | 74 | +40.75 | +40.75 | [+35.92,+45.90] | +771.3816 | -0.1265 | OPT |
| gated_maker | 20 | 16 | long_only | 0 | 0 | 0 | 0 | 0.6860 | 80 | +37.72 | +37.72 | [+32.40,+43.38] | +19386.0510 | -0.1914 | OPT |
| fixed_maker | 1 | 16 | long_short | 0 | 0 | 0 | 0 | 0.1370 | 76 | +37.06 | +37.06 | [+32.78,+41.14] | +3122.7775 | -0.1058 | OPT |
| gated_maker | 5 | 32 | long_short | 0.9 | 60 | 0 | 0 | 0.0794 | 76 | +33.40 | +33.40 | [+28.05,+39.12] | +292.4887 | -0.1856 | OPT |
| fixed_maker | 5 | 16 | long_short | 0 | 0 | 0 | 0 | 0.1354 | 77 | +30.76 | +30.76 | [+26.47,+35.09] | +1079.9019 | -0.1843 | OPT |
| gated_maker | 20 | 32 | long_short | 0 | 60 | 0 | 0 | 0.1441 | 78 | +30.46 | +30.46 | [+27.46,+33.39] | +34.3377 | -0.0500 | OPT |
| gated_maker | 20 | 16 | long_short | 0.75 | 60 | 0 | 0 | 0.0892 | 79 | +29.75 | +29.75 | [+27.32,+32.22] | +155.5461 | -0.0853 | OPT |
| gated_maker | 20 | 16 | long_only | 0.9 | 0 | 0 | 0 | 0.4871 | 81 | +29.37 | +29.37 | [+24.35,+34.80] | +57.6721 | -0.1240 | OPT |
| gated_maker | 20 | 32 | long_only | 0.9 | 0 | 0 | 0 | 0.2436 | 81 | +29.37 | +29.37 | [+24.35,+34.80] | +6.7050 | -0.0637 | OPT |
| gated_maker | 20 | 16 | long_short | 0 | 60 | 0 | 0 | 0.0889 | 79 | +29.12 | +29.12 | [+26.26,+32.13] | +144.2478 | -0.0741 | OPT |
| gated_maker | 20 | 16 | long_short | 0.9 | 60 | 0 | 0 | 0.0891 | 79 | +28.87 | +28.87 | [+26.32,+31.32] | +140.1585 | -0.0433 | OPT |
| gated_maker | 20 | 32 | long_short | 0.75 | 60 | 0 | 0 | 0.1125 | 78 | +28.20 | +28.20 | [+24.22,+32.20] | +41.9421 | -0.1137 | OPT |
| gated_maker | 20 | 16 | long_only | 0.75 | 0 | 0 | 0 | 1.0270 | 80 | +27.47 | +27.47 | [+22.56,+32.60] | +2172.4171 | -0.2460 | OPT |
| gated_maker | 20 | 32 | long_only | 0.75 | 0 | 0 | 0 | 0.5135 | 80 | +27.47 | +27.47 | [+22.56,+32.60] | +46.7679 | -0.1304 | OPT |
| gated_maker | 20 | 32 | long_only | 0 | 0 | 0 | 0 | 0.5903 | 79 | +26.71 | +26.71 | [+22.01,+31.65] | +285.8498 | -0.1906 | OPT |
| fixed_maker | 1 | 32 | long_short | 0 | 0 | 0 | 0 | 0.0936 | 75 | +25.33 | +25.33 | [+21.10,+29.70] | +172.9922 | -0.1187 | OPT |
| fixed_maker | 5 | 32 | long_short | 0 | 0 | 0 | 0 | 0.0931 | 76 | +18.84 | +18.84 | [+14.75,+23.32] | +50.0650 | -0.3157 | OPT |
| gated_maker | 20 | 32 | long_short | 0.9 | 60 | 0 | 0 | 0.0814 | 79 | +18.58 | +18.58 | [+14.50,+22.95] | +20.8142 | -0.1272 | OPT |
| fixed_maker | 20 | 16 | long_short | 0 | 0 | 0 | 0 | 0.1335 | 78 | +16.87 | +16.87 | [+13.98,+20.24] | +44.9932 | -0.2331 | OPT |
| fixed_maker | 1 | 16 | long_only | 0 | 0 | 0 | 0 | 0.0686 | 76 | +15.96 | +15.96 | [+11.22,+20.91] | +67.7477 | -0.3263 | OPT |
| gated_maker | 1 | 16 | long_only | 0 | 60 | 0 | 0 | 0.0334 | 74 | +14.94 | +14.94 | [+10.51,+19.71] | +61.7592 | -0.3192 | OPT |
| fixed_maker | 5 | 16 | long_only | 0 | 0 | 0 | 0 | 0.0678 | 77 | +14.12 | +14.12 | [+9.61,+18.64] | +41.2481 | -0.3423 | OPT |
| gated_maker | 1 | 16 | long_only | 0.75 | 60 | 0 | 0 | 0.0334 | 74 | +13.55 | +13.55 | [+9.15,+18.28] | +46.1686 | -0.3220 | OPT |
| gated_maker | 1 | 16 | long_only | 0.9 | 60 | 0 | 0 | 0.0332 | 74 | +13.51 | +13.51 | [+9.12,+18.36] | +48.6220 | -0.3579 | OPT |
| fixed_maker | 1 | 16 | long_short | 0 | 0 | 0 | 0 | 0.0463 | 75 | +13.23 | +13.23 | [+10.11,+16.62] | +16.0742 | -0.1492 | OPT |
| gated_maker | 5 | 16 | long_only | 0.75 | 60 | 0 | 0 | 0.0334 | 77 | +12.60 | +12.60 | [+8.31,+17.27] | +35.1555 | -0.3319 | OPT |
| fixed_maker | 1 | 32 | long_only | 0 | 0 | 0 | 0 | 0.0468 | 75 | +12.45 | +12.45 | [+7.88,+17.51] | +14.3674 | -0.2954 | OPT |
| gated_maker | 5 | 16 | long_only | 0.9 | 60 | 0 | 0 | 0.0333 | 77 | +12.28 | +12.28 | [+7.99,+16.88] | +33.2540 | -0.3501 | OPT |
| gated_maker | 5 | 16 | long_only | 0 | 60 | 0 | 0 | 0.0335 | 77 | +12.17 | +12.17 | [+7.87,+16.89] | +28.0609 | -0.3189 | OPT |
| gated_maker | 1 | 32 | long_only | 0 | 60 | 0 | 0 | 0.0332 | 74 | +11.74 | +11.74 | [+7.16,+16.55] | +18.3910 | -0.3231 | OPT |
| fixed_maker | 20 | 32 | long_short | 0 | 0 | 0 | 0 | 0.0927 | 78 | +11.36 | +11.36 | [+8.11,+15.00] | +9.6159 | -0.2746 | OPT |
| gated_maker | 1 | 32 | long_only | 0.75 | 60 | 0 | 0 | 0.0332 | 74 | +11.11 | +11.11 | [+6.66,+15.87] | +17.1447 | -0.3501 | OPT |
| gated_maker | 1 | 32 | long_only | 0.9 | 60 | 0 | 0 | 0.0328 | 74 | +10.46 | +10.46 | [+5.90,+15.08] | +14.0803 | -0.3473 | OPT |
| gated_maker | 5 | 32 | long_only | 0 | 60 | 0 | 0 | 0.0333 | 76 | +10.22 | +10.22 | [+5.79,+15.02] | +11.8850 | -0.3206 | OPT |
| fixed_maker | 5 | 32 | long_only | 0 | 0 | 0 | 0 | 0.0467 | 76 | +10.12 | +10.12 | [+5.83,+14.69] | +8.0130 | -0.2917 | OPT |
| gated_maker | 5 | 32 | long_only | 0.75 | 60 | 0 | 0 | 0.0333 | 76 | +9.71 | +9.71 | [+5.24,+14.51] | +11.4080 | -0.3441 | OPT |
| gated_maker | 5 | 32 | long_only | 0.9 | 60 | 0 | 0 | 0.0329 | 77 | +9.35 | +9.35 | [+4.93,+14.02] | +10.3342 | -0.3496 | OPT |
| fixed_maker | 1 | 32 | long_short | 0 | 0 | 0 | 0 | 0.0314 | 75 | +9.21 | +9.21 | [+5.35,+13.26] | +5.3941 | -0.1920 | OPT |
| fixed_maker | 5 | 16 | long_short | 0 | 0 | 0 | 0 | 0.0460 | 76 | +8.64 | +8.64 | [+5.41,+12.03] | +5.4234 | -0.2358 | OPT |
| fixed_maker | 20 | 16 | long_only | 0 | 0 | 0 | 0 | 0.0670 | 78 | +8.30 | +8.30 | [+4.41,+12.46] | +7.3193 | -0.2712 | OPT |
| fixed_maker | 20 | 16 | long_short | 0 | 0 | 0 | 0 | 0.0448 | 78 | +7.93 | +7.93 | [+4.63,+11.34] | +4.7633 | -0.2093 | OPT |
| gated_maker | 20 | 16 | long_only | 0 | 60 | 0 | 0 | 0.0336 | 80 | +7.67 | +7.67 | [+3.65,+12.06] | +6.4930 | -0.3436 | OPT |
| fixed_maker | 5 | 32 | long_short | 0 | 0 | 0 | 0 | 0.0315 | 76 | +6.73 | +6.73 | [+3.09,+10.63] | +2.7302 | -0.2274 | OPT |
| fixed_maker | 20 | 32 | long_only | 0 | 0 | 0 | 0 | 0.0465 | 78 | +6.64 | +6.64 | [+2.66,+10.82] | +2.9921 | -0.2518 | OPT |
| gated_maker | 20 | 16 | long_only | 0.9 | 60 | 0 | 0 | 0.0333 | 80 | +6.55 | +6.55 | [+2.73,+10.89] | +5.0929 | -0.3552 | OPT |
| gated_maker | 20 | 16 | long_only | 0.75 | 60 | 0 | 0 | 0.0334 | 80 | +6.45 | +6.45 | [+2.53,+10.91] | +4.7413 | -0.3692 | OPT |
| fixed_maker | 1 | 16 | long_only | 0 | 0 | 0 | 0 | 0.0231 | 75 | +6.12 | +6.12 | [+2.06,+10.46] | +3.7792 | -0.3455 | OPT |
| gated_maker | 20 | 32 | long_only | 0 | 60 | 0 | 0 | 0.0334 | 79 | +6.01 | +6.01 | [+2.01,+10.45] | +3.2667 | -0.3293 | OPT |
| gated_maker | 20 | 32 | long_only | 0.75 | 60 | 0 | 0 | 0.0333 | 79 | +5.73 | +5.73 | [+1.69,+10.24] | +3.2002 | -0.3733 | OPT |
| fixed_maker | 20 | 32 | long_short | 0 | 0 | 0 | 0 | 0.0310 | 77 | +5.71 | +5.71 | [+2.06,+9.59] | +2.1702 | -0.3138 | OPT |

## Scope honesty
Per-bar accrual, OOS test. Maker fills from OHLC touch (no intra-bar microstructure); fill-rate and adverse selection are coupled in reality — flat fill×flat adv_sel is a crude first cut, so the optimistic corner is flagged OPT and the verdict rests on the realistic sweep (Kraken maker 0-16bp + adv 0-10bp). Turnover MEASURED. No look-ahead (G11/G12).

## Deploy plan
Cell gated_maker H=1 k=16 long_short τ=0.75 N=60 cleared the gate at realistic cost (turnover 0.0868). Paper-trade via a thin real_trade_jax.py sibling: per bar build features_linear_xs→rank→conviction-percentile/min-hold rule; post maker limits, track realized fill-rate. Gate-3 paper before live; compare realized post-cost Sharpe to this cell.
