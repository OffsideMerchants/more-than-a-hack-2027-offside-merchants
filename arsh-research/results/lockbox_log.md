# Lockbox log (J03WMX)

| # | date | family | what | git | note |
|---|---|---|---|---|---|
| 1 | 2026-11-01 | S3 GNN | `lockbox_gnn_real`, `lockbox_pool_real` (ball=real, trained on all 6 dev matches, 3000 steps, seed 0) | uncommitted working tree | first scoring; an earlier attempt crashed on an assertion before any scoring |

| model | copy | spellF1 [5-95% bootstrap] | macroF1 | alive/dead F1 | team acc |
|---|---|---|---|---|---|
| gnn | deg_A | 0.488 [0.444, 0.532] | 0.891 | 0.968 | 0.889 |
| gnn | deg_B | 0.467 [0.428, 0.505] | 0.891 | 0.973 | 0.881 |
| pool | deg_A | 0.329 [0.286, 0.364] | 0.821 | 0.916 | 0.863 |
| pool | deg_B | 0.338 [0.300, 0.373] | 0.833 | 0.925 | 0.868 |

Selection of the ball=real configuration was made on the dev folds only. Early-stopping validation match J03WOY was also in the training set for these final runs.

## Public window check (BAR-ATH frames 11185-18684)

| date | what | spellF1 | macroF1 |
|---|---|---:|---:|
| 2026-11-02 | five-model ensemble cs0-cs4, correctness check only, no tuning | 0.739 | 0.906 |

One check for this gate. Only 17 true possessions, so the noise band is large.
