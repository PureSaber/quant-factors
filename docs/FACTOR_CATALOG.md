# Factor Catalog

| Factor | Window | Columns required |
|--------|--------|------------------|
| momentum_5d/10d/20d/60d | 5–60 | close |
| log_momentum_20d | 20 | close |
| reversal_5d/10d | 5–10 | close |
| volatility_10d/20d/60d | 10–60 | close |
| downside_vol_20d | 20 | close |
| mean_reversion_z_20d | 20 | close |
| volume_surge_5d | 5 vs 20 | volume |
| turnover_20d | 20 | volume |
| amihud_illiq_20d | 20 | close, volume |
| pe_inv, pb_inv | — | pe_ratio / pb_ratio |

Tests: `tests/test_momentum.py`, `test_reversal.py`, `test_volatility.py`, `test_liquidity.py`, `test_fundamental_optional.py`, `test_neutralize.py`, `test_cli.py`.
