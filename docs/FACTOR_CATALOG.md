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
| ep_ttm | same row | net_profit_parent_ttm, market_cap, statement_currency |
| roe_ttm | same row | net_profit_parent_ttm, book_equity, statement_currency |
| gross_profitability | same row | gross_profit, total_assets, statement_currency |
| asset_growth_yoy | same row | total_assets, total_assets_prior_year, statement_currency |
| reversal_20d | 20 | close |
| size_log_mcap | same row | market_cap |
| size_log_mcap_ex_shell | cross section | market_cap |
| nonlinear_size | cross section | market_cap |
| beta_252d | 252 | close, market_return |
| momentum_252_21 | 252, skip 21 | close |
| residual_vol_252d | 252 | close, market_return |
| liquidity_log_turnover_20d | 20 | volume, free_float_shares, volume_unit, share_basis |
| book_to_price | same row | book_equity, market_cap, statement_currency |
| earnings_yield_ttm | same row | net_profit_parent_ttm, market_cap, statement_currency |
| earnings_growth_yoy | same row | net_profit_parent_ttm, net_profit_parent_ttm_prior_year, statement_currency |
| book_leverage | same row | total_assets, book_equity, statement_currency |

The academic and risk-exposure names are opt-in and are not part of the historical default set. They are not return forecasts and not MSCI descriptors. `ep_ttm` keeps negative earnings; it is not `1 / pe_ratio`. Statement fields are point-in-time inputs. `market_cap` stays a same-day capitalization column and is not given a statement PIT mapping. `asset_growth_yoy` uses the supplied prior-year asset stock; it does not shift the panel by 252 sessions. `reversal_20d` equals minus `momentum_20d`.

`size_log_mcap_ex_shell` drops names at or below the date's 30th percentile of positive market cap, and is missing when that date has fewer than 10 positive caps. `nonlinear_size` is the cube of the cross-sectional log-cap z-score, residualized on log cap. `beta_252d` gives each of the 252 sessions equal weight. It is not an exponentially weighted Barra beta. `momentum_252_21` is a neutralization exposure; it is not an A-share long-only alpha. `earnings_yield_ttm` is the same trailing ratio as `ep_ttm`, labeled as a risk exposure; cash-earnings and forecast blends are not implemented. `book_to_price` keeps a negative book and is a risk exposure, not an A-share value alpha. `residual_vol_252d` is the annualized residual standard deviation from the same complete 252-session regression as `beta_252d`. `liquidity_log_turnover_20d` is the log of mean raw share turnover and requires the then-known free float.

Tests: `tests/test_momentum.py`, `test_reversal.py`, `test_volatility.py`, `test_liquidity.py`, `test_fundamental_optional.py`, `test_neutralize.py`, `test_cli.py`, `test_academic.py`, `test_risk_exposures.py`, `test_exposure_descriptors.py`, `test_families.py`.

## Families

Every registered factor has one or more labels from `quant_factors.families.FAMILY_DEFINITIONS`. A label names the economic quantity. It does not change the formula or its sign. A close-to-close return through 42 sessions is `reversal`, including default names such as `momentum_20d`. A return of 60 sessions or more is `momentum`. Price-volume patterns keep `technical` and add another label when they also measure a return, volatility, or trading activity. `quant-factors list` prints the labels. `list --json` stays the default registry and does not add labels.
