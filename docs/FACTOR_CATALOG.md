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
| liquidity_log_turnover_60d | 60 | volume, free_float_shares, volume_unit, share_basis |
| liquidity_log_turnover_240d | 240 | volume, free_float_shares, volume_unit, share_basis |
| beta_ew_252d | 252, half-life 63 | close, market_return |
| short_term_reversal_21d | 21 | close |
| long_term_reversal_504_252 | 504 to 252 sessions ago, negated | close |
| seasonality_21d_lag_252 | 21 sessions starting 252 earlier | close |
| dividend_yield_ttm | same row | cash_dividend_ttm, market_cap, statement_currency |
| cash_earnings_yield_ttm | same row | operating_cashflow_ttm, market_cap, statement_currency |
| market_leverage | same row | market_cap, total_debt, statement_currency |
| debt_to_assets | same row | total_debt, total_assets, statement_currency |
| sales_growth_yoy | same row | revenue_ttm, revenue_ttm_prior_year, statement_currency |
| earnings_variability_1260d | 1260 | net_profit_parent_ttm, statement_currency |
| analyst_revision | supplied | analyst_revision |
| forward_earnings_yield | same row | forward_net_profit, market_cap, statement_currency |
| expected_growth | supplied | expected_growth |
| industry_momentum_20d | 20, cross section | close, industry |

The academic and risk-exposure names are opt-in and are not part of the historical default set. They are not return forecasts and not MSCI descriptors. `ep_ttm` keeps negative earnings; it is not `1 / pe_ratio`. Statement fields are point-in-time inputs. `market_cap` stays a same-day capitalization column and is not given a statement PIT mapping. `asset_growth_yoy` uses the supplied prior-year asset stock; it does not shift the panel by 252 sessions. `reversal_20d` equals minus `momentum_20d`.

`size_log_mcap_ex_shell` drops names at or below the date's 30th percentile of positive market cap, and is missing when that date has fewer than 10 positive caps. `nonlinear_size` is the cube of the cross-sectional log-cap z-score, residualized on log cap. `beta_252d` gives each of the 252 sessions equal weight. `beta_ew_252d` uses the same window with an independent 63-session half-life; that half-life is not a vendor calibration. `momentum_252_21` is a neutralization exposure; it is not an A-share long-only alpha. `earnings_yield_ttm` is the same trailing ratio as `ep_ttm`. `cash_earnings_yield_ttm` and `forward_earnings_yield` are separate inputs. A blend, when wanted, is done by the risk model with caller-supplied weights and does not fill a missing component. `book_to_price` keeps a negative book and is a risk exposure, not an A-share value alpha. `residual_vol_252d` is the annualized residual standard deviation from the same complete 252-session regression as `beta_252d`. `liquidity_log_turnover_20d` is the log of mean raw share turnover and requires the then-known free float. `analyst_revision` and `expected_growth` pass through supplied values and stay missing when the column is absent. `industry_momentum_20d` is the leave-one-out industry return. `long_term_reversal_504_252` is minus the second-year return, so it does not overlap `momentum_252_21`. `seasonality_21d_lag_252` is last year's return over the month that is about to start, not the month that just ended. `earnings_variability_1260d` is the standard deviation of trailing earnings over its absolute mean; price does not enter it.

Tests: `tests/test_momentum.py`, `test_reversal.py`, `test_volatility.py`, `test_liquidity.py`, `test_fundamental_optional.py`, `test_neutralize.py`, `test_cli.py`, `test_academic.py`, `test_risk_exposures.py`, `test_exposure_descriptors.py`, `test_style_gap_descriptors.py`.
