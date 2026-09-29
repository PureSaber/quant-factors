"""Risk exposures for neutralization. Not return forecasts and not MSCI descriptors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_factors.core import DEFAULT_FACTORS, compute_factors
from quant_factors.expressions import expression_requirements

RISK_NAMES = (
    "size_log_mcap",
    "size_log_mcap_ex_shell",
    "nonlinear_size",
    "beta_252d",
    "momentum_252_21",
)


def _caps(values: list[float], *, currency: list[str] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02"] * len(values),
            "symbol": [f"S{i:02d}" for i in range(len(values))],
            "close": [10.0] * len(values),
            "market_cap": values,
        }
    )
    if currency is not None:
        frame["statement_currency"] = currency
    return frame


def test_risk_exposures_are_opt_in_and_not_fundamental() -> None:
    for name in RISK_NAMES:
        assert name not in DEFAULT_FACTORS
        requirement = expression_requirements([name])[name]
        assert requirement["role"] == "risk_exposure"
        assert requirement["pit_columns"] == []
        assert requirement["pit_required"] is False


def test_log_size_keeps_only_positive_caps_and_one_currency() -> None:
    out = compute_factors(_caps([10.0, 0.0, -2.0, np.nan]), ["size_log_mcap"])
    assert out["size_log_mcap"].iloc[0] == pytest.approx(np.log(10.0))
    assert out["size_log_mcap"].iloc[1:].isna().all()
    mixed = _caps([10.0, 12.0], currency=["CNY", "HKD"])
    with pytest.raises(ValueError, match="statement_currency"):
        compute_factors(mixed, ["size_log_mcap"])
    compute_factors(_caps([10.0, 12.0]), ["size_log_mcap"])


def test_shell_band_drops_smallest_thirty_percent_and_thin_dates() -> None:
    thin = compute_factors(_caps([float(i) for i in range(1, 10)]), ["size_log_mcap_ex_shell"])
    assert thin["size_log_mcap_ex_shell"].isna().all()
    caps = [float(i) for i in range(1, 11)]
    out = compute_factors(_caps(caps), ["size_log_mcap_ex_shell", "size_log_mcap"])
    assert out["size_log_mcap_ex_shell"].iloc[:3].isna().all()
    assert out["size_log_mcap_ex_shell"].iloc[3] == pytest.approx(np.log(4.0))
    assert out.loc[out.market_cap.gt(3.7), "size_log_mcap_ex_shell"].notna().all()
    later = _caps(caps)
    later["date"] = "2024-01-03"
    later["market_cap"] = 100.0
    both = pd.concat([_caps(caps), later], ignore_index=True)
    both.loc[both.date.eq("2024-01-03"), "market_cap"] = 1.0
    screened = compute_factors(both, ["size_log_mcap_ex_shell"])
    first = screened.loc[screened.date.eq("2024-01-02"), "size_log_mcap_ex_shell"]
    pd.testing.assert_series_equal(
        first.reset_index(drop=True),
        out["size_log_mcap_ex_shell"].reset_index(drop=True),
        check_names=False,
    )


def test_nonlinear_size_is_orthogonal_to_log_cap() -> None:
    frame = _caps([float(i) for i in range(1, 11)])
    out = compute_factors(frame, ["nonlinear_size", "size_log_mcap"])
    residual = out["nonlinear_size"]
    assert residual.notna().all()
    assert residual.std(ddof=0) > 0
    correlation = np.corrcoef(residual, out["size_log_mcap"])[0, 1]
    assert correlation == pytest.approx(0.0, abs=1e-8)
    thin = compute_factors(_caps([1.0, 2.0, 3.0]), ["nonlinear_size"])
    assert thin["nonlinear_size"].isna().all()


def test_beta_is_the_full_window_slope_and_market_return_is_unique() -> None:
    periods = 254
    market = np.linspace(-0.01, 0.02, periods)
    close = np.empty(periods)
    close[0] = 100.0
    for index in range(1, periods):
        close[index] = close[index - 1] * (1.0 + 1.5 * market[index])
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=periods),
            "symbol": ["A"] * periods,
            "close": close,
            "market_return": market,
        }
    )
    out = compute_factors(frame, ["beta_252d"])
    assert pd.isna(out["beta_252d"].iloc[251])
    assert out["beta_252d"].iloc[252] == pytest.approx(1.5)
    flat = frame.copy()
    flat["symbol"] = "B"
    flat["close"] = 100.0
    paired = compute_factors(pd.concat([frame, flat], ignore_index=True), ["beta_252d"])
    assert paired.loc[paired.symbol.eq("A"), "beta_252d"].iloc[252] == pytest.approx(1.5)
    assert paired.loc[paired.symbol.eq("B"), "beta_252d"].iloc[252] == pytest.approx(0.0, abs=1e-10)
    changed = frame.copy()
    changed.loc[periods - 1, "close"] = close[-1] * 2.0
    again = compute_factors(changed, ["beta_252d"])
    assert again["beta_252d"].iloc[252] == pytest.approx(out["beta_252d"].iloc[252])
    missing = compute_factors(frame.drop(columns=["market_return"]), ["beta_252d"])
    assert missing["beta_252d"].isna().all()
    conflict = pd.concat(
        [frame.iloc[[252]], frame.iloc[[252]].assign(symbol="B", market_return=0.5)],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="one market_return"):
        compute_factors(conflict, ["beta_252d"])
    assert expression_requirements(["beta_252d"])["beta_252d"]["warmup_bars"] == 253


def test_skipped_month_momentum_is_not_the_full_lookback() -> None:
    close = np.arange(1, 301, dtype=float)
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=len(close)),
            "symbol": ["A"] * len(close),
            "close": close,
        }
    )
    out = compute_factors(frame, ["momentum_252_21"])
    assert pd.isna(out["momentum_252_21"].iloc[251])
    assert out["momentum_252_21"].iloc[252] == pytest.approx(232.0 / 1.0 - 1.0)
    full_lookback = close[252] / close[0] - 1.0
    assert out["momentum_252_21"].iloc[252] != pytest.approx(full_lookback)
    changed = frame.copy()
    changed.loc[len(close) - 1, "close"] = 10_000.0
    again = compute_factors(changed, ["momentum_252_21"])
    assert again["momentum_252_21"].iloc[252] == pytest.approx(out["momentum_252_21"].iloc[252])
    assert expression_requirements(["momentum_252_21"])["momentum_252_21"]["warmup_bars"] == 253
