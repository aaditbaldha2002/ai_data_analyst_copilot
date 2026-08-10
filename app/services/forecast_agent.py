import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


def _to_series(result: list[dict], date_key: str, value_key: str) -> pd.Series:
    df = pd.DataFrame(result)
    df[date_key] = pd.to_datetime(df[date_key])
    df = df.sort_values(date_key)
    series = pd.Series(df[value_key].values, index=pd.DatetimeIndex(df[date_key]))
    return series


def _detect_seasonality_strength(series: pd.Series, period: int) -> float:
    """Returns a 0-1 score for how strong seasonality is, using seasonal_decompose."""
    if len(series) < period * 2:
        return 0.0
    try:
        decomposition = seasonal_decompose(series, period=period, model="additive", extrapolate_trend="freq")
        seasonal_var = np.var(decomposition.seasonal)
        residual_var = np.var(decomposition.resid.dropna())
        if seasonal_var + residual_var == 0:
            return 0.0
        return seasonal_var / (seasonal_var + residual_var)
    except Exception:
        return 0.0


def _linear_trend_forecast(series: pd.Series, periods: int) -> list[dict]:
    x = np.arange(len(series))
    y = series.values
    coeffs = np.polyfit(x, y, 1)
    trend = np.poly1d(coeffs)

    freq = pd.infer_freq(series.index) or "MS"
    future_index = pd.date_range(start=series.index[-1], periods=periods + 1, freq=freq)[1:]
    future_x = np.arange(len(series), len(series) + periods)
    predictions = trend(future_x)

    return [
        {"date": str(d.date()), "predicted_value": float(v)}
        for d, v in zip(future_index, predictions)
    ]


def _prophet_forecast(series: pd.Series, periods: int) -> list[dict]:
    from prophet import Prophet

    df = pd.DataFrame({"ds": series.index, "y": series.values})
    model = Prophet()
    model.fit(df)

    freq = pd.infer_freq(series.index) or "MS"
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)

    future_only = forecast.tail(periods)
    return [
        {"date": str(row["ds"].date()), "predicted_value": float(row["yhat"])}
        for _, row in future_only.iterrows()
    ]


def _arima_forecast(series: pd.Series, periods: int, seasonal: bool, seasonal_period: int) -> list[dict]:
    if seasonal:
        model = SARIMAX(
            series, order=(1, 1, 1),
            seasonal_order=(1, 1, 1, seasonal_period),
            enforce_stationarity=False, enforce_invertibility=False,
        )
    else:
        model = ARIMA(series, order=(1, 1, 1))

    fitted = model.fit()
    forecast_values = fitted.forecast(steps=periods)

    freq = pd.infer_freq(series.index) or "MS"
    future_index = pd.date_range(start=series.index[-1], periods=periods + 1, freq=freq)[1:]

    return [
        {"date": str(d.date()), "predicted_value": float(v)}
        for d, v in zip(future_index, forecast_values)
    ]


def generate_forecast(
    result: list[dict],
    date_key: str,
    value_key: str,
    periods: int = 3,
    seasonal_period: int = 12,
) -> dict:
    """
    Detects an appropriate model based on series length/seasonality, fits it,
    and returns predictions with fallback on failure.
    """
    series = _to_series(result, date_key, value_key)
    n = len(series)

    model_used = None
    predictions = None
    warning = None

    if n < 8:
        model_used = "linear_trend"
        predictions = _linear_trend_forecast(series, periods)

    else:
        seasonality_strength = _detect_seasonality_strength(series, seasonal_period)
        strong_seasonality = seasonality_strength > 0.5 and n >= seasonal_period * 2

        if n >= 24:
            try:
                model_used = "sarima" if strong_seasonality else "arima"
                predictions = _arima_forecast(
                    series, periods, seasonal=strong_seasonality, seasonal_period=seasonal_period
                )
            except Exception:
                warning = "ARIMA/SARIMA failed to converge; falling back to Prophet."
                model_used = None

        if predictions is None:
            try:
                model_used = "prophet"
                predictions = _prophet_forecast(series, periods)
            except Exception:
                warning = (warning or "") + " Prophet failed; falling back to linear trend."
                model_used = None

        if predictions is None:
            model_used = "linear_trend"
            predictions = _linear_trend_forecast(series, periods)

    return {
        "model_used": model_used,
        "periods_forecasted": periods,
        "predictions": predictions,
        "warning": warning,
    }

def _identify_columns(result: list[dict]) -> tuple[str, str]:
    """Guess which result column is the date/period and which is the numeric value."""
    if not result:
        raise ValueError("No result rows to forecast from.")

    sample_row = result[0]
    date_key = None
    value_key = None

    for key, value in sample_row.items():
        if date_key is None:
            try:
                pd.to_datetime(str(value))
                date_key = key
                continue
            except (ValueError, TypeError):
                pass
        if value_key is None and isinstance(value, (int, float)) and not isinstance(value, bool):
            value_key = key

    if date_key is None or value_key is None:
        raise ValueError("Could not identify a date column and a numeric value column for forecasting.")

    return date_key, value_key