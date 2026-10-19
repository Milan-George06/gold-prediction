from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
CSV_PATH = "combined.csv"
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")
HORIZON_ALIASES = {"day": 1, "week": 7, "month": 30}
TRAIN_FRACTION = 0.8
MIN_FILLED_RUN = 10      # >= this many days with the exact same daily change = filler data
MAX_RELIABLE_DAYS = 90     # warn beyond this horizon
RANDOM_STATE = 42

log = logging.getLogger("gold_forecast")


# --------------------------------------------------------------------------- #
# 1. Data loading & cleaning
# --------------------------------------------------------------------------- #
def load_data(path: str) -> pd.DataFrame:
    """Read the CSV, sort oldest -> newest, drop duplicates / incomplete rows."""
    df = pd.read_csv(path, parse_dates=["Date"])
    n_raw = len(df)

    df = (
        df.sort_values("Date")
        .drop_duplicates(subset="Date")
        .dropna(subset=["Price", "Open", "High", "Low"])
        .reset_index(drop=True)
    )
    log.info("Loaded %d rows (%d removed as duplicate/incomplete): %s -> %s",
             len(df), n_raw - len(df), df["Date"].min().date(), df["Date"].max().date())
    return df


def drop_filled_runs(df: pd.DataFrame, min_run: int = MIN_FILLED_RUN) -> pd.DataFrame:
    """
    Remove stretches where the price moves by the exact same amount every day.
    Real markets never do that - it means the values were interpolated / copied in.
    """
    step = df["Price"].diff()
    run_id = (step != step.shift()).cumsum()
    run_len = step.groupby(run_id).transform("size")
    is_filler = (run_len >= min_run) & step.notna()

    if is_filler.any():
        log.warning(
            "Dropping %d rows that look like filler data (constant daily change). "
            "Check your data source for: %s",
            is_filler.sum(),
            ", ".join(f"{g['Date'].min().date()} -> {g['Date'].max().date()}"
                      for _, g in df[is_filler].groupby(run_id[is_filler])),
        )
    return df.loc[~is_filler].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. Features & target
# --------------------------------------------------------------------------- #
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features describing 'the situation as of the close of each day'."""
    price = df["Price"]
    log_price = np.log(price)
    daily_ret = log_price.diff()

    f = pd.DataFrame(index=df.index)
    for n in (1, 3, 5, 10, 21):
        f[f"ret_{n}d"] = log_price.diff(n)                       # recent momentum
    f["vol_5d"] = daily_ret.rolling(5).std()                     # recent choppiness
    f["vol_21d"] = daily_ret.rolling(21).std()
    for n in (7, 30, 90):
        f[f"gap_ma{n}"] = price / price.rolling(n).mean() - 1    # stretched vs. average?
    f["range_pct"] = (df["High"] - df["Low"]) / price            # today's range
    f["body_pct"] = (price - df["Open"]) / df["Open"]            # open -> close move
    return f


def build_target(df: pd.DataFrame, days_ahead: int) -> pd.Series:
    """
    log(price in `days_ahead` calendar days / price today), for every row.
    Rows whose target date is beyond the data are NaN (never guessed).
    """
    target_date = df["Date"] + pd.Timedelta(days=days_ahead)
    future = df[["Date", "Price"]].rename(columns={"Date": "fdate", "Price": "future_price"})
    merged = pd.merge_asof(
        pd.DataFrame({"Date": target_date}), future,
        left_on="Date", right_on="fdate",
        direction="backward", tolerance=pd.Timedelta(days=5),   # weekends / holidays
    )
    future_price = merged["future_price"].where(target_date <= df["Date"].max())
    return np.log(future_price / df["Price"])


# --------------------------------------------------------------------------- #
# 3. Models & evaluation
# --------------------------------------------------------------------------- #
def candidate_models() -> dict:
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=10, max_features=0.5,
            n_jobs=-1, random_state=RANDOM_STATE),
        "GradientBoosting": HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=200,
            min_samples_leaf=30, random_state=RANDOM_STATE),
    }


def price_metrics(price_now: pd.Series, y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Convert predicted log-changes back to prices and score them."""
    actual = price_now * np.exp(y_true)
    predicted = price_now * np.exp(y_pred)
    mape = float(np.mean(np.abs(actual - predicted) / actual) * 100)
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAPE_%": mape,
        "Accuracy_%": 100 - mape,     # how close the predicted price was, on average
        "Direction_%": float(np.mean(np.sign(y_true) == np.sign(y_pred)) * 100),
    }


def evaluate(data: pd.DataFrame, feature_cols: list[str], days_ahead: int):
    """
    Chronological hold-out test (never shuffled - that would let the model peek at the future).
    A `days_ahead` gap is left between train and test so no training target overlaps the test period.
    """
    split_date = data["Date"].iloc[int(len(data) * TRAIN_FRACTION)]
    train = data[data["Date"] + pd.Timedelta(days=days_ahead) < split_date]
    test = data[data["Date"] >= split_date]
    log.info("Train: %d rows (to %s) | Test: %d rows (from %s)",
             len(train), train["Date"].max().date(), len(test), split_date.date())

    results, fitted, test_preds = {}, {}, {}
    # Baseline: "price won't change" - any useful model must beat this.
    results["Naive (no change)"] = price_metrics(
        test["Price"], test["target"], np.zeros(len(test)))
    results["Naive (no change)"]["Direction_%"] = np.nan   # "no change" has no direction

    for name, model in candidate_models().items():
        model.fit(train[feature_cols], train["target"])
        pred = model.predict(test[feature_cols])
        results[name] = price_metrics(test["Price"], test["target"], pred)
        fitted[name], test_preds[name] = model, pred

    table = pd.DataFrame(results).T.round(2)
    log.info("Test accuracy (%d-day horizon):\n%s", days_ahead, table[["Accuracy_%"]].to_string())

    best = min(fitted, key=lambda m: results[m]["MAE"])
    if results[best]["MAE"] >= results["Naive (no change)"]["MAE"]:
        log.warning("The best model does NOT beat the naive 'no change' baseline at this "
                    "horizon. Treat the forecast as a rough guide only.")
    residuals = test["target"].to_numpy() - test_preds[best]
    return best, table, residuals, test.assign(pred=test_preds[best])


# --------------------------------------------------------------------------- #
# 4. Forecast
# --------------------------------------------------------------------------- #
def forecast(csv_path: str, ahead_text: str | None) -> dict:
    df = drop_filled_runs(load_data(csv_path))
    df = df[df["Date"] <= "2026-01-02"].reset_index(drop=True)
    last_date, last_price = df["Date"].iloc[-1], df["Price"].iloc[-1]

    days_ahead = resolve_horizon(ahead_text, last_date)
    target_date = last_date + pd.Timedelta(days=days_ahead)
    if days_ahead > MAX_RELIABLE_DAYS:
        log.warning("%d days is a long horizon - expect wide uncertainty.", days_ahead)
    if (pd.Timestamp.today().normalize() - last_date).days > 7:
        log.warning("Latest data is from %s. The forecast starts from that date, not today - "
                    "append recent prices to the CSV for an up-to-date forecast.", last_date.date())

    feats = build_features(df)
    feature_cols = list(feats.columns)
    data = pd.concat([df[["Date", "Price"]], feats], axis=1)
    data["target"] = build_target(df, days_ahead)

    latest_row = data.iloc[[-1]]                       # today's snapshot (no target yet)
    data = data.dropna().reset_index(drop=True)        # rows with full features AND a known outcome

    best_name, table, residuals, _ = evaluate(data, feature_cols, days_ahead)

    # Refit the winner on ALL available history, then predict from the latest snapshot.
    final_model = candidate_models()[best_name].fit(data[feature_cols], data["target"])
    pred_change = float(final_model.predict(latest_row[feature_cols])[0])
    lo_q, hi_q = np.percentile(residuals, [10, 90])    # typical error band from the test set

    result = {
        "model": best_name,
        "last_date": last_date,
        "last_price": float(last_price),
        "target_date": target_date,
        "days_ahead": days_ahead,
        "forecast": last_price * np.exp(pred_change),
        "low": last_price * np.exp(pred_change + lo_q),
        "high": last_price * np.exp(pred_change + hi_q),
        "metrics": table,
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": final_model, "features": feature_cols, "days_ahead": days_ahead},
                MODEL_DIR / f"{best_name.lower()}_{days_ahead}d.joblib")
    return result


def resolve_horizon(text: str | None, last_date: pd.Timestamp) -> int:
    if text is None:
        text = input("Forecast how far ahead? (day / week / month / number of days / YYYY-MM-DD): ")
    text = text.strip().lower()

    if text in HORIZON_ALIASES:
        return HORIZON_ALIASES[text]
    if text.isdigit() and int(text) > 0:
        return int(text)
    try:
        days = (pd.Timestamp(text) - last_date).days
    except ValueError:
        raise SystemExit(f"Could not understand '{text}'. Use day, week, month, a number, or YYYY-MM-DD.")
    if days <= 0:
        raise SystemExit(f"{text} is not after the last date in the data ({last_date.date()}).")
    return days


def print_report(r: dict) -> None:
    change = (r["forecast"] / r["last_price"] - 1) * 100
    m = r["metrics"].loc[r["model"]]
    naive = r["metrics"].loc["Naive (no change)"]

    print("\n" + "=" * 56)
    print(f" Gold price forecast  ({r['model']}, {r['days_ahead']}-day horizon)")
    print("=" * 56)
    print(f" Last known price  {r['last_date'].date()} : {r['last_price']:>12,.0f}")
    print(f" Forecast for      {r['target_date'].date()} : {r['forecast']:>12,.0f}  ({change:+.2f}%)")
    print(f" Likely range (80%)             : {r['low']:>12,.0f}  to  {r['high']:,.0f}")
    print("-" * 56)
    print(f" Price accuracy (test)          : {m['Accuracy_%']:>11.2f}%")

    print("=" * 56 + "\n")


# --------------------------------------------------------------------------- #
# 5. Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast the gold price N days ahead.")
    parser.add_argument("--csv", default=CSV_PATH, help="path to the price CSV")
    parser.add_argument("--ahead", help="day | week | month | <number of days> | YYYY-MM-DD")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)
    print_report(forecast(args.csv, args.ahead))


if __name__ == "__main__":
    main()