import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
CSV_PATH = "combined.csv"
HORIZON_DAYS = 7              # predict the price this many days ahead
CUTOFF_DATE = "2026-01-02"    # ignore data after this date
TRAIN_FRACTION = 0.8
MIN_FILLED_RUN = 10           # >= this many days with the exact same daily change = filler data
RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# 1. Preprocessing
# --------------------------------------------------------------------------- #
def load_data(path):
    """Read the CSV, sort oldest -> newest, drop duplicates / incomplete rows."""
    df = pd.read_csv(path, parse_dates=["Date"])
    df = (
        df.sort_values("Date")
        .drop_duplicates(subset="Date")
        .dropna(subset=["Price", "Open", "High", "Low"])
        .reset_index(drop=True)
    )
    return df


def drop_filled_runs(df, min_run=MIN_FILLED_RUN):
    """Remove stretches where the price changes by the exact same amount every day."""
    step = df["Price"].diff()
    run_id = (step != step.shift()).cumsum()
    run_len = step.groupby(run_id).transform("size")
    is_filler = (run_len >= min_run) & step.notna()
    return df.loc[~is_filler].reset_index(drop=True)


def build_features(df):
    """Features describing the situation as of the close of each day."""
    price = df["Price"]
    log_price = np.log(price)
    daily_ret = log_price.diff()

    f = pd.DataFrame(index=df.index)
    for n in (1, 3, 5, 10, 21):
        f[f"ret_{n}d"] = log_price.diff(n)
    f["vol_5d"] = daily_ret.rolling(5).std()
    f["vol_21d"] = daily_ret.rolling(21).std()
    for n in (7, 30, 90):
        f[f"gap_ma{n}"] = price / price.rolling(n).mean() - 1
    f["range_pct"] = (df["High"] - df["Low"]) / price
    f["body_pct"] = (price - df["Open"]) / df["Open"]
    return f


def build_target(df, days_ahead):
    """log(price in `days_ahead` days / price today). NaN if that date is beyond the data."""
    target_date = df["Date"] + pd.Timedelta(days=days_ahead)
    future = df[["Date", "Price"]].rename(columns={"Date": "fdate", "Price": "future_price"})
    merged = pd.merge_asof(
        pd.DataFrame({"Date": target_date}), future,
        left_on="Date", right_on="fdate",
        direction="backward", tolerance=pd.Timedelta(days=5),
    )
    future_price = merged["future_price"].where(target_date <= df["Date"].max())
    return np.log(future_price / df["Price"])


# --------------------------------------------------------------------------- #
# 2. Train / test split (chronological, never shuffled)
# --------------------------------------------------------------------------- #
def split_data(data, days_ahead):
    split_date = data["Date"].iloc[int(len(data) * TRAIN_FRACTION)]
    # gap of `days_ahead` so no training target overlaps the test period
    train = data[data["Date"] + pd.Timedelta(days=days_ahead) < split_date]
    test = data[data["Date"] >= split_date]
    return train, test


# --------------------------------------------------------------------------- #
# 3. Models, training and accuracy
# --------------------------------------------------------------------------- #
def candidate_models():
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=10, max_features=0.5,
            n_jobs=-1, random_state=RANDOM_STATE),
        "GradientBoosting": HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=200,
            min_samples_leaf=30, random_state=RANDOM_STATE),
    }


def accuracy_pct(price_now, y_true, y_pred):
    """Convert log-changes back to prices; accuracy = 100 - average % error."""
    actual = price_now * np.exp(y_true)
    predicted = price_now * np.exp(y_pred)
    mape = np.mean(np.abs(actual - predicted) / actual) * 100
    return 100 - mape


def main():
    # preprocessing
    df = drop_filled_runs(load_data(CSV_PATH))
    df = df[df["Date"] <= CUTOFF_DATE].reset_index(drop=True)

    feats = build_features(df)
    feature_cols = list(feats.columns)
    data = pd.concat([df[["Date", "Price"]], feats], axis=1)
    data["target"] = build_target(df, HORIZON_DAYS)
    data = data.dropna().reset_index(drop=True)

    # splitting
    train, test = split_data(data, HORIZON_DAYS)
    print(f"Horizon: {HORIZON_DAYS} days | Train rows: {len(train)} | Test rows: {len(test)}")

    # training + accuracy on test data
    for name, model in candidate_models().items():
        model.fit(train[feature_cols], train["target"])
        pred = model.predict(test[feature_cols])
        acc = accuracy_pct(test["Price"], test["target"], pred)
        print(f"{name} test accuracy: {acc:.2f}%")


if __name__ == "__main__":
    main()