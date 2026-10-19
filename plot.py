import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gold_pred_lr as gf

HORIZONS = {"1 day": 1, "1 week": 7, "1 month": 30}
gf.OUTPUT_DIR.mkdir(exist_ok=True)


def prepare(days):
    df = gf.drop_filled_runs(gf.load_data(gf.CSV_PATH))
    df = df[df["Date"] <= "2026-01-02"].reset_index(drop=True)
    feats = gf.build_features(df)
    cols = list(feats.columns)
    data = pd.concat([df[["Date", "Price"]], feats], axis=1)
    data["target"] = gf.build_target(df, days)
    data = data.dropna().reset_index(drop=True)
    return data, cols


acc_model, acc_naive, best_names = [], [], []
for label, days in HORIZONS.items():
    data, cols = prepare(days)
    best, table, residuals, test = gf.evaluate(data, cols, days)
    acc_model.append(table.loc[best, "Accuracy_%"])
    acc_naive.append(table.loc["Naive (no change)", "Accuracy_%"])
    best_names.append(best)

    if days == 7:  # plots 1, 3, 4 use the 1-week horizon
        # 1. Actual vs predicted
        when = test["Date"] + pd.Timedelta(days=days)
        actual = test["Price"] * np.exp(test["target"])
        predicted = test["Price"] * np.exp(test["pred"])
        plt.figure(figsize=(10, 5))
        plt.plot(when, actual, label="Actual price", color="black")
        plt.plot(when, predicted, label=f"Predicted ({best})", color="tab:orange")
        plt.title("Actual vs predicted gold price (1-week horizon, test period)")
        plt.xlabel("Date"); plt.ylabel("Price (Rs per 10 g)")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(gf.OUTPUT_DIR / "1_actual_vs_predicted.png", dpi=200); plt.close()

        # 3. Feature importance (Random Forest, fitted on all data)
        rf = gf.candidate_models()["RandomForest"].fit(data[cols], data["target"])
        imp = pd.Series(rf.feature_importances_, index=cols).sort_values()
        plt.figure(figsize=(8, 6))
        (imp * 100).plot.barh(color="tab:blue")
        plt.title("Random Forest feature importance (1-week horizon)")
        plt.xlabel("Importance (%)"); plt.tight_layout()
        plt.savefig(gf.OUTPUT_DIR / "3_feature_importance.png", dpi=200); plt.close()

        # 4. Residual histogram with 10th-90th percentile band
        lo, hi = np.percentile(residuals, [10, 90])
        plt.figure(figsize=(8, 5))
        plt.hist(residuals, bins=40, color="tab:green", alpha=0.8)
        plt.axvline(lo, color="red", linestyle="--", label="10th percentile")
        plt.axvline(hi, color="red", linestyle="--", label="90th percentile")
        plt.title("Test errors (log-change) used for the 80% likely range")
        plt.xlabel("Actual - predicted log-change"); plt.ylabel("Count")
        plt.legend(); plt.tight_layout()
        plt.savefig(gf.OUTPUT_DIR / "4_error_histogram.png", dpi=200); plt.close()

# 2. Accuracy by horizon vs naive baseline
x = np.arange(len(HORIZONS)); w = 0.35
plt.figure(figsize=(8, 5))
plt.bar(x - w / 2, acc_model, w, label="Best model", color="tab:orange")
plt.bar(x + w / 2, acc_naive, w, label="Naive (no change)", color="gray")
plt.xticks(x, list(HORIZONS)); plt.ylim(min(acc_model + acc_naive) - 2, 100)
plt.ylabel("Price accuracy (%)")
plt.title("Accuracy by forecast horizon: model vs naive baseline")
plt.legend(); plt.tight_layout()
plt.savefig(gf.OUTPUT_DIR / "2_accuracy_by_horizon.png", dpi=200); plt.close()
print("Saved plots to", gf.OUTPUT_DIR)