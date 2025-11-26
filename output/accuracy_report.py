import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

PRED_PATH = "output/predictions_daily.csv"
OUT_PATH = "output/accuracy_report.csv"

df = pd.read_csv(PRED_PATH)

# Only historical rows
hist = df[df["is_future"] == 0].dropna(subset=["actual_qty"])

# --- GLOBAL ACCURACY ---
mae = mean_absolute_error(hist["actual_qty"], hist["predicted_qty"])
rmse = mean_squared_error(hist["actual_qty"], hist["predicted_qty"], squared=False)
mape = (abs(hist["actual_qty"] - hist["predicted_qty"]) / hist["actual_qty"]).mean() * 100

summary = {
    "Metric": ["MAE", "RMSE", "MAPE (%)"],
    "Value": [mae, rmse, mape]
}

summary_df = pd.DataFrame(summary)

# --- PRODUCT-WISE MAPE ---
prod_mape = hist.groupby(["store_id", "product_id"]).apply(
    lambda x: (abs(x["actual_qty"] - x["predicted_qty"]) / x["actual_qty"]).mean() * 100
).reset_index()

prod_mape.columns = ["store_id", "product_id", "MAPE (%)"]

# Combine both
final_df = pd.concat([
    summary_df,
    pd.DataFrame({"Metric": [], "Value": []}),  # blank separator
    prod_mape
], ignore_index=True)

final_df.to_csv(OUT_PATH, index=False)

print("Accuracy report generated:")
print(f"Saved to: {OUT_PATH}")
