import os
import pandas as pd
from prophet import Prophet

# ---------------------------------------
# 1. Paths
# ---------------------------------------
FEATURES_PATH = "output/daily_features.csv"
OUTPUT_DIR = "output"

# ---------------------------------------
# 2. Load feature data
# ---------------------------------------
df = pd.read_csv(FEATURES_PATH)
df["transaction_date"] = pd.to_datetime(df["transaction_date"])

# ---------------------------------------
# Helper function
# ---------------------------------------
def prepare_df(group):
    dfp = group.rename(columns={
        "transaction_date": "ds",
        "daily_qty": "y"
    })

    dfp["temp_max"] = dfp["temperature_2m_max"]
    dfp["temp_min"] = dfp["temperature_2m_min"]
    dfp["rain"] = dfp["precipitation_sum"]
    dfp["dow"] = dfp["day_of_week"]
    dfp["holiday_flag"] = dfp["is_holiday"]

    return dfp[["ds", "y", "temp_max", "temp_min", "rain", "dow", "holiday_flag"]]

# ---------------------------------------
# 3. Storage for combined output
# ---------------------------------------
combined_output = []

# ---------------------------------------
# 4. Forecast per store + product
# ---------------------------------------
groups = df.groupby(["store_id", "product_id"])

for (store, product), g in groups:
    g = g.sort_values("transaction_date")
    d = prepare_df(g)

    # Train model
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False
    )

    model.add_regressor("temp_max")
    model.add_regressor("temp_min")
    model.add_regressor("rain")
    model.add_regressor("dow")
    model.add_regressor("holiday_flag")

    model.fit(d)

    # ---------------------------
    # Historical predictions
    # ---------------------------
    hist_future = model.make_future_dataframe(periods=0)

    hist_reg = d.set_index("ds").reindex(hist_future["ds"]).reset_index()
    hist_pred = model.predict(hist_reg)

    hist_df = pd.DataFrame({
        "date": hist_pred["ds"],
        "predicted_qty": hist_pred["yhat"],
        "actual_qty": g.set_index("transaction_date")["daily_qty"].reindex(hist_pred["ds"]).values,
        "store_id": store,
        "product_id": product,
        "is_future": 0
    })

    combined_output.append(hist_df)

    # ---------------------------
    # Future forecast (30 days)
    # ---------------------------
    future_days = 30
    future_df = model.make_future_dataframe(periods=future_days)

    last = d.iloc[-1]

    future_extra = pd.DataFrame({
        "ds": future_df["ds"].tail(future_days),
        "temp_max": last["temp_max"],
        "temp_min": last["temp_min"],
        "rain": last["rain"],
        "dow": future_df["ds"].tail(future_days).dt.dayofweek,
        "holiday_flag": 0
    })

    future_pred = model.predict(future_extra)

    future_df_out = pd.DataFrame({
        "date": future_pred["ds"],
        "predicted_qty": future_pred["yhat"],
        "actual_qty": None,
        "store_id": store,
        "product_id": product,
        "is_future": 1
    })

    combined_output.append(future_df_out)

# ---------------------------------------
# 5. Save combined output
# ---------------------------------------
combined_final = pd.concat(combined_output)
combined_final.to_csv(f"{OUTPUT_DIR}/predictions_daily.csv", index=False)

print("Forecasting complete.")
print("Saved: output/predictions_daily.csv")
