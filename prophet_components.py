import os
import pandas as pd
from prophet import Prophet

FEATURES_PATH = "output/daily_features.csv"
OUTPUT_DIR = "output"
OUT_PATH = os.path.join(OUTPUT_DIR, "prophet_components.csv")

# 1. Load features
df = pd.read_csv(FEATURES_PATH)
df["transaction_date"] = pd.to_datetime(df["transaction_date"])

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

all_components = []

groups = df.groupby(["store_id", "product_id"])

for (store, product), g in groups:
    g = g.sort_values("transaction_date")
    d = prepare_df(g)

    # 2. Train Prophet model
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

    # 3. Build full (history + 30-day future) dataframe
    future_days = 30
    full_future = model.make_future_dataframe(periods=future_days)

    hist_len = len(d)

    # Build regressors for full period
    last = d.iloc[-1]

    reg_df = pd.DataFrame({"ds": full_future["ds"]})

    # For historical part, use actual regressors
    hist_part = d.set_index("ds").reindex(full_future["ds"]).reset_index()

    reg_df["temp_max"] = hist_part["temp_max"].fillna(last["temp_max"])
    reg_df["temp_min"] = hist_part["temp_min"].fillna(last["temp_min"])
    reg_df["rain"] = hist_part["rain"].fillna(last["rain"])
    reg_df["dow"] = reg_df["ds"].dt.dayofweek
    reg_df["holiday_flag"] = hist_part["holiday_flag"].fillna(0)

    # 4. Predict with full components
    forecast = model.predict(reg_df)

    # 5. Extract components we care about
    comp = forecast[[
        "ds",
        "yhat",
        "trend",
        "weekly",
        "temp_max",
        "temp_min",
        "rain",
        "dow",
        "holiday_flag"
    ]].copy()

    comp["store_id"] = store
    comp["product_id"] = product

    # is_future flag
    comp["is_future"] = (comp.index >= hist_len).astype(int)

    comp = comp.rename(columns={"ds": "date", "yhat": "predicted_qty"})

    all_components.append(comp)

# 6. Save all components
os.makedirs(OUTPUT_DIR, exist_ok=True)
final_components = pd.concat(all_components)
final_components.to_csv(OUT_PATH, index=False)

print("Prophet components exported.")
print(f"Saved to: {OUT_PATH}")
