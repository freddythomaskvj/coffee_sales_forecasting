import os
import requests
import pandas as pd
import holidays
from datetime import date

# -------------------------------
# 0. Paths
# -------------------------------
DATA_PATH = "data/transactions.csv"
OUTPUT_DIR = "output"
FEATURES_PATH = os.path.join(OUTPUT_DIR, "daily_features.csv")

# -------------------------------
# 1. Load transaction data
# -------------------------------
df = pd.read_csv(DATA_PATH)

# Your dates are like 01-01-2023 → use dayfirst=True
df["transaction_date"] = pd.to_datetime(df["transaction_date"], dayfirst=True)

# Sales amount
df["sales_amount"] = df["transaction_qty"] * df["unit_price"]

# -------------------------------
# 2. Aggregate to daily per store/product
# -------------------------------
daily = (
    df.groupby(
        ["transaction_date", "store_id", "store_location", "product_id"],
        as_index=False,
    )
    .agg(
        daily_qty=("transaction_qty", "sum"),
        daily_sales_amount=("sales_amount", "sum"),
        transactions_count=("transaction_id", "nunique"),
    )
)

# -------------------------------
# 3. Calendar features (India)
# -------------------------------
def add_calendar_features(df_in: pd.DataFrame, date_col: str = "transaction_date"):
    df_out = df_in.copy()
    d = df_out[date_col]

    df_out["day_of_week"] = d.dt.dayofweek       # 0=Mon, 6=Sun
    df_out["day_name"] = d.dt.day_name()
    df_out["is_weekend"] = df_out["day_of_week"].isin([5, 6]).astype(int)

    df_out["day"] = d.dt.day
    df_out["month"] = d.dt.month
    df_out["month_name"] = d.dt.month_name()
    df_out["quarter"] = d.dt.quarter
    df_out["year"] = d.dt.year
    df_out["is_month_end"] = d.dt.is_month_end.astype(int)

    # Simple Indian seasons
    def month_to_season(m):
        if m in [12, 1, 2]:
            return "Winter"
        elif m in [3, 4, 5]:
            return "Summer"
        elif m in [6, 7, 8, 9]:
            return "Monsoon"
        else:
            return "Post-monsoon"

    df_out["season"] = df_out["month"].apply(month_to_season)

    return df_out


daily = add_calendar_features(daily, "transaction_date")

# -------------------------------
# 4. Holidays (India + State specific)
# -------------------------------
india_holidays = holidays.country_holidays("IN")

store_state_code = {
    "Delhi": "DL",
    "Cochin": "KL",
    "Bengaluru": "KA",
}

state_holidays = {
    loc: holidays.country_holidays("IN", subdiv=code)
    for loc, code in store_state_code.items()
}

def add_holiday_features(df_in: pd.DataFrame):
    df_out = df_in.copy()

    is_holiday_list = []
    holiday_name_list = []

    for _, row in df_out.iterrows():
        dt: date = row["transaction_date"].date()
        loc = row["store_location"]

        nat_name = india_holidays.get(dt)
        st_obj = state_holidays.get(loc)
        st_name = st_obj.get(dt) if st_obj is not None else None

        final_name = st_name or nat_name
        is_holiday_list.append(1 if final_name else 0)
        holiday_name_list.append(final_name)

    df_out["is_holiday"] = is_holiday_list
    df_out["holiday_name"] = holiday_name_list

    return df_out


daily = add_holiday_features(daily)

# -------------------------------
# 5. Weather (Open-Meteo)
# -------------------------------
store_coords = {
    "Delhi": {"lat": 28.7041, "lon": 77.1025},
    "Cochin": {"lat": 9.9312, "lon": 76.2673},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946},
}

start_date = daily["transaction_date"].min().date()
end_date = daily["transaction_date"].max().date()

print(f"Fetching weather from {start_date} to {end_date}...")

def fetch_weather_for_location(location_name: str, lat: float, lon: float):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "windspeed_10m_max",
        ],
        "timezone": "Asia/Kolkata",
    }

    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data_json = resp.json()

    daily_data = data_json.get("daily", {})
    if "time" not in daily_data:
        raise ValueError(f"No daily data returned for {location_name}")

    wdf = pd.DataFrame(daily_data)
    wdf.rename(columns={"time": "transaction_date"}, inplace=True)
    wdf["transaction_date"] = pd.to_datetime(wdf["transaction_date"])
    wdf["store_location"] = location_name

    return wdf


weather_frames = []
for loc, coords in store_coords.items():
    print(f" - Fetching weather for {loc}...")
    wdf = fetch_weather_for_location(loc, coords["lat"], coords["lon"])
    weather_frames.append(wdf)

weather_all = pd.concat(weather_frames, ignore_index=True)

# -------------------------------
# 6. Merge all feature data
# -------------------------------
daily_features = daily.merge(
    weather_all,
    on=["transaction_date", "store_location"],
    how="left",
)

# -------------------------------
# 7. Save file
# -------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
daily_features.to_csv(FEATURES_PATH, index=False)

print("✅ Feature building complete.")
print(f"Saved to {FEATURES_PATH}")
print("Rows:", len(daily_features))
print("Columns:", list(daily_features.columns))

