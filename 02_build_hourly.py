import glob
import numpy as np
import pandas as pd

files = sorted(glob.glob("data/delhi_*.parquet"))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
df = df.rename(columns={
    "Station ID": "sid", "Timestamp": "ts",
    "PM2.5 (µg/m³)": "pm25", "PM10 (µg/m³)": "pm10", "NO2 (µg/m³)": "no2",
    "CO (mg/m³)": "co", "AT (°C)": "temp", "RH (%)": "rh", "WS (m/s)": "ws",
    "WD (deg)": "wd", "RF (mm)": "rain", "SR (W/mt2)": "sr", "BP (mmHg)": "bp"})

df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)

limits = {"pm25": (2, 999), "pm10": (2, 1500), "no2": (0, 1000), "co": (0, 50),
          "temp": (-2, 50), "rh": (2, 100), "ws": (0, 20), "wd": (0, 360),
          "rain": (0, 200), "sr": (0, 1400), "bp": (650, 800)}
for v, (lo, hi) in limits.items():
    df.loc[(df[v] < lo) | (df[v] > hi), v] = np.nan

df = df.sort_values(["sid", "ts"])
for v in ["pm25", "temp", "rh", "ws", "sr"]:
    same = df.groupby("sid")[v].transform(lambda s: s.groupby((s != s.shift()).cumsum()).transform("size"))
    stuck = (same >= 24) & df[v].notna() & ~((v in ("sr", "rain")) & (df[v] == 0))
    df.loc[stuck, v] = np.nan

df["hour"] = df["ts"].dt.floor("h")
num = list(limits)

df["wu"] = -df["ws"] * np.sin(np.deg2rad(df["wd"]))
df["wv"] = -df["ws"] * np.cos(np.deg2rad(df["wd"]))
st = df.groupby(["sid", "hour"])[num + ["wu", "wv"]].mean()

city = st.groupby("hour").median()
count = st.groupby("hour").count()
city["n_pm"] = count["pm25"]
city["n_met"] = count["ws"]

city.loc[city["n_pm"] < 5, ["pm25", "pm10"]] = np.nan
city.loc[city["n_met"] < 3, ["temp", "rh", "ws", "wd", "sr", "bp", "wu", "wv", "rain"]] = np.nan
city["wd"] = (np.rad2deg(np.arctan2(-city["wu"], -city["wv"])) + 360) % 360

full = pd.date_range(city.index.min(), city.index.max(), freq="h")
city = city.reindex(full)
city.index.name = "time"
city.to_csv("data/delhi_hourly.csv")

print(city.shape, city.index.min(), city.index.max())
print(city[["pm25", "temp", "rh", "ws", "sr", "bp"]].notna().mean().round(3))
print("Mean SR by IST hour (should peak near 12-13 h):")
print(city.groupby(city.index.hour)["sr"].mean().round(0).to_string())
