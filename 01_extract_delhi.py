import subprocess, sys, os
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow.compute as pc

RAW = "raw"
OUT = "data"
os.makedirs(OUT, exist_ok=True)
os.makedirs(RAW, exist_ok=True)

COLS = ["Station ID", "Station Name", "City", "State", "Timestamp",
        "PM2.5 (µg/m³)", "PM10 (µg/m³)", "NO2 (µg/m³)", "CO (mg/m³)",
        "AT (°C)", "RH (%)", "WS (m/s)", "WD (deg)", "RF (mm)",
        "SR (W/mt2)", "BP (mmHg)"]

for year in sys.argv[1:]:
    f = f"{RAW}/cpcb-{year}.parquet"
    if not os.path.exists(f):
        url = f"https://github.com/Vonter/india-cpcb-aqi/releases/download/{year}/cpcb-air-quality-{year}.parquet"
        subprocess.run(["curl", "-sL", "--max-time", "1200", "-o", f, url], check=True)
    dset = ds.dataset(f)
    cols = [c for c in COLS if c in dset.schema.names]
    t = dset.to_table(columns=cols, filter=(pc.field("State") == "Delhi"))
    pq.write_table(t, f"{OUT}/delhi_{year}.parquet")
    print(year, t.num_rows, "rows,", len(set(t.column("Station ID").to_pylist())), "stations")
    os.remove(f)
