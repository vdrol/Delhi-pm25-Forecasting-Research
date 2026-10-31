# Delhi winter smog: AI forecasts with ground sensors and ERA5 upper-air data

## Setup (once)
    pip install -r requirements.txt
Open a terminal **inside this folder** and run every command from here.

## Quick start (uses the data already in this zip, ~10 minutes)
    python 03_models.py        # study 1: AI vs ground physics      -> results/scores.csv
    python 04_experiments.py   # study 1 stress tests               -> results/experiments.json
    python 05_figures.py       # study 1 figures                    -> figures/
    python 06_era5_study.py    # study 2: ERA5 upper-air data       -> results/v2_scores.csv, v2_summary.json
    python 07_ablation.py      # which ERA5 variable helps          -> results/v2_ablation.json
    python 08_figures_v2.py    # study 2 figures                    -> figures/v2_*.png

## Full rebuild from raw data (optional, downloads ~5 GB, needs internet)
    python 01_extract_delhi.py 2018 2019 2020 2021 2022 2023 2024 2025
    python 02_build_hourly.py

ERA5 file (data/era5_delhi_openmeteo.csv) came from the Open-Meteo archive:
https://archive-api.open-meteo.com/v1/archive?latitude=28.61&longitude=77.21&start_date=2018-01-01&end_date=2025-12-31&hourly=boundary_layer_height,wind_speed_100m,wind_direction_100m,cloud_cover,temperature_2m&timezone=Asia%2FKolkata&wind_speed_unit=ms&format=csv

## Files
- physics.py   mixing height and ventilation from ground sensors
- common.py    shared data preparation and model helpers (imported by 03-07)
Results are deterministic: re-running gives the same numbers.

## Data sources and licences
- data/delhi_hourly.csv is derived from Central Pollution Control Board data archived at
  https://github.com/Vonter/india-cpcb-aqi, licensed under the Open Database License (ODbL 1.0).
- data/era5_delhi_openmeteo.csv contains ERA5 reanalysis data (Copernicus Climate Change Service)
  downloaded from Open-Meteo (https://open-meteo.com).
