import sys, io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("06_era5_study.py").read().split("P = pd.DataFrame(index=TE.index)")[0])
groups = {"+ boundary-layer height only": ["e_blh_now", "e_blh_fut"],
          "+ wind at 100 m only": ["e_ws100_now", "e_ws100_fut", "e_u100_now", "e_u100_fut", "e_v100_now", "e_v100_fut"],
          "+ cloud cover only": ["e_cloud_now", "e_cloud_fut"],
          "+ all ERA5 data (N1)": ERA_RAW,
          "AI only (M2)": []}
TEw = TE[TE.target_time.dt.month.isin([10, 11, 12, 1, 2])]
out = {}
for g, f in groups.items():
    r = [metrics(TEw.y.values, predict(train_lgb(TR, VA, RAW_FEATS + f, s), TEw, RAW_FEATS + f))["RMSE"] for s in [0, 1, 2, 3, 4]]
    out[g] = (float(np.mean(r)), float(np.std(r)))
    print(g, round(out[g][0], 1), "+/-", round(out[g][1], 1))
json.dump(out, open(f"{OUT}/v2_ablation.json", "w"), indent=1)

for g, f in {"+ wind speed at 100 m only": ["e_ws100_now", "e_ws100_fut"],
             "+ wind direction at 100 m only": ["e_u100_now", "e_u100_fut", "e_v100_now", "e_v100_fut"]}.items():
    r = [metrics(TEw.y.values, predict(train_lgb(TR, VA, RAW_FEATS + f, s), TEw, RAW_FEATS + f))["RMSE"] for s in [0, 1, 2, 3, 4]]
    out[g] = (float(np.mean(r)), float(np.std(r))); print(g, round(out[g][0], 1), "+/-", round(out[g][1], 1))

j = c.join(e[["ws100", "u100", "v100"]])
gr = {"r_speed": float(j.ws.corr(j.ws100)), "r_u": float(j.wu.corr(j.u100)), "r_v": float(j.wv.corr(j.v100)),
      "median_ground_ws": float(j.ws.median()), "median_ws100": float(j.ws100.median())}
w = j[j.index.month.isin([10, 11, 12, 1, 2])]
gr["winter_median_ground_ws"] = float(w.ws.median()); gr["winter_median_ws100"] = float(w.ws100.median())
print(gr); out["ground_vs_100m_wind"] = gr
json.dump(out, open(f"{OUT}/v2_ablation.json", "w"), indent=1)
