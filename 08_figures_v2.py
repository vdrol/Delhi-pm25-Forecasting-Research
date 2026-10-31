import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from physics import mixing_height

R = "results"; F = "figures"
BLUE, ORANGE, AQUA, GREY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#b9b8b3",
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.titleweight": "bold",
                     "axes.titlesize": 11, "axes.titlelocation": "left", "figure.dpi": 150,
                     "savefig.bbox": "tight"})

c = pd.read_csv("data/delhi_hourly.csv", index_col=0, parse_dates=True)
c = c.join(mixing_height(c))
e = pd.read_csv("data/era5_delhi_openmeteo.csv", skiprows=3, parse_dates=["time"], index_col="time")
e.columns = ["blh", "ws100", "wd100", "cloud", "t2m"]
j = c.join(e)
w = j[j.index.month.isin([11, 12, 1])]
g = w.groupby(w.index.hour).median()

fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw=dict(wspace=0.3))
ax[0].plot(g.index, g.h_mix, color=ORANGE, lw=2, label="Estimated from ground sensors")
ax[0].plot(g.index, g.blh, color=BLUE, lw=2, label="ERA5 (upper-air data)")
ax[0].set_ylabel("Mixing-layer height (m)"); ax[0].set_xlabel("Hour of day (IST)")
ax[0].set_title("(a) Mixing height: the two agree"); ax[0].set_xticks(range(0, 24, 3))
ax[0].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(0, 0.62))
ax[1].plot(g.index, g.ws, color=ORANGE, lw=2, label="Ground sensors (rooftop / mast)")
ax[1].plot(g.index, g.ws100, color=BLUE, lw=2, label="ERA5 wind at 100 m")
ax[1].set_ylabel("Wind speed (m/s)"); ax[1].set_xlabel("Hour of day (IST)"); ax[1].set_ylim(0, 5)
ax[1].set_title("(b) Wind: ground sensors miss it"); ax[1].set_xticks(range(0, 24, 3))
ax[1].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(0, 0.45))
fig.suptitle("Fig. 3  Checking the ground data against ERA5 (winter medians, Nov–Jan 2018–2025)",
             x=0.06, ha="left", fontweight="bold", y=1.03)
fig.savefig(f"{F}/v2_fig3_data_check.png"); plt.close(fig)

s = pd.read_csv(f"{R}/v2_scores.csv")
lab = {"M0 Persistence": "Persistence", "N3 ERA5 box model only": "Physics only (ERA5 box model)",
       "M2 AI only (ground sensors)": "AI only (ground sensors)", "M3 AI + ground physics": "AI + ground physics",
       "N1 AI + ERA5 data": "AI + ERA5 data", "N2 AI + ERA5 physics": "AI + ERA5 data + physics",
       "N4 AI corrects ERA5 box model": "AI correcting ERA5 box model"}
v = s[s.subset == "winter"].set_index("model").RMSE.rename(index=lab).sort_values()
col = {"AI + ERA5 data": BLUE, "AI + ERA5 data + physics": AQUA, "AI correcting ERA5 box model": AQUA,
       "AI only (ground sensors)": GREY, "AI + ground physics": ORANGE}
fig, ax = plt.subplots(figsize=(6.8, 3.4))
ax.barh(v.index, v.values, color=[col.get(i, "#c9c8c3") for i in v.index], height=0.6)
for i, x in enumerate(v.values):
    ax.text(x + 0.6, i, f"{x:.1f}", va="center", fontsize=9, color=INK)
ax.invert_yaxis(); ax.grid(axis="y", visible=False); ax.set_xlim(0, v.max() * 1.12)
ax.set_xlabel("Error, RMSE (µg/m³) — lower is better")
ax.set_title("Fig. 4  New upper-air data helped; physics formulas did not")
fig.savefig(f"{F}/v2_fig4_scores.png"); plt.close(fig)

a = json.load(open(f"{R}/v2_ablation.json"))
order = ["AI only (M2)", "+ boundary-layer height only", "+ cloud cover only", "+ wind speed at 100 m only",
         "+ wind direction at 100 m only", "+ wind at 100 m only", "+ all ERA5 data (N1)"]
names = ["AI only (ground sensors)", "+ boundary-layer height", "+ cloud cover", "+ wind speed at 100 m",
         "+ wind direction at 100 m", "+ wind speed and direction", "+ all ERA5 data"]
m = [a[k][0] for k in order]; sd = [a[k][1] for k in order]
fig, ax = plt.subplots(figsize=(6.8, 3.4))
cols = [GREY, "#c9c8c3", "#c9c8c3", BLUE, BLUE, BLUE, BLUE]
y = np.arange(len(names))
ax.errorbar(m, y, xerr=sd, fmt="none", ecolor=INK2, lw=1, capsize=3)
ax.scatter(m, y, s=70, color=cols, zorder=3, edgecolor="white", linewidth=1.5)
ax.set_yticks(y); ax.set_yticklabels(names)
for i, x in enumerate(m):
    ax.text(x + sd[i] + 0.4, i, f"{x:.1f}", va="center", fontsize=9)
ax.axvline(m[0], color=INK2, ls="--", lw=1)
ax.invert_yaxis(); ax.grid(axis="y", visible=False); ax.set_xlim(63, 71)
ax.set_xlabel("Winter RMSE (µg/m³); dot = mean of 5 runs, line = ± 1 SD")
ax.set_title("Fig. 5  The gain comes from wind above the rooftops")
fig.savefig(f"{F}/v2_fig5_ablation.png"); plt.close(fig)

p = pd.read_csv(f"{R}/v2_test_predictions.csv", index_col=0, parse_dates=["target_time"]).set_index("target_time")
ep = p.loc["2024-11-08":"2024-11-28"]
fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.plot(ep.index, np.exp(ep.obs), color=INK, lw=2, label="Observed")
ax.plot(ep.index, np.exp(ep["M0 Persistence"]), color=GREY, lw=1.2, label="Persistence")
ax.plot(ep.index, np.exp(ep["M2 AI only (ground sensors)"]), color=ORANGE, lw=1.6, label="AI only (ground sensors)")
ax.plot(ep.index, np.exp(ep["N1 AI + ERA5 data"]), color=BLUE, lw=1.6, label="AI + ERA5 data")
ax.set_ylabel("PM2.5 (µg/m³)")
ax.set_title("Fig. 6  24-hour-ahead forecasts, November 2024 smog episode")
ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2))
fig.autofmt_xdate()
fig.savefig(f"{F}/v2_fig6_episode.png"); plt.close(fig)
print("v2 figures written")
