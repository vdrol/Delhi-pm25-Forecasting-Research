import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from physics import mixing_height

OUT = "figures"
import os; os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#b9b8b3",
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                     "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left",
                     "figure.dpi": 150, "savefig.bbox": "tight"})

c = pd.read_csv("data/delhi_hourly.csv", index_col=0, parse_dates=True)
c = c.join(mixing_height(c))
win = c[c.index.month.isin([11, 12, 1])]

g = win.groupby(win.index.hour)
fig, ax = plt.subplots(2, 1, figsize=(6.5, 5), sharex=True)
for a, col, lab, colr in [(ax[0], "pm25", "PM2.5 (µg/m³)", ORANGE), (ax[1], "h_mix", "Mixing height (m)", BLUE)]:
    med = g[col].median(); q1 = g[col].quantile(0.25); q3 = g[col].quantile(0.75)
    a.fill_between(med.index, q1, q3, color=colr, alpha=0.18, lw=0)
    a.plot(med.index, med, color=colr, lw=2)
    a.set_ylabel(lab)
ax[0].set_title("Fig. 1  Winter day in Delhi: pollution falls when the mixing layer grows")
ax[1].set_xlabel("Hour of day (IST)")
ax[1].set_xticks(range(0, 24, 3))
ax[1].text(0.2, 300, "night: shallow\ninversion layer", color=INK2, fontsize=9)
ax[1].text(17.6, 300, "evening: layer\ncollapses", color=INK2, fontsize=9)
ax[0].text(0.3, ax[0].get_ylim()[0] + 8, "line = median, band = middle 50% (Nov–Jan, 2018–2025)", color=INK2, fontsize=8)
fig.savefig(f"{OUT}/fig1_diurnal.png"); plt.close(fig)

d = c[["pm25", "vc"]].resample("D").mean().dropna()
d = d[d.index.month.isin([10, 11, 12, 1, 2])]
x, y = np.log(d.vc), np.log(d.pm25)
b, a0 = np.polyfit(x, y, 1)
r = np.corrcoef(x, y)[0, 1]
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.scatter(d.vc, d.pm25, s=9, color=BLUE, alpha=0.35, lw=0)
xs = np.linspace(x.min(), x.max(), 50)
ax.plot(np.exp(xs), np.exp(a0 + b * xs), color=INK, lw=2)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Daily mean ventilation coefficient, h × U (m²/s, log scale)")
ax.set_ylabel("Daily mean PM2.5 (µg/m³, log scale)")
ax.set_title("Fig. 2  Less ventilation, more smog (winter days, Oct–Feb)")
ax.text(0.98, 0.95, f"slope = {b:.2f}  (box-model theory: −1)\nr = {r:.2f},  n = {len(d)} days",
        transform=ax.transAxes, ha="right", va="top", color=INK, fontsize=9)
fig.savefig(f"{OUT}/fig2_ventilation.png"); plt.close(fig)
json.dump(dict(slope=b, r=r, n=len(d)), open("results/fig2_stats.json", "w"))

p = pd.read_csv("results/test_predictions.csv", index_col=0, parse_dates=["target_time"])
p = p.set_index("target_time")
e = p.loc["2024-11-08":"2024-11-28"]
fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.plot(e.index, np.exp(e.obs), color=INK, lw=2, label="Observed")
ax.plot(e.index, np.exp(e.M0_Persistence), color=GREY, lw=1.2, label="Persistence (yesterday's value)")
ax.plot(e.index, np.exp(e["M2_AI_only"]), color=BLUE, lw=1.6, label="AI only")
ax.plot(e.index, np.exp(e["M4_Physics+AI_residual"]), color=ORANGE, lw=1.6, label="Physics + AI")
ax.set_ylabel("PM2.5 (µg/m³)")
ax.set_title("Fig. 4  24-hour-ahead forecasts during the November 2024 smog episode")
ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2))
fig.autofmt_xdate()
fig.savefig(f"{OUT}/fig4_episode.png"); plt.close(fig)

s = pd.read_csv("results/scores.csv")
lab = {"M0_Persistence": "Persistence", "M1_Physics_only": "Physics only",
       "M2_AI_only": "AI only", "M3_Physics+AI": "Physics + AI (features)",
       "M4_Physics+AI_residual": "Physics + AI (residual)", "C_Placebo": "Placebo features (control)",
       "M5_AI_residual_on_persistence": "AI residual on persistence (control)"}
w = s[s.subset == "Test, winter (Oct-Feb)"].set_index("model").RMSE.rename(index=lab).sort_values()
fig, ax = plt.subplots(figsize=(6.5, 3.4))
cols = [ORANGE if "Physics" in i and "only" not in i else (BLUE if i == "AI only" else GREY) for i in w.index]
ax.barh(w.index, w.values, color=cols, height=0.6)
for i, v in enumerate(w.values):
    ax.text(v + 0.6, i, f"{v:.1f}", va="center", fontsize=9, color=INK)
ax.invert_yaxis(); ax.set_xlim(0, max(w) * 1.12); ax.grid(axis="y", visible=False)
ax.set_xlabel("Error, RMSE (µg/m³) — lower is better")
ax.set_title("Fig. 3  Unseen winters (Oct 2024–Dec 2025): physics features did not help")
fig.savefig(f"{OUT}/fig3_scores.png"); plt.close(fig)

ex = json.load(open("results/experiments.json"))
E1 = ex["E1_data_efficiency"]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw=dict(width_ratios=[1.3, 1], wspace=0.45))
ns = [E1[k]["n_train"] / 24 / 365.25 for k in E1]
for key, colr, name in [("AI only", BLUE, "AI only"), ("Physics + AI (residual)", ORANGE, "Physics + AI (residual)"),
                        ("AI residual on persistence (control)", GREY, "AI residual on persistence (control)")]:
    ax[0].plot(ns, [E1[k][key][0] for k in E1], "o-", color=colr, lw=2, ms=6, label=name)
ax[0].axhline(E1[list(E1)[0]]["Persistence"][0], color=INK2, ls="--", lw=1)
ax[0].text(ns[-1], E1[list(E1)[0]]["Persistence"][0] + 0.4, "persistence", ha="right", va="bottom", fontsize=8, color=INK2)
ax[0].set_xlabel("Years of training data"); ax[0].set_ylabel("Winter RMSE (µg/m³)")
ax[0].set_title("(a) Less training data")
ax[0].legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1, 0.9))
E2 = ex["E2_unseen_season"]
order = ["AI only", "Physics + AI", "AI residual on persistence (control)", "Physics + AI (residual)", "Physics only", "Persistence"]
short = ["AI only", "Physics + AI\n(features)", "AI residual on\npersistence", "Physics + AI\n(residual)", "Physics only", "Persistence"]
vals = [E2[k][0] for k in order]
cc = [BLUE, ORANGE, GREY, ORANGE, GREY, GREY]
ax[1].barh(short, vals, color=cc, height=0.6)
for i, v in enumerate(vals):
    ax[1].text(v + 1, i, f"{v:.0f}", va="center", fontsize=9)
ax[1].invert_yaxis(); ax[1].grid(axis="y", visible=False); ax[1].set_xlim(0, max(vals) * 1.15)
ax[1].set_xlabel("Winter RMSE (µg/m³)")
ax[1].set_title("(b) Trained without any winter data")
fig.suptitle("Fig. 5  Stress tests", x=0.06, ha="left", fontweight="bold")
fig.savefig(f"{OUT}/fig5_stress.png"); plt.close(fig)

sh = pd.read_csv("results/shap_importance_M3.csv", index_col=0).iloc[:12, 0][::-1]
nice = {"lpm_0": "PM2.5 now", "lpm_mean24": "PM2.5, last 24 h mean", "hour_fut": "Hour of target",
        "lpm_lag23": "PM2.5, 23 h ago", "doy_cos": "Season (cos)", "doy_sin": "Season (sin)",
        "phys_pred": "Physics forecast*", "lvc_mean24_fut": "Ventilation, target day*",
        "lh_fut": "Mixing height at target*", "lvc_fut": "Ventilation at target*", "dlvc": "Ventilation change*",
        "lh_now": "Mixing height now*", "lvc_now": "Ventilation now*"}
base = {"rh": "Humidity", "bp": "Air pressure", "ws": "Wind speed", "temp": "Temperature", "sr": "Sunlight",
        "wu": "Wind (east-west)", "wv": "Wind (north-south)", "rain": "Rain", "dtemp": "Temperature change"}
def nm(i):
    if i in nice: return nice[i]
    v, _, when = i.rpartition("_")
    return f"{base.get(v, v)} ({'target' if when == 'fut' else 'now'})"
names = [nm(i) for i in sh.index]
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.barh(names, sh.values, color=[ORANGE if n.endswith("*") else BLUE for n in names], height=0.6)
ax.grid(axis="y", visible=False)
ax.set_xlabel("Mean |SHAP| contribution to ln PM2.5")
ax.set_title("Fig. 6  What the Physics + AI model uses (orange = physics)")
fig.savefig(f"{OUT}/fig6_shap.png"); plt.close(fig)
print("figures written")
