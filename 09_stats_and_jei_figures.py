import json, os, shutil, numpy as np, pandas as pd

rng = np.random.default_rng(1)
P = pd.read_csv("results/v2_test_predictions.csv", index_col=0, parse_dates=["target_time"])
W = P[P.target_time.dt.month.isin([10, 11, 12, 1, 2])].copy(); W["day"] = W.target_time.dt.normalize()
G = [g for _, g in W.groupby("day")]; n = len(G)
rm = lambda d, m: np.sqrt(np.mean((np.exp(d[m]) - np.exp(d.obs)) ** 2))
pairs = {"ERA5 data vs AI only": ("N1 AI + ERA5 data", "M2 AI only (ground sensors)"),
         "ERA5 physics vs ERA5 data": ("N2 AI + ERA5 physics", "N1 AI + ERA5 data"),
         "AI correcting box model vs AI only": ("N4 AI corrects ERA5 box model", "M2 AI only (ground sensors)"),
         "ground physics vs AI only": ("M3 AI + ground physics", "M2 AI only (ground sensors)"),
         "ERA5 data vs persistence": ("N1 AI + ERA5 data", "M0 Persistence")}
bs = {k: [] for k in pairs}
for _ in range(5000):
    s = pd.concat([G[i] for i in rng.integers(0, n, n)])
    for k, (a, b) in pairs.items():
        bs[k].append(rm(s, a) - rm(s, b))
out = {}
for k, v in bs.items():
    v = np.array(v); p = 2 * min((v <= 0).mean(), (v >= 0).mean())
    out[k] = dict(diff=float(v.mean()), lo95=float(np.percentile(v, 2.5)), hi95=float(np.percentile(v, 97.5)),
                  p=float(max(p, 1 / 5000)))
    print(k, {kk: round(vv, 4) for kk, vv in out[k].items()})
json.dump(dict(n_days=n, comparisons=out), open("results/v2_bootstrap_pvalues.json", "w"), indent=1)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
_st, _sup = Axes.set_title, Figure.suptitle
Axes.set_title = lambda self, label, *a, **k: None if str(label).startswith("Fig.") else _st(self, label, *a, **k)
Figure.suptitle = lambda self, t, *a, **k: None if str(t).startswith("Fig.") else _sup(self, t, *a, **k)
plt.rcParams["savefig.dpi"] = 300
for script in ["05_figures.py", "08_figures_v2.py"]:
    exec(open(script).read(), {"__name__": "__main__"})
os.makedirs("figures_jei", exist_ok=True)
for src, dst in [("fig1_diurnal", "Figure1"), ("fig2_ventilation", "Figure2"), ("v2_fig3_data_check", "Figure3"),
                 ("v2_fig4_scores", "Figure4"), ("v2_fig5_ablation", "Figure5"), ("v2_fig6_episode", "Figure6")]:
    shutil.copy(f"figures/{src}.png", f"figures_jei/{dst}.png")
print("JEI figures written to figures_jei/")
