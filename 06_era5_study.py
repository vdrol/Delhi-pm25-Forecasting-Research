from common import *

e = pd.read_csv("data/era5_delhi_openmeteo.csv", skiprows=3,
                parse_dates=["time"], index_col="time")
e.columns = ["blh", "ws100", "wd100", "cloud", "t2m"]
e["u100"] = -e.ws100 * np.sin(np.deg2rad(e.wd100))
e["v100"] = -e.ws100 * np.cos(np.deg2rad(e.wd100))
e["vce"] = e.blh * e.ws100.clip(lower=0.5)
e = e.reindex(c.index)

ERA_RAW = []
for v in ["blh", "ws100", "u100", "v100", "cloud"]:
    for name, sh in (("now", 0), ("fut", -H)):
        data[f"e_{v}_{name}"] = e[v].shift(sh).reindex(data.index)
        ERA_RAW.append(f"e_{v}_{name}")
data["e_lblh_fut"] = np.log(data["e_blh_fut"])
data["e_lvc_now"] = np.log(e.vce).reindex(data.index)
data["e_lvc_fut"] = np.log(e.vce).shift(-H).reindex(data.index)
data["e_lvc_mean24_fut"] = np.log(e.vce.rolling(24, min_periods=18).mean()).shift(-H).reindex(data.index)
data["e_dlvc"] = data["e_lvc_now"] - data["e_lvc_fut"]
ERA_PHYS = ["e_lblh_fut", "e_lvc_now", "e_lvc_fut", "e_lvc_mean24_fut", "e_dlvc"]

ok = data[ERA_RAW + ["e_lvc_now", "e_lvc_fut"]].notna().all(axis=1)
TR = data[data.index.isin(tr.index) & ok].copy()
VA = data[data.index.isin(va.index) & ok].copy()
TE = data[data.index.isin(te.index)].copy()
print("rows: train", len(TR), "val", len(VA), "test", len(TE), "test missing ERA5:", (~ok[TE.index]).sum())

def fit_box(d, col):
    A = np.c_[d[col], np.ones(len(d))]
    return np.linalg.lstsq(A, d["y"] - d["lpm_0"], rcond=None)[0]
be, ce = fit_box(TR, "e_dlvc")
for d in (TR, VA, TE):
    d["phys_pred"] = d["lpm_0"] + beta * d["dlvc"] + c0
    d["e_phys_pred"] = d["lpm_0"] + be * d["e_dlvc"] + ce
print(f"ERA5 box model beta = {be:.3f} (ground-sensor beta = {beta:.3f})")

SPECS = {
    "M2 AI only (ground sensors)": (RAW_FEATS, None),
    "M3 AI + ground physics": (RAW_FEATS + PHYS_FEATS + ["phys_pred"], None),
    "N1 AI + ERA5 data": (RAW_FEATS + ERA_RAW, None),
    "N2 AI + ERA5 physics": (RAW_FEATS + ERA_RAW + ERA_PHYS + ["e_phys_pred"], None),
    "N4 AI corrects ERA5 box model": (RAW_FEATS + ERA_RAW + ERA_PHYS, "e_phys_pred"),
}

def run(trd, vad, ted, specs, seeds):
    preds = {}
    for n, (f, init) in specs.items():
        preds[n] = np.mean([predict(train_lgb(trd, vad, f, s, init), ted, f, init) for s in seeds], axis=0)
    return preds

P = pd.DataFrame(index=TE.index)
P["obs"] = TE["y"]; P["target_time"] = TE["target_time"]
P["M0 Persistence"] = TE["lpm_0"]
sm_e = np.log(np.mean(np.exp(VA["y"] - VA["e_phys_pred"])))
P["N3 ERA5 box model only"] = TE["e_phys_pred"] + sm_e
for n, p in run(TR, VA, TE, SPECS, [0, 1, 2, 3, 4]).items():
    P[n] = p
P.to_csv(f"{OUT}/v2_test_predictions.csv")
W = P[P.target_time.dt.month.isin([10, 11, 12, 1, 2])]
SEV = P[np.exp(P.obs) > 250]
names = [n for n in P.columns if n not in ("obs", "target_time")]
rows = []
for sub, d in [("winter", W), ("all", P), ("severe", SEV)]:
    for n in names:
        r = metrics(d.obs.values, d[n].values); r.update(model=n, subset=sub, n=len(d)); rows.append(r)
S = pd.DataFrame(rows); S.to_csv(f"{OUT}/v2_scores.csv", index=False)
print(S.pivot(index="model", columns="subset", values="RMSE").round(1).to_string())

Wd = W.assign(day=W.target_time.dt.normalize()); days = Wd.day.unique()
G = {d: g for d, g in Wd.groupby("day")}
def rm(d, n): return np.sqrt(np.mean((np.exp(d[n]) - np.exp(d.obs)) ** 2))
pairs = {"Q1 new data: N1 - M2": ("N1 AI + ERA5 data", "M2 AI only (ground sensors)"),
         "Q2 physics on new data: N2 - N1": ("N2 AI + ERA5 physics", "N1 AI + ERA5 data"),
         "N2 - M2": ("N2 AI + ERA5 physics", "M2 AI only (ground sensors)"),
         "Q3 anchor: N4 - M2": ("N4 AI corrects ERA5 box model", "M2 AI only (ground sensors)"),
         "old: M3 - M2": ("M3 AI + ground physics", "M2 AI only (ground sensors)")}
bs = {k: [] for k in pairs}
for _ in range(2000):
    smp = pd.concat([G[d] for d in rng.choice(days, len(days))])
    for k, (a, b) in pairs.items():
        bs[k].append(rm(smp, a) - rm(smp, b))
B = {k: dict(mean=float(np.mean(v)), lo95=float(np.percentile(v, 2.5)), hi95=float(np.percentile(v, 97.5)))
     for k, v in bs.items()}
print(json.dumps(B, indent=1))

ST = {"M2 AI only (ground sensors)": SPECS["M2 AI only (ground sensors)"],
      "N1 AI + ERA5 data": SPECS["N1 AI + ERA5 data"],
      "N2 AI + ERA5 physics": SPECS["N2 AI + ERA5 physics"],
      "N4 AI corrects ERA5 box model": SPECS["N4 AI corrects ERA5 box model"]}
TEw = TE[TE.target_time.dt.month.isin([10, 11, 12, 1, 2])]
stress = {}
for label, trd, vad in [
        ("one winter of training", TR[(TR.target_time >= "2022-10-01") & (TR.target_time <= "2023-02-28")], VA),
        ("no winter in training", TR[TR.target_time.dt.month.isin(range(3, 10))],
         VA[VA.target_time.dt.month.isin(range(3, 10))])]:
    trd, vad, ted = trd.copy(), vad.copy(), TEw.copy()
    b1, c1 = fit_box(trd, "e_dlvc")
    for d in (trd, vad, ted):
        d["e_phys_pred"] = d["lpm_0"] + b1 * d["e_dlvc"] + c1
    pr = run(trd, vad, ted, ST, [0, 1, 2])
    stress[label] = {n: metrics(ted.y.values, p)["RMSE"] for n, p in pr.items()}
    stress[label]["M0 Persistence"] = metrics(ted.y.values, ted.lpm_0.values)["RMSE"]
    print(label, {k: round(v, 1) for k, v in stress[label].items()})

dd = pd.DataFrame({"pm": c.pm25, "vc_ground": c.vc, "vc_era5": e.vce, "blh": e.blh, "h_ground": c.h_mix}).resample("D").mean()
dd = dd[dd.index.month.isin([10, 11, 12, 1, 2])].dropna()
corr = {k: float(np.corrcoef(np.log(dd.pm), np.log(dd[k]))[0, 1]) for k in ["vc_ground", "vc_era5", "h_ground", "blh"]}
hours = c.join(e[["blh"]]).dropna(subset=["blh", "h_mix"])
agree = float(np.corrcoef(np.log(hours.h_mix), np.log(hours.blh))[0, 1])
print("daily winter corr with ln PM:", {k: round(v, 3) for k, v in corr.items()}, " ground h vs ERA5 BLH (hourly, log) r =", round(agree, 3))

json.dump(dict(era5_beta=be, ground_beta=beta, bootstrap=B, stress=stress, daily_corr=corr,
               h_vs_blh_r=agree, n_train=len(TR), n_val=len(VA), n_test=len(TE), n_test_winter=len(W)),
          open(f"{OUT}/v2_summary.json", "w"), indent=1)
