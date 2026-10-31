from common import *

wte = winter(te)
SEEDS3 = [0, 1, 2]

def run(trd, vad, ted, specs):
    out = {}
    for name, (feats, init) in specs.items():
        r = []
        for s in SEEDS3:
            m = train_lgb(trd, vad, feats, s, init)
            r.append(metrics(ted["y"].values, predict(m, ted, feats, init))["RMSE"])
        out[name] = (float(np.mean(r)), float(np.std(r)))
    return out

SPECS = {"AI only": MODELS["M2_AI_only"],
         "Physics + AI": MODELS["M3_Physics+AI"],
         "Physics + AI (residual)": MODELS["M4_Physics+AI_residual"],
         "AI residual on persistence (control)": MODELS["M5_AI_residual_on_persistence"]}
res = {}

E1 = {}
windows = {"1 winter (Oct 2022-Feb 2023)": ("2022-10-01", "2023-02-28"),
           "1 year": ("2022-10-01", "2023-09-30"),
           "2 years": ("2021-10-01", "2023-09-30"),
           "3 years": ("2020-10-01", "2023-09-30"),
           "5.75 years (all)": ("2018-01-01", "2023-09-30")}
for label, (a, b) in windows.items():
    d = tr[(tr.target_time >= a) & (tr.target_time <= b)].copy()
    b1, c1 = fit_physics(d)
    dd = {}
    for k, x in (("tr", d), ("va", va.copy()), ("te", wte.copy())):
        x["phys_pred"] = x["lpm_0"] + b1 * x["dlvc"] + c1
        dd[k] = x
    r = run(dd["tr"], dd["va"], dd["te"], SPECS)
    r["Persistence"] = (metrics(wte["y"].values, wte["lpm_0"].values)["RMSE"], 0.0)
    r["n_train"] = len(d)
    E1[label] = r
    print("E1", label, {k: (round(v[0], 1) if isinstance(v, tuple) else v) for k, v in r.items()})
res["E1_data_efficiency"] = E1

d = tr[tr.target_time.dt.month.isin(range(3, 10))].copy()
v2 = va[va.target_time.dt.month.isin(range(3, 10))].copy()
b2, c2 = fit_physics(d)
for x in (d, v2):
    x["phys_pred"] = x["lpm_0"] + b2 * x["dlvc"] + c2
t2 = wte.copy(); t2["phys_pred"] = t2["lpm_0"] + b2 * t2["dlvc"] + c2
E2 = run(d, v2, t2, SPECS)
sm2 = np.log(np.mean(np.exp(v2["y"] - v2["phys_pred"])))
E2["Physics only"] = (metrics(t2["y"].values, t2["phys_pred"].values + sm2)["RMSE"], 0.0)
E2["Persistence"] = (metrics(wte["y"].values, wte["lpm_0"].values)["RMSE"], 0.0)
print("E2", {k: round(v[0], 1) for k, v in E2.items()})
res["E2_unseen_season"] = E2

MET_FUT = [f"{v}_fut" for v in MET] + ["hour_fut", "doy_sin", "doy_cos", "dow_fut"]
PH_FUT = ["lh_fut", "lvc_fut", "lvc_mean24_fut"]
SPECS3 = {"AI only": (MET_FUT, None), "Physics + AI": (MET_FUT + PH_FUT, None)}
E3 = run(tr, va, wte, SPECS3)

def phys_diag(d):
    A = np.c_[d["lvc_mean24_fut"], pd.get_dummies(d["hour_fut"]).values.astype(float)]
    return A
coef, *_ = np.linalg.lstsq(phys_diag(tr), tr["y"], rcond=None)
sm3 = np.log(np.mean(np.exp(va["y"] - phys_diag(va) @ coef)))
E3["Physics only (box model)"] = (metrics(wte["y"].values, phys_diag(wte) @ coef + sm3)["RMSE"], 0.0)
E3["box_model_slope_lnVC"] = float(coef[0])
print("E3", E3)
res["E3_no_live_sensor"] = E3

json.dump(res, open(f"{OUT}/experiments.json", "w"), indent=1)
