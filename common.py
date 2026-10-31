import os, json
import numpy as np
import pandas as pd
import lightgbm as lgb
from physics import mixing_height

rng = np.random.default_rng(0)
OUT = "results"
import os; os.makedirs(OUT, exist_ok=True)

c = pd.read_csv("data/delhi_hourly.csv", index_col=0, parse_dates=True)
c = c.join(mixing_height(c))
c["rain"] = c["rain"].fillna(0)

H = 24
MET = ["temp", "rh", "ws", "wu", "wv", "sr", "bp", "rain"]
X = pd.DataFrame(index=c.index)
lp = np.log(c["pm25"])
X["lpm_0"] = lp
for k in [1, 3, 6, 12, 23]:
    X[f"lpm_lag{k}"] = lp.shift(k)
X["lpm_mean24"] = np.log(c["pm25"].rolling(24, min_periods=18).mean())
X["lpm_trend6"] = lp - lp.shift(6)
for v in MET:
    X[f"{v}_now"] = c[v]
    X[f"{v}_fut"] = c[v].shift(-H)
X["dtemp_fut"] = X["temp_fut"] - X["temp_now"]
tgt_time = c.index + pd.Timedelta(hours=H)
X["hour_fut"] = tgt_time.hour
X["doy_sin"] = np.sin(2 * np.pi * tgt_time.dayofyear / 365.25)
X["doy_cos"] = np.cos(2 * np.pi * tgt_time.dayofyear / 365.25)
X["dow_fut"] = tgt_time.dayofweek
RAW_FEATS = list(X.columns)

X["lh_now"] = np.log(c["h_mix"])
X["lh_fut"] = np.log(c["h_mix"]).shift(-H)
X["lvc_now"] = np.log(c["vc"])
X["lvc_fut"] = np.log(c["vc"]).shift(-H)
X["lvc_mean24_fut"] = np.log(c["vc"].rolling(24, min_periods=18).mean()).shift(-H)
X["dlvc"] = X["lvc_now"] - X["lvc_fut"]
PHYS_FEATS = ["lh_now", "lh_fut", "lvc_now", "lvc_fut", "lvc_mean24_fut", "dlvc"]

y = lp.shift(-H)
data = X.assign(y=y).dropna(subset=["lpm_0", "y", "lvc_now", "lvc_fut"])
t_target = data.index + pd.Timedelta(hours=H)
data["target_time"] = t_target

TRAIN_END = pd.Timestamp("2023-09-30 23:00")
VAL_END = pd.Timestamp("2024-09-30 23:00")
tr = data[data.target_time <= TRAIN_END - pd.Timedelta(hours=H)]
va = data[(data.index > TRAIN_END) & (data.target_time <= VAL_END - pd.Timedelta(hours=H))]
te = data[data.index > VAL_END]
winter = lambda d: d[d.target_time.dt.month.isin([10, 11, 12, 1, 2])]
print(f"train {len(tr)}  val {len(va)}  test {len(te)} (test winter {len(winter(te))})")

def fit_physics(d):
    A = np.c_[d["dlvc"], np.ones(len(d))]
    b = d["y"] - d["lpm_0"]
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    return coef
beta, c0 = fit_physics(tr)
for d in (tr, va, te, data):
    d["phys_pred"] = d["lpm_0"] + beta * d["dlvc"] + c0
data["phys_pred"] = data["lpm_0"] + beta * data["dlvc"] + c0
PHYS_SMEAR = float(np.log(np.mean(np.exp(va["y"] - va["phys_pred"]))))
print(f"physics beta = {beta:.3f}, c = {c0:.3f}")

pl_src = ["ws_now", "ws_fut", "sr_now", "sr_fut", "temp_fut", "rh_fut"]
def add_placebo(d, seed=0):
    r = np.random.default_rng(seed)
    out = d.copy()
    for i in range(len(PHYS_FEATS)):
        a, b = r.choice(pl_src, 2, replace=False)
        wa, wb = r.normal(size=2)
        out[f"plc{i}"] = np.log1p(np.abs(wa * out[a] + wb * out[b]))
    return out
PLC_FEATS = [f"plc{i}" for i in range(len(PHYS_FEATS))]

PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=31,
              min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, verbose=-1)

def train_lgb(trd, vad, feats, seed, init=None):
    p = dict(PARAMS, seed=seed)
    it_tr = trd[init].values if init else None
    it_va = vad[init].values if init else None
    dtr = lgb.Dataset(trd[feats], trd["y"], init_score=it_tr)
    dva = lgb.Dataset(vad[feats], vad["y"], init_score=it_va, reference=dtr)
    m = lgb.train(p, dtr, 3000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100, verbose=False)])

    pv = m.predict(vad[feats], num_iteration=m.best_iteration) + (it_va if init else 0)
    m.smear = float(np.log(np.mean(np.exp(vad["y"].values - pv))))
    return m

def predict(m, d, feats, init=None):
    p = m.predict(d[feats], num_iteration=m.best_iteration)
    return p + (d[init].values if init else 0) + m.smear

MODELS = {
    "M2_AI_only": (RAW_FEATS, None),
    "M3_Physics+AI": (RAW_FEATS + PHYS_FEATS + ["phys_pred"], None),
    "M4_Physics+AI_residual": (RAW_FEATS + PHYS_FEATS, "phys_pred"),
    "C_Placebo": (RAW_FEATS + PLC_FEATS, None),
    "M5_AI_residual_on_persistence": (RAW_FEATS, "lpm_0"),
}
SEEDS = [0, 1, 2, 3, 4]

def metrics(obs_l, pred_l):
    o, p = np.exp(obs_l), np.exp(pred_l)
    e = p - o
    return dict(RMSE=float(np.sqrt(np.mean(e ** 2))), MAE=float(np.mean(np.abs(e))),
                R2=float(1 - np.sum(e ** 2) / np.sum((o - o.mean()) ** 2)),
                bias=float(np.mean(e)))
