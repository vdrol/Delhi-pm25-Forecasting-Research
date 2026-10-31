import json
import numpy as np
import pandas as pd
from common import *

trp, vap, tep = add_placebo(tr), add_placebo(va), add_placebo(te)
preds = pd.DataFrame(index=te.index)
preds["obs"] = te["y"]; preds["target_time"] = te["target_time"]
preds["M0_Persistence"] = te["lpm_0"]
preds["M1_Physics_only"] = te["phys_pred"] + PHYS_SMEAR
seed_scores = {}
importance = None
for name, (feats, init) in MODELS.items():
    ps = []
    for s in SEEDS:
        m = train_lgb(trp, vap, feats, s, init)
        p = predict(m, tep, feats, init)
        ps.append(p)
        seed_scores.setdefault(name, []).append(metrics(winter(te)["y"].values,
                                                       pd.Series(p, te.index).loc[winter(te).index].values)["RMSE"])
        if name == "M3_Physics+AI" and s == 0:
            importance = pd.Series(m.feature_importance("gain"), feats).sort_values(ascending=False)
            contrib = m.predict(winter(tep)[feats], num_iteration=m.best_iteration, pred_contrib=True)
            shap = pd.Series(np.abs(contrib[:, :-1]).mean(0), feats).sort_values(ascending=False)
    preds[name] = np.mean(ps, axis=0)
    print(name, "winter RMSE by seed:", np.round(seed_scores[name], 1))
preds.to_csv(f"{OUT}/test_predictions.csv")
shap.to_csv(f"{OUT}/shap_importance_M3.csv", header=["mean_abs_shap"])

names = ["M0_Persistence", "M1_Physics_only"] + list(MODELS)
rows = []
for subset, d in [("Test, winter (Oct-Feb)", preds[preds.target_time.dt.month.isin([10, 11, 12, 1, 2])]),
                  ("Test, all seasons", preds),
                  ("Test, severe hours (PM2.5>250)", preds[np.exp(preds.obs) > 250])]:
    for n in names:
        r = metrics(d["obs"].values, d[n].values); r.update(model=n, subset=subset, n=len(d))
        rows.append(r)
scores = pd.DataFrame(rows)
scores.to_csv(f"{OUT}/scores.csv", index=False)
print(scores.pivot(index="model", columns="subset", values="RMSE").round(1).to_string())

w = preds[preds.target_time.dt.month.isin([10, 11, 12, 1, 2])].copy()
w["day"] = w.target_time.dt.normalize()
days = w["day"].unique()
groups = {d: g for d, g in w.groupby("day")}
def rmse(d, n): return np.sqrt(np.mean((np.exp(d[n]) - np.exp(d["obs"])) ** 2))
boot = {k: [] for k in ["M3_vs_M2", "M4_vs_M2", "C_vs_M2", "M2_vs_M0", "M1_vs_M0", "M4_vs_M5"]}
for _ in range(2000):
    smp = pd.concat([groups[d] for d in rng.choice(days, len(days), replace=True)])
    boot["M3_vs_M2"].append(rmse(smp, "M3_Physics+AI") - rmse(smp, "M2_AI_only"))
    boot["M4_vs_M2"].append(rmse(smp, "M4_Physics+AI_residual") - rmse(smp, "M2_AI_only"))
    boot["C_vs_M2"].append(rmse(smp, "C_Placebo") - rmse(smp, "M2_AI_only"))
    boot["M2_vs_M0"].append(rmse(smp, "M2_AI_only") - rmse(smp, "M0_Persistence"))
    boot["M1_vs_M0"].append(rmse(smp, "M1_Physics_only") - rmse(smp, "M0_Persistence"))
    boot["M4_vs_M5"].append(rmse(smp, "M4_Physics+AI_residual") - rmse(smp, "M5_AI_residual_on_persistence"))
bsum = {k: dict(mean=float(np.mean(v)), lo95=float(np.percentile(v, 2.5)),
                hi95=float(np.percentile(v, 97.5)), p_better=float(np.mean(np.array(v) < 0)))
        for k, v in boot.items()}
print(json.dumps(bsum, indent=1))

json.dump(dict(beta=beta, c=c0, n_train=len(tr), n_val=len(va), n_test=len(te),
               n_test_winter=len(winter(te)), seed_rmse=seed_scores, bootstrap=bsum,
               top_importance_gain=importance.head(12).round(0).to_dict()),
          open(f"{OUT}/summary.json", "w"), indent=1)
