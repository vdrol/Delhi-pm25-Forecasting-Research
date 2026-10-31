import numpy as np
import pandas as pd

KAPPA = 0.4
Z = 10.0
Z0 = 1.0
RHO = 1.2
CP = 1005.0
GAMMA = 0.005
A_ENT = 0.2
ALPHA = 0.3
U_MIN = 0.2
H_MIN = 30.0
SR_NIGHT = 6.0

def mixing_height(df: pd.DataFrame) -> pd.DataFrame:

    U = df["ws"].clip(lower=U_MIN).interpolate(limit=3)
    sr = (df["sr"].interpolate(limit=3) - SR_NIGHT).clip(lower=0)

    ustar = KAPPA * U / np.log(Z / Z0)
    h_sbl = (2400.0 * ustar ** 1.5).clip(lower=H_MIN)

    H = ALPHA * sr
    out = pd.DataFrame(index=df.index)
    h_cbl = np.full(len(df), np.nan)
    day = df.index.normalize()
    for d, idx in pd.Series(np.arange(len(df)), index=df.index).groupby(day):
        pos = idx.values
        h_day = H.iloc[pos].values
        h0 = h_sbl.iloc[pos].values
        sunny = np.nan_to_num(h_day) > 5.0
        if not sunny.any():
            continue
        start = np.argmax(sunny)
        end = len(pos) - np.argmax(sunny[::-1])
        cum = np.nancumsum(np.nan_to_num(h_day[start:end])) * 3600.0
        hc = np.sqrt(h0[start] ** 2 + 2 * (1 + 2 * A_ENT) * cum / (RHO * CP * GAMMA))
        h_cbl[pos[start:end]] = hc
    out["h_sbl"] = h_sbl
    out["h_cbl"] = h_cbl
    out["h_mix"] = np.fmax(h_sbl.values, np.nan_to_num(h_cbl, nan=0.0))
    out.loc[h_sbl.isna(), "h_mix"] = np.nan
    out["vc"] = out["h_mix"] * U
    return out
