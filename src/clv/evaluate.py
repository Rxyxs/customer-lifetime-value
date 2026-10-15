"""Predict each customer's next 26 weeks and compare with what they really did.

Calibration ends 2011-06-10; the 26 weeks after it (including the Christmas
peak) are the holdout. Three predictors of holdout purchases and revenue:

- naive: the customer's past purchase rate, extended over 26 weeks;
- BG/NBD x Gamma-Gamma, fitted on the calibration period only;
- LightGBM on recency / frequency / monetary features over fixed lookback
  windows. It needs labelled examples, so it is trained one year earlier (features up to 2010-06-10,
  target = the 26 weeks after), the same season as the holdout.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .btyd import BGNBD, GammaGamma
from .data import daily_purchases, summarise

CUTOFF = pd.Timestamp("2011-06-10")
TRAIN_CUTOFF = pd.Timestamp("2010-06-10")
H = 26.0
FEATS = ["n_13w", "n_26w", "v_13w", "v_26w", "mean_value", "recency", "tenure_capped", "uk"]


def rfm(s: pd.DataFrame) -> pd.DataFrame:
    return s.assign(recency=s["T"] - s["t_x"], tenure_capped=s["T"].clip(upper=26))


def windowed(p: pd.DataFrame, s: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Lookback features over fixed windows, so they mean the same thing at any cutoff.

    Lifetime totals grow with tenure: a model trained when the data was six months
    old and applied when it is eighteen months old would see feature values it
    never trained on and extrapolate.
    """
    out = s.copy()
    for w in (13, 26):
        win = p[(p.day < cutoff) & (p.day >= cutoff - pd.Timedelta(weeks=w))].groupby("customer")
        out[f"n_{w}w"] = win.size().reindex(s.index).fillna(0)
        out[f"v_{w}w"] = win.value.sum().reindex(s.index).fillna(0.0)
    return out


def capture(actual: np.ndarray, pred: np.ndarray, frac: float) -> float:
    k = int(len(actual) * frac)
    return float(actual[np.argsort(-pred)[:k]].sum() / actual.sum())


def scores(actual, pred) -> dict:
    return {"pred_total": float(pred.sum()), "actual_total": float(actual.sum()),
            "total_error_pct": float(100 * (pred.sum() / actual.sum() - 1)),
            "mae": float(np.mean(np.abs(pred - actual))), "spearman": float(spearmanr(pred, actual)[0]),
            "top10_capture": capture(actual, pred, 0.10), "top20_capture": capture(actual, pred, 0.20)}


def main() -> None:
    p = daily_purchases()
    s = windowed(p, rfm(summarise(p, CUTOFF, H)), CUTOFF)
    s = s[s["T"] > 0]
    res = {"customers": int(len(s)), "repeaters_share": float((s.x > 0).mean()),
           "holdout_active_share": float((s.holdout_purchases > 0).mean())}

    bg = BGNBD().fit(s.x, s.t_x, s["T"])
    rep = s[s.x > 0]
    gg = GammaGamma().fit(rep.x, rep.monetary)
    res["bgnbd"] = {k: float(getattr(bg, k)) for k in ("r", "alpha", "a", "b")}
    res["gamma_gamma"] = {k: float(getattr(gg, k)) for k in ("p", "q", "v")}
    res["spend_frequency_corr"] = float(np.corrcoef(rep.x, rep.monetary)[0, 1])

    n_bg = bg.expected_purchases(H, s.x, s.t_x, s["T"])
    spend = gg.expected_value(s.x, s.monetary.fillna(0))
    v_bg = n_bg * spend
    n_naive = (s.x + 1) / s["T"].clip(lower=1) * H
    v_naive = n_naive * s.mean_value

    tr = windowed(p, rfm(summarise(p, TRAIN_CUTOFF, H)), TRAIN_CUTOFF)
    tr = tr[tr["T"] > 0]
    gbm_n = lgb.LGBMRegressor(objective="poisson", n_estimators=300, learning_rate=0.03, num_leaves=15,
                              min_child_samples=30, random_state=0, verbose=-1).fit(tr[FEATS], tr.holdout_purchases)
    gbm_v = lgb.LGBMRegressor(objective="tweedie", n_estimators=300, learning_rate=0.03, num_leaves=15,
                              min_child_samples=30, random_state=0, verbose=-1).fit(tr[FEATS], tr.holdout_value)
    n_gbm, v_gbm = gbm_n.predict(s[FEATS]), gbm_v.predict(s[FEATS])

    yn, yv = s.holdout_purchases.to_numpy(), s.holdout_value.to_numpy()
    res["purchases"] = {"naive": scores(yn, n_naive.to_numpy()), "bgnbd": scores(yn, np.asarray(n_bg)),
                        "lightgbm": scores(yn, n_gbm)}
    res["revenue"] = {"naive": scores(yv, v_naive.to_numpy()), "bgnbd_gg": scores(yv, np.asarray(v_bg)),
                      "lightgbm": scores(yv, v_gbm)}

    pa = bg.p_alive(s.x, s.t_x, s["T"])
    bands = pd.cut(pa, [0, 0.25, 0.5, 0.75, 0.95, 1.0], include_lowest=True)
    res["p_alive_check"] = (pd.DataFrame({"band": bands.astype(str), "active": (yn > 0)})
                            .groupby("band").active.agg(["mean", "size"]).round(3).to_dict(orient="index"))
    pd.DataFrame({"customer": s.index, "x": s.x, "T": s["T"], "p_alive": pa, "n_bgnbd": n_bg, "v_bgnbd": v_bg,
                  "n_gbm": n_gbm, "v_gbm": v_gbm, "n_naive": n_naive, "actual_n": yn, "actual_v": yv}) \
        .to_parquet("reports/predictions.parquet", index=False)

    # Cohort retention for the figure: share of each monthly cohort buying in later months
    first = p.groupby("customer").day.min().dt.to_period("M")
    q = p.assign(cohort=p.customer.map(first), month=p.day.dt.to_period("M"))
    q["age"] = (q.month - q.cohort).apply(lambda o: o.n)
    ret = q.groupby(["cohort", "age"]).customer.nunique().unstack(fill_value=0)
    (ret.div(ret[0], axis=0)).to_csv("reports/cohort_retention.csv")

    Path("reports/results.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: res[k] for k in ("customers", "repeaters_share", "holdout_active_share", "bgnbd",
                                          "gamma_gamma", "spend_frequency_corr")}, indent=1))
    for kind in ("purchases", "revenue"):
        for name, sc in res[kind].items():
            print(kind, name, {k: round(v, 3) for k, v in sc.items()})
    print(res["p_alive_check"])


if __name__ == "__main__":
    Path("reports").mkdir(exist_ok=True)
    main()
