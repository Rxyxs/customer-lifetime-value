"""README figures: conditional expectations, revenue capture, cohorts, P(alive)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path("docs/img")
COLORS = {"naive": "#BAB0AC", "bgnbd": "#4E79A7", "gbm": "#F28E2B"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet("reports/predictions.parquet")

    # 1. Predicted vs actual holdout purchases by calibration frequency
    band = d.x.clip(upper=12)
    g = d.groupby(band).agg(actual=("actual_n", "mean"), bgnbd=("n_bgnbd", "mean"),
                            gbm=("n_gbm", "mean"), naive=("n_naive", "mean"), n=("x", "size"))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(g.index, g.actual, "ko-", lw=2, label="actual")
    ax.plot(g.index, g.bgnbd, "s--", color=COLORS["bgnbd"], label="BG/NBD")
    ax.plot(g.index, g.gbm, "^--", color=COLORS["gbm"], label="LightGBM")
    ax.plot(g.index, g.naive, "d:", color=COLORS["naive"], label="naive rate")
    ax.set_xticks(g.index, [str(i) if i < 12 else "12+" for i in g.index])
    ax.set_xlabel("repeat purchases in the calibration period")
    ax.set_ylabel("purchases in the next 26 weeks (mean)")
    ax.set_title("Predicted vs actual purchases, Jun-Dec 2011")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "conditional.png", dpi=130)
    plt.close(fig)

    # 2. Revenue capture
    fig, ax = plt.subplots(figsize=(7.5, 4))
    frac = np.arange(1, len(d) + 1) / len(d)
    total = d.actual_v.sum()
    for col, lab, c in (("v_bgnbd", "BG/NBD x Gamma-Gamma", COLORS["bgnbd"]), ("v_gbm", "LightGBM", COLORS["gbm"])):
        order = np.argsort(-d[col].to_numpy())
        ax.plot(frac * 100, np.cumsum(d.actual_v.to_numpy()[order]) / total * 100, color=c, lw=2, label=lab)
    best = np.sort(d.actual_v.to_numpy())[::-1]
    ax.plot(frac * 100, np.cumsum(best) / total * 100, "k:", lw=1, label="perfect foresight")
    ax.set_xlim(0, 50)
    ax.set_xlabel("% of customers, ranked by predicted value")
    ax.set_ylabel("% of next-26-week revenue")
    ax.set_title("Who will be worth the most? Revenue captured by ranking")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "capture.png", dpi=130)
    plt.close(fig)

    # 3. Cohorts
    ret = pd.read_csv("reports/cohort_retention.csv", index_col=0)
    ret = ret.iloc[:, 1:13] * 100
    # Hide months that had not happened yet for a cohort when the data ends.
    last = pd.Period(ret.index[-1], "M")
    for cohort in ret.index:
        seen = (last - pd.Period(cohort, "M")).n
        ret.loc[cohort, [c for c in ret.columns if int(c) > seen]] = np.nan
    ret = ret.iloc[:-1]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    im = ax.imshow(np.ma.masked_invalid(ret.to_numpy()), cmap="Blues", aspect="auto", vmin=0, vmax=60)
    ax.set_yticks(range(len(ret)), ret.index)
    ax.set_xticks(range(ret.shape[1]), [str(c) for c in ret.columns])
    ax.set_xlabel("months since first purchase")
    ax.set_ylabel("cohort (month of first purchase)")
    ax.set_title("Share of each cohort buying again, by month")
    fig.colorbar(im, ax=ax, label="% of cohort")
    fig.tight_layout()
    fig.savefig(OUT / "cohorts.png", dpi=130)
    plt.close(fig)

    # 4. P(alive) check
    res = json.loads(Path("reports/results.json").read_text())["p_alive_check"]
    labels = list(res)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(range(len(labels)), [res[k]["mean"] * 100 for k in labels], color=COLORS["bgnbd"])
    for i, k in enumerate(labels):
        ax.text(i, res[k]["mean"] * 100 + 1, f"n={res[k]['size']:,}", ha="center", fontsize=8)
    ax.set_xticks(range(len(labels)), ["0-25%", "25-50%", "50-75%", "75-95%", "95-100%"])
    ax.set_xlabel("BG/NBD probability the customer is still active")
    ax.set_ylabel("% who bought in the next 26 weeks")
    ax.set_title("P(alive) sorts customers, but not at its top band")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "p_alive.png", dpi=130)
    plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
