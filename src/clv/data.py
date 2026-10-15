"""Clean Online Retail II and summarise customers at a cutoff date.

Cleaning keeps identified customers, positive quantities and prices, and real
products (fees, postage and manual adjustments are dropped). Cancellations are
removed. Several invoices on the same day count as one purchase, the unit the
BG/NBD model assumes.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
DATASET = "mashlyn/online-retail-ii-uci"
NON_PRODUCT = {"POST", "DOT", "M", "C2", "D", "S", "BANK CHARGES", "ADJUST", "AMAZONFEE", "CRUK", "PADS", "B"}
WEEK = pd.Timedelta(days=7)


def download(raw: Path = RAW) -> None:
    if (raw / "online_retail_II.csv").exists():
        return
    raw.mkdir(parents=True, exist_ok=True)
    kaggle = Path(sys.executable).with_name("kaggle.exe" if sys.platform == "win32" else "kaggle")
    subprocess.run([str(kaggle), "datasets", "download", "-d", DATASET, "-p", str(raw), "--unzip"], check=True)


def daily_purchases() -> pd.DataFrame:
    download()
    d = pd.read_csv(RAW / "online_retail_II.csv", parse_dates=["InvoiceDate"],
                    dtype={"Invoice": str, "StockCode": str})
    d = d.dropna(subset=["Customer ID"])
    d = d[~d.Invoice.str.startswith("C") & (d.Quantity > 0) & (d.Price > 0)]
    d = d[~d.StockCode.str.upper().isin(NON_PRODUCT)]
    d["value"] = d.Quantity * d.Price
    d["day"] = d.InvoiceDate.dt.normalize()
    out = d.groupby(["Customer ID", "day"], as_index=False).agg(value=("value", "sum"), country=("Country", "first"))
    return out.rename(columns={"Customer ID": "customer"})


def summarise(p: pd.DataFrame, cutoff: pd.Timestamp, horizon_weeks: float) -> pd.DataFrame:
    """Per-customer calibration summary up to cutoff, plus what happened in the next horizon."""
    cal = p[p.day < cutoff]
    hold = p[(p.day >= cutoff) & (p.day < cutoff + horizon_weeks * WEEK)]
    g = cal.groupby("customer")
    first, last = g.day.min(), g.day.max()
    n = g.size()
    s = pd.DataFrame({
        "x": n - 1,
        "t_x": (last - first) / WEEK,
        "T": (cutoff - first) / WEEK,
        "monetary": (g.value.sum() - cal.loc[cal.groupby("customer").day.idxmin(), ["customer", "value"]]
                     .set_index("customer").value) / (n - 1).replace(0, np.nan),
        "total_value": g.value.sum(),
        "mean_value": g.value.mean(),
        "uk": g.country.first().eq("United Kingdom").astype(int),
        "first_day": first,
    })
    hg = hold.groupby("customer")
    s["holdout_purchases"] = hg.size().reindex(s.index).fillna(0)
    s["holdout_value"] = hg.value.sum().reindex(s.index).fillna(0.0)
    return s
