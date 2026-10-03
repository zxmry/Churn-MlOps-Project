"""Point-in-time features and churn labels.

Every feature for snapshot T is computed from transactions strictly before T.
Label = 1 if the customer makes no purchase in [T, T + HORIZON).
"""
import numpy as np
import pandas as pd

HORIZON = pd.Timedelta(days=90)
# Fixed lookback for population AND features, so every snapshot sees the same amount of history.
# (v1 used unbounded lifetime features; they drifted by construction as the data window grew.)
LOOKBACK = pd.Timedelta(days=180)

FEATURES = [
    "recency_days", "tenure_days", "n_invoices_180d", "n_invoices_90d", "spend_180d",
    "spend_90d", "spend_30d", "avg_basket", "n_products_180d", "spend_trend", "is_uk",
]

# Embargoed temporal split: every label window ends before the next split's first snapshot.
# Train starts 2010-06: the first month with a full LOOKBACK of history (data begins 2009-12).
SPLITS = {
    "train": pd.date_range("2010-06-01", "2010-10-01", freq="MS"),
    "val": pd.date_range("2011-01-01", "2011-02-01", freq="MS"),
    "test": pd.date_range("2011-06-01", "2011-07-01", freq="MS"),
    "prod": pd.date_range("2011-08-01", "2011-12-01", freq="MS"),  # replayed as live traffic
}


def build_features(tx: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    hist = tx[tx["ts"] <= t]
    first_seen = hist.groupby("customer_id")["ts"].min()
    past = hist[hist["ts"] >= t - LOOKBACK]  # population: bought within the lookback
    g = past.groupby("customer_id")
    f = pd.DataFrame({
        "recency_days": (t - g["ts"].max()).dt.days,
        "n_invoices_180d": g["invoice"].nunique(),
        "spend_180d": g["amount"].sum(),
        "n_products_180d": g["stock_code"].nunique(),
        "is_uk": (g["country"].agg(lambda s: s.mode().iat[0]) == "United Kingdom").astype(int),
    })
    f["tenure_days"] = (t - first_seen.reindex(f.index)).dt.days.clip(upper=LOOKBACK.days)
    recent = past[past["ts"] >= t - pd.Timedelta(days=90)]
    f["n_invoices_90d"] = recent.groupby("customer_id")["invoice"].nunique()
    f["spend_90d"] = recent.groupby("customer_id")["amount"].sum()
    f["spend_30d"] = past[past["ts"] >= t - pd.Timedelta(days=30)].groupby("customer_id")["amount"].sum()
    prev90 = past[past["ts"] < t - pd.Timedelta(days=90)].groupby("customer_id")["amount"].sum()
    f = f.fillna(0)
    f["avg_basket"] = f["spend_180d"] / f["n_invoices_180d"]
    f["spend_trend"] = np.log1p(f["spend_90d"]) - np.log1p(prev90.reindex(f.index).fillna(0))
    return f[FEATURES]


def build_labels(tx: pd.DataFrame, t: pd.Timestamp, customers: pd.Index) -> pd.Series:
    future = tx[(tx["ts"] >= t) & (tx["ts"] < t + HORIZON)]
    return pd.Series((~customers.isin(future["customer_id"])).astype(int), index=customers,
                     name="churn")


def build_dataset(tx: pd.DataFrame, snapshots, with_labels: bool = True) -> pd.DataFrame:
    frames = []
    for t in snapshots:
        f = build_features(tx, t)
        if with_labels:
            f["churn"] = build_labels(tx, t, f.index)
        frames.append(f.assign(snapshot=t).reset_index())
    return pd.concat(frames, ignore_index=True)
