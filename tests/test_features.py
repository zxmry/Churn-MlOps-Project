import pandas as pd

from churn.features import FEATURES, build_features, build_labels

T = pd.Timestamp("2011-01-01")


def tx(rows):
    df = pd.DataFrame(rows, columns=["customer_id", "invoice", "ts", "stock_code", "amount"])
    return df.assign(ts=pd.to_datetime(df["ts"]), country="United Kingdom")


BASE = tx([
    (1, "a", "2010-06-01", "X", 10.0),
    (1, "b", "2010-12-15", "Y", 20.0),
    (2, "c", "2010-11-01", "X", 5.0),
    (3, "d", "2010-06-01", "X", 5.0),  # nothing in the 180d lookback: excluded
])


def test_no_leakage_from_future_transactions():
    future = tx([(1, "z", "2011-01-01", "Z", 999.0), (4, "y", "2011-02-01", "Z", 1.0)])
    before = build_features(BASE, T)
    after = build_features(pd.concat([BASE, future]), T)
    pd.testing.assert_frame_equal(before, after)


def test_feature_values_and_population():
    f = build_features(BASE, T)
    assert list(f.columns) == FEATURES
    assert sorted(f.index) == [1, 2]
    assert f.loc[1, "recency_days"] == 17
    assert f.loc[1, "spend_30d"] == 20.0
    assert f.loc[1, "n_invoices_180d"] == 1  # 2010-06-01 purchase is outside the lookback
    assert f.loc[1, "tenure_days"] == 180  # true tenure 214, clipped to lookback


def test_labels_use_90_day_horizon():
    future = tx([(1, "z", "2011-03-01", "Z", 1.0), (2, "y", "2011-04-15", "Z", 1.0)])
    y = build_labels(pd.concat([BASE, future]), T, pd.Index([1, 2]))
    assert y.to_dict() == {1: 0, 2: 1}  # customer 2 returns after day 90 -> churned
