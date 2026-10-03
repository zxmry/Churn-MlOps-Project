"""Replay the held-out 'prod' months (Aug-Dec 2011) against the running API as live traffic."""
import json
import sys
import urllib.request

import pandas as pd

from churn.data import CLEAN
from churn.features import FEATURES, SPLITS, build_features

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"


def post(path, payload):
    req = urllib.request.Request(URL + path, json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))


tx = pd.read_parquet(CLEAN)
for t in SPLITS["prod"]:
    f = build_features(tx, t).reset_index()
    rows = f[["customer_id", *FEATURES]].to_dict("records")
    for i in range(0, len(rows), 1000):
        out = post("/predict", {"as_of": t.date().isoformat(), "instances": rows[i:i + 1000]})
    print(f"{t.date()}: scored {len(rows)} customers (model v{out['model_version']})")
