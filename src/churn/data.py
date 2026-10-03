"""Download Online Retail II, clean it, write parquet."""
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
DATA_DIR = Path("data")
RAW = DATA_DIR / "online_retail_II.xlsx"
CLEAN = DATA_DIR / "transactions.parquet"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download() -> None:
    if RAW.exists():
        return
    DATA_DIR.mkdir(exist_ok=True)
    blob = urllib.request.urlopen(URL).read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = next(n for n in z.namelist() if n.endswith(".xlsx"))
        RAW.write_bytes(z.read(name))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"Customer ID": "customer_id", "InvoiceDate": "ts", "Invoice": "invoice",
                            "StockCode": "stock_code", "Quantity": "qty", "Price": "price",
                            "Country": "country"})
    df = df.dropna(subset=["customer_id"])
    df = df[~df["invoice"].astype(str).str.startswith("C")]  # cancellations
    df = df[(df["qty"] > 0) & (df["price"] > 0)]
    df = df.assign(customer_id=df["customer_id"].astype(int),
                   stock_code=df["stock_code"].astype(str),
                   invoice=df["invoice"].astype(str),
                   amount=df["qty"] * df["price"])
    cols = ["customer_id", "invoice", "ts", "stock_code", "qty", "price", "amount", "country"]
    return df[cols].drop_duplicates().sort_values("ts").reset_index(drop=True)


def main() -> None:
    download()
    sheets = pd.read_excel(RAW, sheet_name=None)  # two sheets: 2009-10, 2010-11
    df = clean(pd.concat(sheets.values(), ignore_index=True))
    df.to_parquet(CLEAN, index=False)
    print(f"{len(df):,} rows, {df.customer_id.nunique():,} customers, "
          f"{df.ts.min()} -> {df.ts.max()}, raw sha256={sha256(RAW)[:12]}")


if __name__ == "__main__":
    main()
