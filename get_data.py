"""Download both datasets into data/ and check them against the SHA-256 sums the pipeline expects.

IBM Telco Customer Churn: IBM/telco-customer-churn-on-icp4d on GitHub (Apache-2.0).
Cell2Cell: Kaggle dataset jpacse/datasets-for-churn-telecom, version 2 (licence listed as "Unknown",
so it is not redistributed here). Only cell2celltrain.csv (51,047 labelled customers) is used.
Iranian Churn: UCI Machine Learning Repository, https://doi.org/10.24432/C5JW3Z (CC BY 4.0).
"""
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
IBM_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
C2C_URL = "https://www.kaggle.com/api/v1/datasets/download/jpacse/datasets-for-churn-telecom"
IRAN_URL = "https://archive.ics.uci.edu/static/public/563/data.csv"   # UCI Iranian Churn, CC BY 4.0
SHA = {"Telco-Customer-Churn.csv": "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91",
       "cell2celltrain.csv": "3f6b8ff0b8e8f5975aa0de9f1f54967e6c8cb867a9bec6cbb64f4c8990f9a645",
       "iranian_churn.csv": "90d5fb6bd1630cd4de4b4d28fcf8b4cb92a8f6ab7484605b0799d47386f7dbe1"}


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})) as r:
        return r.read()


def save(name, raw):
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == SHA[name], f"{name}: checksum {digest} does not match the one used in the paper"
    (DATA / name).write_bytes(raw)
    print("ok", name)


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    save("Telco-Customer-Churn.csv", fetch(IBM_URL))
    save("cell2celltrain.csv", zipfile.ZipFile(io.BytesIO(fetch(C2C_URL))).read("cell2celltrain.csv"))
    save("iranian_churn.csv", fetch(IRAN_URL))
