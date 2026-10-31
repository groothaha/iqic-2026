"""SEC Insider Transactions Data Sets(Form 3/4/5) 분기 ZIP → 장내 매수·매도(P/S)만 parquet.

출처: https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets
저장: ~/data/us/sec_insider/ps/<YYYYqN>.parquet  (원본 ZIP은 추출 후 삭제 — 디스크 여유 부족)
PIT 기준은 FILING_DATE(공시일). TRANS_DATE는 체결일이라 신호 시점으로 쓰면 미래 정보 누수.
"""
import io, os, sys, time, zipfile
from pathlib import Path
import pandas as pd
import requests

OUT = Path.home() / "data/us/sec_insider/ps"
RAW = Path.home() / "data/us/sec_insider/raw"
UA = {"User-Agent": os.environ.get("SEC_USER_AGENT", "")}   # SEC 정책: "이름 이메일" 형식
if not UA["User-Agent"]:
    sys.exit("SEC_USER_AGENT 환경 변수 필요 (예: export SEC_USER_AGENT='Name name@example.com')")
URL = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{q}_form345.zip"

SUB = ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK", "ISSUERTRADINGSYMBOL"]
OWN = ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"]
TRN = ["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE",
       "TRANS_ACQUIRED_DISP_CD", "SHRS_OWND_FOLWNG_TRANS", "DIRECT_INDIRECT_OWNERSHIP"]


def read(z, name, cols):
    with z.open(name) as f:
        return pd.read_csv(f, sep="\t", usecols=cols, dtype=str, quoting=3, on_bad_lines="skip")


def one(q: str) -> str:
    dst = OUT / f"{q}.parquet"
    if dst.exists():
        return "있음"
    zp = RAW / f"{q}.zip"
    if not zp.exists():
        r = requests.get(URL.format(q=q), headers=UA, timeout=120)
        if r.status_code != 200:
            return f"HTTP {r.status_code}"
        zp.write_bytes(r.content)
    with zipfile.ZipFile(zp) as z:
        t = read(z, "NONDERIV_TRANS.tsv", TRN)
        t = t[t.TRANS_CODE.isin(["P", "S"])]
        s = read(z, "SUBMISSION.tsv", SUB)
        o = read(z, "REPORTINGOWNER.tsv", OWN)
    d = t.merge(s, on="ACCESSION_NUMBER", how="inner").merge(o, on="ACCESSION_NUMBER", how="left").copy()
    for c in ["FILING_DATE", "TRANS_DATE"]:
        d[c] = pd.to_datetime(d[c], format="%d-%b-%Y", errors="coerce")
    for c in ["TRANS_SHARES", "TRANS_PRICEPERSHARE", "SHRS_OWND_FOLWNG_TRANS"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["quarter"] = q
    d.to_parquet(dst, index=False)
    zp.unlink()
    return f"{len(d):,}행"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
    qs = [f"{y}q{k}" for y in range(2009, 2027) for k in range(1, 5)]
    for q in qs:
        try:
            msg = one(q)
        except Exception as e:
            msg = f"실패 {e!r}"
        print(q, msg, flush=True)
        if msg.startswith("HTTP 404") and q >= "2026q1":
            break
        time.sleep(0.3)
