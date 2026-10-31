"""SEC submissions API → 8-K Item 2.02(실적 발표) 공시일. ~/iqic/out/ea_dates.parquet (ticker, cik, filing_date, acceptance, items).

대상 = contest_sim 유니버스(~/us_quant2/data/raw closes 의 티커). 2005 이전 페이지는 건너뜀. SEC 요청 한도 10/s → 8/s 로 제한.
"""
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
import requests

H = Path.home()
UA = {"User-Agent": os.environ.get("SEC_USER_AGENT", "")}   # SEC 정책: "이름 이메일" 형식
if not UA["User-Agent"]:
    sys.exit("SEC_USER_AGENT 환경 변수 필요 (예: export SEC_USER_AGENT='Name name@example.com')")
_lock = threading.Lock()
_last = [0.0]


def get(url):
    for k in range(4):
        with _lock:
            wait = _last[0] + 0.125 - time.time()
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.time()
        try:
            r = requests.get(url, headers=UA, timeout=30)
        except requests.RequestException:
            time.sleep(2 ** k); continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return None
        time.sleep(2 ** k)
    return None


def pick(block, t, cik):
    return [(t, cik, d, a, it) for f, d, a, it in
            zip(block["form"], block["filingDate"], block["acceptanceDateTime"], block["items"])
            if f == "8-K" and "2.02" in (it or "")]


def one(tc):
    t, cik = tc
    d = get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    if d is None:
        return t, None
    out = pick(d["filings"]["recent"], t, cik)
    for p in d["filings"].get("files", []):
        if p["filingTo"] < "2005-01-01":
            continue
        q = get(f"https://data.sec.gov/submissions/{p['name']}")
        if q is None:
            return t, None
        out += pick(q, t, cik)
    return t, out


if __name__ == "__main__":
    tick = list(pd.read_parquet(H / "us_quant2/data/raw/closes.parquet").columns)
    m = {v["ticker"]: int(v["cik_str"]) for v in get("https://www.sec.gov/files/company_tickers.json").values()}
    todo = [(t, m[t]) for t in tick if t in m]
    miss = [t for t in tick if t not in m]
    print(f"티커 {len(tick)} · CIK 매핑 {len(todo)} · 미매핑 {len(miss)}: {miss[:40]}", flush=True)
    rows, fail = [], []
    with ThreadPoolExecutor(4) as ex:
        for i, (t, r) in enumerate(ex.map(one, todo)):
            if r is None:
                fail.append(t)
            else:
                rows += r
            if i % 100 == 0:
                print(i, len(rows), len(fail), flush=True)
    D = pd.DataFrame(rows, columns=["ticker", "cik", "filing_date", "acceptance", "items"])
    D["filing_date"] = pd.to_datetime(D.filing_date)
    D = D.drop_duplicates(["ticker", "filing_date"]).sort_values(["ticker", "filing_date"])
    D.to_parquet(H / "iqic/out/ea_dates.parquet", index=False)
    cov = D.groupby(D.filing_date.dt.year).ticker.nunique()
    print(f"2.02 공시 {len(D):,}건 · 종목 {D.ticker.nunique()} · 실패 {len(fail)}: {fail[:20]}", flush=True)
    print("연도별 2.02 공시 종목 수:", {int(k): int(v) for k, v in cov.items() if k >= 2004}, flush=True)
