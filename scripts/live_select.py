"""E0 실전 선정 — 현재 S&P 1500 중 실적 발표 완료 + 252일 변동성 낮은 순 30개, 정수 주식 수.

규칙(PREREG_earnings.md E0 그대로): 시작일 i 기준 2.02 공시일이 거래일 [i−30, i−2] 안 → 발표 완료.
그중 252일 일변동성 낮은 순 30개, 동일가중($100,000/30), 정수 주식(1주도 못 사면 건너뛰고 다음 순위).
실행: python3 scripts/live_select.py [시작일=2026-11-02]  → out/live_e0_<기준일>.csv
"""
import io
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))
from fetch_8k import get, pick

H = Path.home()
CAP, K = 100_000, 30
SECTOR_MAX = 6
WIKI = {"S&P 500": "List_of_S%26P_500_companies", "S&P 400": "List_of_S%26P_400_companies",
        "S&P 600": "List_of_S%26P_600_companies"}


def sp1500():
    out = []
    for idx, page in WIKI.items():
        html = requests.get(f"https://en.wikipedia.org/wiki/{page}", headers={"User-Agent": "Mozilla/5.0 research"}, timeout=30).text
        for t in pd.read_html(io.StringIO(html)):
            sym = next((c for c in t.columns if str(c).lower() in ("symbol", "ticker symbol", "ticker")), None)
            sec = next((c for c in t.columns if "gics sector" in str(c).lower()), None)
            if sym is not None and sec is not None and len(t) > 100:
                name = next((c for c in t.columns if str(c).lower() in ("security", "company")), sym)
                out.append(pd.DataFrame({"ticker": t[sym].astype(str).str.replace(".", "-", regex=False),
                                         "name": t[name], "sector": t[sec], "index": idx}))
                break
    return pd.concat(out).drop_duplicates("ticker").reset_index(drop=True)


if __name__ == "__main__":
    start = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2026-11-02")
    U = sp1500()
    print(f"S&P 1500 구성 {len(U)}종목 ({U['index'].value_counts().to_dict()})", flush=True)

    px = yf.download(U.ticker.tolist(), period="2y", auto_adjust=True, progress=False, threads=True)["Close"]
    px = px.dropna(how="all")
    asof = px.index[-1]
    r = px.pct_change(fill_method=None)
    ok = px.iloc[-253:].notna().all()                     # 1년치 가격 완비(백테스트와 같은 조건)
    vol = r.iloc[-252:].std() * np.sqrt(252)
    last = px.iloc[-1]
    print(f"가격 기준일 {asof.date()} · 1년 완비 {ok.sum()}종목", flush=True)

    # 거래일 위치: 기준일 이후는 평일로 연장(9/21~11/2 사이 NYSE 휴장일 없음)
    cal = px.index.append(pd.bdate_range(asof + pd.Timedelta(days=1), start))
    i = cal.get_loc(start)
    lo, hi = cal[i - 30], cal[i - 2]
    print(f"발표 완료 구간 {lo.date()} ~ {hi.date()} (지금 시점까지 공시만 반영)", flush=True)

    m = {v["ticker"]: int(v["cik_str"]) for v in get("https://www.sec.gov/files/company_tickers.json").values()}
    U["cik"] = U.ticker.map(m)
    fdate = {}
    for n_, (t, cik) in enumerate(U.dropna(subset=["cik"])[["ticker", "cik"]].itertuples(index=False)):
        d = get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json")
        if d is not None:
            ds = [pd.Timestamp(x[2]) for x in pick(d["filings"]["recent"], t, cik)]
            ds = [x for x in ds if lo <= x <= hi]
            if ds:
                fdate[t] = max(ds)
        if n_ % 300 == 0:
            print(f"  8-K {n_}/{U.cik.notna().sum()}", flush=True)
    U["filed"] = U.ticker.map(fdate)
    U["vol"] = U.ticker.map(vol); U["price"] = U.ticker.map(last); U["full"] = U.ticker.map(ok).fillna(False)
    print(f"CIK 미매핑 {U.cik.isna().sum()} · 발표 완료 {U.filed.notna().sum()}종목", flush=True)

    C = U[U.full & U.filed.notna() & U.vol.notna()].sort_values("vol").reset_index(drop=True)
    C["shares"] = np.floor(CAP / K / C.price).astype(int)
    keep, cnt, skipped, capped = [], {}, [], []          # 섹터당 최대 SECTOR_MAX (PREREG_risk.md R_sector)
    for j, row in C.iterrows():
        if row.shares < 1:
            skipped.append(row.ticker); continue
        if cnt.get(row.sector, 0) >= SECTOR_MAX:
            capped.append(row.ticker); continue
        keep.append(j); cnt[row.sector] = cnt.get(row.sector, 0) + 1
        if len(keep) == K:
            break
    pick30 = C.loc[keep].copy()
    pick30["amount"] = pick30.shares * pick30.price
    out = H / f"iqic/out/live_e0_{asof.date()}.csv"
    pick30.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print(pick30[["ticker", "name", "sector", "index", "vol", "price", "shares", "amount", "filed"]]
          .assign(vol=lambda x: (x.vol * 100).round(1), price=lambda x: x.price.round(2), amount=lambda x: x.amount.round(0),
                  filed=lambda x: x.filed.dt.date).to_string())
    print(f"\n섹터 한도로 건너뜀: {capped}")
    print(f"고가로 건너뜀: {skipped} · 투자액 ${pick30.amount.sum():,.0f} (현금 ${CAP - pick30.amount.sum():,.0f})")
    print("섹터 분포:", pick30.sector.value_counts().to_dict())
    print(f"저장: {out}")
