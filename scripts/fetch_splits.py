"""yfinance 분할 이력 → ~/iqic/out/splits.parquet (ticker, date, ratio).

시총 = 수정주가 × (공시 당시 주식수 × 공시일 이후 분할 누적배율). 공시 주식수를 오늘 분할 기준으로 맞추는 용도.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
import yfinance as yf

H = Path.home()


def one(t):
    try:
        s = yf.Ticker(t).splits
        if s is None or len(s) == 0:
            return []
        idx = pd.to_datetime(s.index).tz_localize(None)
        return [(t, d, float(r)) for d, r in zip(idx, s.values) if r > 0]
    except Exception:
        return None


if __name__ == "__main__":
    px = pd.read_parquet(H / "fundamental_ml/data/prices_m.parquet")
    tick = list(px.columns)
    rows, fail = [], []
    with ThreadPoolExecutor(6) as ex:
        for i, (t, r) in enumerate(zip(tick, ex.map(one, tick))):
            if r is None:
                fail.append(t)
            else:
                rows += r
            if i % 250 == 0:
                print(i, len(rows), len(fail), flush=True)
    out = pd.DataFrame(rows, columns=["ticker", "date", "ratio"])
    out.to_parquet(H / "iqic/out/splits.parquet", index=False)
    print(f"분할 {len(out):,}건 · 종목 {out.ticker.nunique():,} · 실패 {len(fail)}: {fail[:20]}", flush=True)
