"""SEC submissions → 백테스트 유니버스 티커별 SIC 코드. ~/iqic/out/sic.parquet (ticker, cik, sic, sic_desc)."""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fetch_8k import get

H = Path.home()


def one(tc):
    t, cik = tc
    d = get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    return (t, cik, d.get("sic"), d.get("sicDescription")) if d else (t, cik, None, None)


if __name__ == "__main__":
    tick = list(pd.read_parquet(H / "us_quant2/data/raw/closes.parquet").columns)
    m = {v["ticker"]: int(v["cik_str"]) for v in get("https://www.sec.gov/files/company_tickers.json").values()}
    todo = [(t, m[t]) for t in tick if t in m]
    with ThreadPoolExecutor(4) as ex:
        rows = list(ex.map(one, todo))
    D = pd.DataFrame(rows, columns=["ticker", "cik", "sic", "sic_desc"])
    D["sic"] = pd.to_numeric(D.sic, errors="coerce")
    D.to_parquet(H / "iqic/out/sic.parquet", index=False)
    print(f"SIC {D.sic.notna().sum()}/{len(tick)} 종목")
