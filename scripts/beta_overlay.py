"""베타 조정 — 선정 30종목의 사전 베타 β̂ 로 S&P 500 선물(MES) 매수 계약 수를 정한다.

규칙(PREREG_beta.md 의 B_x 와 같은 식): β̂ = 30종목 각각의 252일 베타(일수익률, SPY 대비) 평균.
노셔널 = (0.6 − β̂) × $100,000 → MES(지수 × $5) 정수 계약으로 올림 (점수 곡선이 0.6 위로는 평평, 아래로는 하락).
MES 를 주문할 수 없으면 같은 노셔널을 S&P 500 ETF 로 증거금 매수, SPY·IVV 반씩(단일 포지션 50% 한도).
β̂ 가 0.6 이상이면 선물 없음.
실행: python3 scripts/beta_overlay.py [기준일=2026-10-30]  → out/beta_overlay_<기준일>.csv
"""
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

H = Path.home() / "iqic"
CAP, TARGET, MULT = 100_000, 0.6, 5


def beta(x, m):
    return float(np.cov(x, m)[0, 1] / np.var(m, ddof=1))


asof = sys.argv[1] if len(sys.argv) > 1 else "2026-10-30"
sel = pd.read_csv(H / f"out/live_e0_{asof}.csv")
tk = sel.ticker.tolist()
end = (pd.Timestamp(asof) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
px = yf.download(tk + ["SPY", "IVV", "MES=F"], start="2024-06-01", end=end, auto_adjust=True, progress=False)["Close"]
assert px.index[-1].strftime("%Y-%m-%d") == asof, f"마지막 가격일 {px.index[-1].date()} ≠ {asof}"
r = px[tk + ["SPY"]].dropna(subset=["SPY"]).pct_change(fill_method=None).iloc[-252:]
assert r[tk].notna().all().all(), "252일 수익률 결측"

bhat = float(np.mean([beta(r[t], r.SPY) for t in tk]))
w = sel.set_index("ticker").amount
rp = (r[tk] * (w / w.sum())).sum(axis=1)
port = {n: beta(rp.iloc[-n:], r.SPY.iloc[-n:]) for n in (15, 63, 126, 252)}

mes = float(px["MES=F"].iloc[-1])
need = max(0.0, (TARGET - bhat) * CAP)
k = math.ceil(need / (MULT * mes)) if need > 0 else 0
notional = k * MULT * mes
spy, ivv = float(px.SPY.iloc[-1]), float(px.IVV.iloc[-1])
spy_sh, ivv_sh = math.floor(notional / 2 / spy), math.floor(notional / 2 / ivv)

print(f"기준일 {asof} · β̂(252일 종목 베타 평균) {bhat:+.3f}")
print("포트 베타(실제 금액 비중): " + " · ".join(f"{n}일 {b:+.2f}" for n, b in port.items()))
print(f"필요 노셔널 ${need:,.0f} → MES {k}계약 (가격 {mes:,.2f}, 노셔널 ${notional:,.0f}) · 예상 베타 {bhat + notional / CAP:.2f}")
print(f"MES 불가 시: SPY {spy_sh}주(${spy_sh * spy:,.0f}) + IVV {ivv_sh}주(${ivv_sh * ivv:,.0f}), 증거금 매수")
out = pd.DataFrame([dict(asof=asof, bhat=bhat, **{f"beta_port_{n}d": b for n, b in port.items()},
                         need=need, mes_price=mes, mes_contracts=k, notional=notional,
                         beta_expected=bhat + notional / CAP, spy_shares=spy_sh, ivv_shares=ivv_sh)])
out.to_csv(H / f"out/beta_overlay_{asof}.csv", index=False)
print("저장:", H / f"out/beta_overlay_{asof}.csv")
