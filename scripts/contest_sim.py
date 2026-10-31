"""IQIC 트레이딩 점수 시뮬레이션 — 어떤 포트가 백분위 순위를 잘 받나.

대회 배점(트레이딩 100): 전체 참가자 대비 백분위 Sharpe 40 · Alpha 30 · Treynor 20 · 절대수익 10.
과거 15거래일 창(매월 첫 거래일 시작, 2006~2024)마다 가상 참가자 N명 + 후보 전략 1개씩 넣고 점수 계산.

가정(대회 규정 미공개라 전부 추정):
  - 벤치 = SPY, 무위험 = 13주 T-bill(^IRX), 지표는 일간 수익 15개로 계산.
  - Alpha = Jensen(일평균 초과 − β·시장 일평균 초과), Treynor = 기간 초과수익 / β.
  - 참가자 행동은 FIELDS 의 혼합(학생 대회라 소수 종목 집중·인기주·추격매수 위주로 가정).
데이터: ~/us_quant2/data/raw (S&P1500 현재 구성 일봉, 생존편향 — 참가자·후보 모두에 같은 방향).
"""
from pathlib import Path
import numpy as np
import pandas as pd

H = Path.home()
RAW = H / "us_quant2/data/raw"
BT = H / "AAAAAAbacktest/tier_a"
W = 15
rng = np.random.default_rng(7)


def load():
    c = pd.read_parquet(RAW / "closes.parquet")
    v = pd.read_parquet(RAW / "volumes.parquet").reindex_like(c)
    mac = pd.read_parquet(RAW / "macro.parquet").reindex(c.index).ffill()
    spy = pd.read_parquet(BT / "SPY_daily_full.parquet")["Close"]
    qqq = pd.read_parquet(BT / "QQQ_daily.parquet")["Close"]
    for s in (spy, qqq):
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    spy = spy.reindex(c.index).ffill(); qqq = qqq.reindex(c.index).ffill()
    r = c.pct_change(fill_method=None).where(lambda x: x.abs() < 0.5)
    rf = (mac["irx_2y"] / 100 / 252).fillna(0)          # ^IRX(13주) 연% → 일
    gold = mac["gold"].pct_change(fill_method=None)
    return c, v, r, spy.pct_change(), qqq.pct_change(), rf, gold


def metrics(R, m, rf):
    """R: (W, K) 일간 수익. 반환 dict of (K,) 배열."""
    ex = R - rf[:, None]; mex = m - rf
    sd = R.std(0, ddof=1)
    sharpe = np.where(sd > 1e-9, ex.mean(0) / np.maximum(sd, 1e-12), 0.0)
    vm = mex.var(ddof=1)
    beta = ((ex - ex.mean(0)) * (mex - mex.mean())[:, None]).sum(0) / (W - 1) / vm
    alpha = ex.mean(0) - beta * mex.mean()
    tot = np.prod(1 + R, 0) - 1
    exc_tot = tot - (np.prod(1 + rf) - 1)
    b = np.where(np.abs(beta) < 0.05, np.sign(beta + 1e-12) * 0.05, beta)   # β≈0 폭주 방지(가정)
    treynor = exc_tot / b
    return dict(sharpe=sharpe, alpha=alpha, treynor=treynor, ret=tot)


def pct_rank(x):
    o = x.argsort().argsort()
    return o / (len(x) - 1)


def score(mt):
    return 40 * pct_rank(mt["sharpe"]) + 30 * pct_rank(mt["alpha"]) + \
           20 * pct_rank(mt["treynor"]) + 10 * pct_rank(mt["ret"])


# ---------------------------------------------------------------- 참가자 모형
FIELDS = {
    # 이름: [(비율, 유형)]
    "학생형": [(0.35, "popular"), (0.25, "chase"), (0.20, "random10"), (0.10, "etf"), (0.10, "lottery")],
    "숙련형": [(0.20, "popular"), (0.15, "chase"), (0.25, "random10"), (0.30, "etf"), (0.10, "lowvol")],
}


def field_weights(kind, ctx):
    """참가자 1명의 종목 가중 벡터(유니버스 길이) + ETF 비중 (spy, qqq, cash)."""
    U = ctx["univ"]; n = len(U)
    w = np.zeros(n); e = np.zeros(3)
    if kind == "popular":                     # 거래대금 상위 50 중 5~10개, 인기 비례
        k = rng.integers(5, 11); p = ctx["dv_rank_top50"]
        idx = rng.choice(p, k, replace=False); w[idx] = 1 / k
    elif kind == "chase":                     # 최근 3개월 수익 상위 10% 중 5~10개
        k = rng.integers(5, 11); idx = rng.choice(ctx["mom_top"], k, replace=False); w[idx] = 1 / k
    elif kind == "random10":
        k = rng.integers(10, 21); idx = rng.choice(n, k, replace=False); w[idx] = 1 / k
    elif kind == "lottery":
        k = rng.integers(2, 5); idx = rng.choice(ctx["mom_top"], k, replace=False); w[idx] = 1 / k
    elif kind == "lowvol":                    # 변동성 하위 60개 중 15~30개 (10/26 수정: 전체 유니버스에서 뽑던 버그)
        k = rng.integers(15, 31); idx = rng.choice(ctx["lowvol_top"][:60], k, replace=False); w[idx] = 1 / k
    elif kind == "etf":
        a = rng.dirichlet([1, 1, 0.3]); e[:] = a
    return w, e


# ---------------------------------------------------------------- 후보 전략
def candidates(ctx):
    U = ctx["univ"]; n = len(U)
    def eq(idx):
        w = np.zeros(n); w[idx] = 1 / len(idx); return w, np.zeros(3)
    z = np.zeros(n)
    out = {
        "SPY 100%": (z, np.array([1, 0, 0.])),
        "QQQ 100%": (z, np.array([0, 1, 0.])),
        "SPY 50% + 현금 50%": (z, np.array([0.5, 0, 0.5])),
        "현금(T-bill) 100%": (z, np.array([0, 0, 1.])),
        "SPY 추세게이트(200일선 위면 SPY, 아래면 현금)": (z, np.array([1, 0, 0.]) if ctx["spy_up"] else np.array([0, 0, 1.])),
        "저변동 30종목": eq(ctx["lowvol_top"][:30]),
        "모멘텀 20종목(12-1)": eq(ctx["mom12_top"][:20]),
        "고베타 20종목": eq(ctx["hibeta_top"][:20]),
        "모멘텀 5종목 집중": eq(ctx["mom12_top"][:5]),
        "모멘텀 10종목": eq(ctx["mom12_top"][:10]),
        "저변동 10종목": eq(ctx["lowvol_top"][:10]),
        "저변동 5종목": eq(ctx["lowvol_top"][:5]),
        "SPY 50% + 금 50%": ("gold", np.array([0.5, 0, 0.])),
    }
    return out


def port_returns(w, e, Rs, spy, qqq, rf, gold):
    if isinstance(w, str):                         # 금 혼합
        return e[0] * spy + 0.5 * gold
    s = w.sum()
    rs = (np.nan_to_num(Rs) @ w) if s > 0 else 0.0
    return rs + e[0] * spy + e[1] * qqq + e[2] * rf


if __name__ == "__main__":
    c, v, r, spy, qqq, rf, gold = load()
    dates = c.index
    starts = pd.Series(dates, index=dates).groupby(dates.to_period("M")).first()
    starts = [d for d in starts if d >= pd.Timestamp("2006-01-01") and dates.get_loc(d) + W < len(dates)]
    N = 300
    rows = []
    spx = (1 + spy.fillna(0)).cumprod()
    for fname, mix in FIELDS.items():
        for d in starts:
            i = dates.get_loc(d)
            hist = slice(max(0, i - 252), i)
            pc = c.iloc[i - 1]
            ok = pc.notna() & (c.iloc[max(0, i - 252)].notna()) & r.iloc[i:i + W].notna().all()
            U = pc.index[ok]
            if len(U) < 300:
                continue
            Rh = r.iloc[hist][U]
            dv = (c.iloc[i - 63:i][U] * v.iloc[i - 63:i][U]).mean()
            mom3 = c.iloc[i - 1][U] / c.iloc[i - 64][U] - 1
            mom12 = c.iloc[i - 22][U] / c.iloc[i - 252][U] - 1
            vol = Rh.std()
            sh = spy.iloc[hist]
            shc = (sh - sh.mean()).values[:, None]
            beta = pd.Series(np.nanmean((Rh - Rh.mean()).values * shc, 0) / np.nanvar(sh.values), index=U)
            ctx = dict(univ=U,
                       dv_rank_top50=np.argsort(-dv.values)[:50],
                       mom_top=np.argsort(-mom3.values)[:max(20, len(U) // 10)],
                       mom12_top=np.argsort(-mom12.values),
                       lowvol_top=np.argsort(vol.values),
                       hibeta_top=np.argsort(-beta.values),
                       spy_up=bool(spx.iloc[i - 1] > spx.iloc[i - 200:i].mean()))
            Rs = r.iloc[i:i + W][U].values
            m = spy.iloc[i:i + W].values; q = qqq.iloc[i:i + W].values
            f = rf.iloc[i:i + W].values; g = gold.iloc[i:i + W].fillna(0).values
            # 참가자
            kinds = rng.choice(len(mix), N, p=[p for p, _ in mix])
            P = np.column_stack([port_returns(*field_weights(mix[k][1], ctx), Rs, m, q, f, g) for k in kinds])
            mkt_up = (np.prod(1 + m) - 1) > 0
            rep = np.clip(rng.normal(45, 13, N), 10, 80) + np.clip(rng.normal(15, 3, N), 0, 20)   # 참가자 보고서+무결성(가정)
            for name, (w, e) in candidates(ctx).items():
                x = port_returns(w, e, Rs, m, q, f, g)
                A = np.column_stack([P, x])
                mt = metrics(A, m, f)
                sc = score(mt)
                rank = (sc > sc[-1]).sum() + 1
                tot = {R0: ((sc[:-1] + rep) > (sc[-1] + R0 + 20)).sum() + 1 for R0 in (60, 70, 75)}
                rows.append(dict(field=fname, start=d, cand=name, score=sc[-1], rank=rank,
                                 top3=rank <= 3, top10pct=rank <= (N + 1) * 0.1, bottom25=rank > (N + 1) * 0.75,
                                 mkt_up=mkt_up, **{f"win_R{k}": t <= 3 for k, t in tot.items()}, **{f"p_{k}": pct_rank(v_)[-1] for k, v_ in mt.items()}))
    D = pd.DataFrame(rows)
    out = H / "iqic/out"; out.mkdir(exist_ok=True)
    D.to_parquet(out / "contest_sim.parquet")
    pd.set_option("display.width", 220)
    for fname in FIELDS:
        g = D[D.field == fname].groupby("cand")
        T = pd.DataFrame({
            "평균점수": g.score.mean(), "중앙": g.score.median(),
            "P(상위3)": g.top3.mean(), "P(상위10%)": g.top10pct.mean(), "P(하위25%)": g.bottom25.mean(),
            "수상P 보고서60": g.win_R60.mean(), "수상P 보고서70": g.win_R70.mean(), "수상P 보고서75": g.win_R75.mean(),
            "상승장 평균": D[(D.field == fname) & D.mkt_up].groupby("cand").score.mean(),
            "하락장 평균": D[(D.field == fname) & ~D.mkt_up].groupby("cand").score.mean(),
        }).sort_values("수상P 보고서70", ascending=False)
        print(f"\n=== 참가자 모형: {fname} (창 {D[D.field==fname].start.nunique()}개, 상승장 비율 {D[D.field==fname].groupby('start').mkt_up.first().mean():.0%}) ===")
        print(T.round(3).to_string())
