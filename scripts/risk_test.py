"""E0 리스크 규칙 — PREREG_risk.md 검정.

R_sector(섹터당 최대 6종목) · R_stop(종목 −15% → 다음 날 종가 매도) · R_dd(포트 −8% → 다음 날 종가 절반 현금) · R_all.
모든 포트는 사서 들고 가는 경로로 계산. 출력: out/risk_test.parquet, 표준출력 요약·판정.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import contest_sim as cs

N = 300
K = 30
CAP_N = 6
STOP = -0.15
DD = -0.08

SPECIAL = {4922: "Energy", 4923: "Utilities", 4924: "Utilities", 5331: "Consumer Staples", 5411: "Consumer Staples",
           5912: "Consumer Staples", 5122: "Health Care", 6798: "Real Estate", 3812: "Industrials", 8731: "Health Care"}
RANGES = [  # (시작, 끝, 섹터) — 위에서부터 첫 일치
    (100, 999, "Consumer Staples"), (1300, 1389, "Energy"), (1000, 1499, "Materials"), (1500, 1799, "Industrials"),
    (2000, 2199, "Consumer Staples"), (2200, 2399, "Consumer Discretionary"), (2400, 2499, "Materials"),
    (2500, 2599, "Consumer Discretionary"), (2600, 2699, "Materials"), (2700, 2799, "Communication Services"),
    (2830, 2836, "Health Care"), (2840, 2844, "Consumer Staples"), (2800, 2899, "Materials"), (2900, 2999, "Energy"),
    (3000, 3099, "Materials"), (3100, 3199, "Consumer Discretionary"), (3200, 3399, "Materials"),
    (3400, 3499, "Industrials"), (3570, 3579, "Information Technology"), (3500, 3599, "Industrials"),
    (3600, 3629, "Industrials"), (3630, 3659, "Consumer Discretionary"), (3660, 3699, "Information Technology"),
    (3710, 3716, "Consumer Discretionary"), (3700, 3799, "Industrials"), (3800, 3829, "Information Technology"),
    (3840, 3851, "Health Care"), (3852, 3999, "Consumer Discretionary"), (4000, 4799, "Industrials"),
    (4800, 4899, "Communication Services"), (4950, 4959, "Industrials"), (4900, 4999, "Utilities"),
    (5140, 5149, "Consumer Staples"), (5000, 5199, "Industrials"), (5200, 5999, "Consumer Discretionary"),
    (6500, 6599, "Real Estate"), (6000, 6799, "Financials"), (7000, 7299, "Consumer Discretionary"),
    (7370, 7379, "Information Technology"), (7300, 7399, "Industrials"), (7500, 7599, "Consumer Discretionary"),
    (7800, 7899, "Communication Services"), (7900, 7999, "Consumer Discretionary"), (8000, 8099, "Health Care"),
    (8200, 8299, "Consumer Discretionary"), (8100, 8999, "Industrials"),
]


def sic_sector(s):
    if pd.isna(s):
        return "Unknown"
    s = int(s)
    if s in SPECIAL:
        return SPECIAL[s]
    return next((sec for a, b, sec in RANGES if a <= s <= b), "Other")


def tstat(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 else np.nan


def path(R, f, stop=False, dd=False):
    """R: (W, 30) 보유 종목 일수익. 사서 들고 감, 결정은 종가·집행은 다음 날 종가. 반환 (일수익, 손절 수, 낙폭 발동)."""
    W_, k = R.shape
    val = np.full(k, 1 / k); px = np.ones(k); alive = np.ones(k, bool)
    cash = 0.0; Vp = 1.0; out = np.zeros(W_)
    pend_stop = np.zeros(k, bool); pend_dd = False; dd_done = False; n_stop = 0
    for t in range(W_):
        val[alive] *= 1 + R[t, alive]; px *= 1 + R[t]; cash *= 1 + f[t]
        if pend_stop.any():
            cash += val[pend_stop].sum(); val[pend_stop] = 0; alive &= ~pend_stop; pend_stop[:] = False
        if pend_dd:
            cash += 0.5 * val[alive].sum(); val[alive] *= 0.5; pend_dd = False; dd_done = True
        V = val.sum() + cash
        out[t] = V / Vp - 1; Vp = V
        if stop:
            new = alive & (px - 1 <= STOP) & ~pend_stop
            pend_stop |= new; n_stop += int(new.sum())
        if dd and not dd_done and V - 1 <= DD:
            pend_dd = True
    return out, n_stop, dd_done or pend_dd


if __name__ == "__main__":
    c, v, r, spy, qqq, rf, gold = cs.load()
    dates = c.index; W = cs.W; T = len(dates)
    X = r.sub(spy, axis=0).values
    col = {t: j for j, t in enumerate(c.columns)}
    ea = pd.read_parquet(cs.H / "iqic/out/ea_dates.parquet")
    ea["pos"] = dates.searchsorted(ea.filing_date)
    ea = ea[(ea.pos >= 1) & (ea.pos <= T - 2)]
    ev = {t: np.sort(g.pos.unique()) for t, g in ea.groupby("ticker") if t in col}
    sic = pd.read_parquet(cs.H / "iqic/out/sic.parquet").set_index("ticker").sic
    sector = {t: sic_sector(sic.get(t)) for t in c.columns}
    print("SIC 섹터 분포(유니버스):", pd.Series(sector).value_counts().to_dict(), flush=True)

    starts = pd.Series(dates, index=dates).groupby(dates.to_period("M")).first()
    starts = [d for d in starts if d >= pd.Timestamp("2006-01-01") and d.month in (2, 5, 8, 11)
              and dates.get_loc(d) + W < T]
    spx = (1 + spy.fillna(0)).cumprod()
    rows = []
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
        sh = spy.iloc[hist]
        shc = (sh - sh.mean()).values[:, None]
        beta = pd.Series(np.nanmean((Rh - Rh.mean()).values * shc, 0) / np.nanvar(sh.values), index=U)
        vol = Rh.std().values
        ctx = dict(univ=U, dv_rank_top50=np.argsort(-dv.values)[:50],
                   mom_top=np.argsort(-mom3.values)[:max(20, len(U) // 10)],
                   mom12_top=np.argsort(-mom12.values), lowvol_top=np.argsort(vol),
                   hibeta_top=np.argsort(-beta.values),
                   spy_up=bool(spx.iloc[i - 1] > spx.iloc[i - 200:i].mean()))
        Rs = r.iloc[i:i + W][U].values
        m = spy.iloc[i:i + W].values; q = qqq.iloc[i:i + W].values
        f = rf.iloc[i:i + W].values; g = gold.iloc[i:i + W].fillna(0).values
        n = len(U)
        rep = np.zeros(n, bool)
        for j, t in enumerate(U):
            p = ev.get(t)
            if p is None:
                continue
            k_ = np.searchsorted(p, i - 1)
            if k_ > 0 and p[k_ - 1] >= i - 30:
                x = X[p[k_ - 1] - 1:p[k_ - 1] + 2, col[t]]
                rep[j] = np.isfinite(x).all()
        order = [j for j in ctx["lowvol_top"] if rep[j]]
        e0 = np.array(order[:K], dtype=int)
        cnt, sec_idx = {}, []
        for j in order:
            s = sector[U[j]]
            if cnt.get(s, 0) < CAP_N:
                sec_idx.append(j); cnt[s] = cnt.get(s, 0) + 1
            if len(sec_idx) == K:
                break
        sec_idx = np.array(sec_idx, dtype=int)
        e0_sec = pd.Series([sector[U[j]] for j in e0]).value_counts()
        arms = {"E0": path(Rs[:, e0], f), "R_sector": path(Rs[:, sec_idx], f),
                "R_stop": path(Rs[:, e0], f, stop=True), "R_dd": path(Rs[:, e0], f, dd=True),
                "R_all": path(Rs[:, sec_idx], f, stop=True, dd=True)}
        for fname, mix in cs.FIELDS.items():
            kinds = cs.rng.choice(len(mix), N, p=[p_ for p_, _ in mix])
            P = np.column_stack([cs.port_returns(*cs.field_weights(mix[k_][1], ctx), Rs, m, q, f, g) for k_ in kinds])
            repsc = np.clip(cs.rng.normal(45, 13, N), 10, 80) + np.clip(cs.rng.normal(15, 3, N), 0, 20)
            for arm, (x, n_stop, ddt) in arms.items():
                sc = cs.score(cs.metrics(np.column_stack([P, x]), m, f))
                rank = (sc > sc[-1]).sum() + 1
                rows.append(dict(field=fname, start=d, arm=arm, score=sc[-1], bottom25=rank > (N + 1) * 0.75,
                                 win_R70=((sc[:-1] + repsc) > (sc[-1] + 90)).sum() < 3,
                                 win_R75=((sc[:-1] + repsc) > (sc[-1] + 95)).sum() < 3,
                                 ret=np.prod(1 + x) - 1, rvol=x.std(ddof=1) * np.sqrt(252), n_stop=n_stop, dd=ddt,
                                 e0_maxsec=e0_sec.iloc[0], e0_top=e0_sec.index[0],
                                 replaced=len(set(e0) - set(sec_idx))))
    D = pd.DataFrame(rows)
    D.to_parquet(cs.H / "iqic/out/risk_test.parquet")
    pd.set_option("display.width", 200)
    A = ["E0", "R_sector", "R_stop", "R_dd", "R_all"]
    w1 = D[(D.field == "학생형") & (D.arm == "E0")]
    print(f"\n창 {D.start.nunique()}개 · E0 최대 섹터 종목 수 중앙 {w1.e0_maxsec.median():.0f} · 7개 이상(한도 걸림) 창 {(w1.e0_maxsec > CAP_N).mean():.0%}"
          f" · 교체 종목 수 평균 {w1.replaced.mean():.1f} · 최다 섹터 빈도 {w1.e0_top.value_counts().head(4).to_dict()}")
    res = {}
    for fname in cs.FIELDS:
        Df = D[D.field == fname]
        g_ = Df.groupby("arm")
        S = pd.DataFrame({"점수": g_.score.mean(), "수상70": g_.win_R70.mean(), "수상75": g_.win_R75.mean(),
                          "하위25%": g_.bottom25.mean(), "15일수익": g_.ret.mean(), "하위5%수익": g_.ret.quantile(0.05),
                          "최악창": g_.ret.min(), "실현변동성": g_.rvol.mean()}).loc[A]
        print(f"\n=== {fname}"); print(S.round(4).to_string())
        piv = Df.pivot(index="start", columns="arm", values="score")
        for arm in A[1:]:
            dd_ = piv[arm] - piv["E0"]
            print(f"  {arm}−E0: 전체 {dd_.mean():+.2f} (t {tstat(dd_):+.2f}) · 2015–24 {dd_[dd_.index.year >= 2015].mean():+.2f}")
            res.setdefault(arm, []).append(dd_.mean() >= -1.0 and tstat(dd_) >= -1.0)
    s1 = D[(D.field == "학생형")]
    st, dt = s1[s1.arm == "R_stop"], s1[s1.arm == "R_dd"]
    print(f"\n발동: 손절 창당 평균 {st.n_stop.mean():.2f}종목 · 1건 이상 창 {(st.n_stop > 0).mean():.0%} · 낙폭 기준 발동 창 {dt.dd.mean():.0%}"
          f" ({', '.join(str(x.date()) for x in dt[dt.dd].start)})")
    ok_all = all(res["R_all"])
    print(f"\n판정: R_all {'통과 → 셋 다 채택' if ok_all else '실패'}" +
          ("" if ok_all else " · 개별: " + ", ".join(f"{a} {'통과' if all(res[a]) else '실패'}" for a in A[1:4])))
