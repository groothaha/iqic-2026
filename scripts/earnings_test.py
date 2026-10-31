"""실적 발표 마친 저변동 + 발표 반응(EAR) — PREREG_earnings.md 검정.

포트 LV30/E0/E1/E2 를 2·5·8·11월 첫 거래일 15일 창에서 contest_sim 과 같은 가상 참가자·채점으로 비교.
출력: out/earnings_test.parquet(창×모형×포트), out/earnings_mech.parquet(창별 메커니즘), 표준출력 요약·판정.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import contest_sim as cs

N = 300
POOL = 60
K = 30


def tstat(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 else np.nan


if __name__ == "__main__":
    c, v, r, spy, qqq, rf, gold = cs.load()
    dates = c.index; W = cs.W; T = len(dates)
    X = r.sub(spy, axis=0).values                          # 종목 − SPY 일수익
    col = {t: j for j, t in enumerate(c.columns)}
    ea = pd.read_parquet(cs.H / "iqic/out/ea_dates.parquet")
    ea["pos"] = dates.searchsorted(ea.filing_date)         # 휴일 공시 → 다음 거래일
    ea = ea[(ea.pos >= 1) & (ea.pos <= T - 2)]
    ev = {t: np.sort(g.pos.unique()) for t, g in ea.groupby("ticker") if t in col}

    def ear(t, p):
        x = X[p - 1:p + 2, col[t]]
        return x.sum() if np.isfinite(x).all() else np.nan

    starts = pd.Series(dates, index=dates).groupby(dates.to_period("M")).first()
    starts = [d for d in starts if d >= pd.Timestamp("2006-01-01") and d.month in (2, 5, 8, 11)
              and dates.get_loc(d) + W < T]
    spx = (1 + spy.fillna(0)).cumprod()
    rows, mech = [], []
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
        # 발표 완료 = 2.02 공시 위치가 [i−30, i−2], 가장 최근 공시의 EAR
        e_rep = np.full(n, np.nan); in_win = np.zeros(n, bool)
        for j, t in enumerate(U):
            p = ev.get(t)
            if p is None:
                continue
            k = np.searchsorted(p, i - 1)
            if k > 0 and p[k - 1] >= i - 30:
                e_rep[j] = ear(t, p[k - 1])
            in_win[j] = ((p >= i) & (p <= i + W - 1)).any()
        rep = np.isfinite(e_rep)
        pool = np.array([j for j in ctx["lowvol_top"] if rep[j]][:POOL], dtype=int)
        b100 = ctx["lowvol_top"][:100]
        rv = Rs[:, b100].std(0, ddof=1); rb = rep[b100]
        mech.append(dict(start=d, n_univ=n, rep_share=rep.mean(), pool=len(pool),
                         rv_diff=(rv[~rb].mean() - rv[rb].mean()) * np.sqrt(252),
                         exante_diff=(vol[b100][~rb].mean() - vol[b100][rb].mean()) * np.sqrt(252),
                         notrep_in_win=in_win[b100][~rb].mean(), rep_in_win=in_win[b100][rb].mean()))
        if len(pool) < K:
            print(f"{d.date()} 풀 {len(pool)} < {K} → 건너뜀", flush=True)
            continue
        ports = {"LV30": ctx["lowvol_top"][:K], "E0": pool[:K],
                 "E1": pool[np.argsort(-e_rep[pool])][:K], "E2": pool[np.argsort(e_rep[pool])][:K]}
        for fname, mix in cs.FIELDS.items():
            kinds = cs.rng.choice(len(mix), N, p=[p_ for p_, _ in mix])
            P = np.column_stack([cs.port_returns(*cs.field_weights(mix[k_][1], ctx), Rs, m, q, f, g) for k_ in kinds])
            repsc = np.clip(cs.rng.normal(45, 13, N), 10, 80) + np.clip(cs.rng.normal(15, 3, N), 0, 20)
            for arm, idx in ports.items():
                w = np.zeros(n); w[idx] = 1 / len(idx)
                x = cs.port_returns(w, np.zeros(3), Rs, m, q, f, g)
                sc = cs.score(cs.metrics(np.column_stack([P, x]), m, f))
                rank = (sc > sc[-1]).sum() + 1
                rows.append(dict(field=fname, start=d, arm=arm, score=sc[-1], bottom25=rank > (N + 1) * 0.75,
                                 win_R70=((sc[:-1] + repsc) > (sc[-1] + 90)).sum() < 3,
                                 win_R75=((sc[:-1] + repsc) > (sc[-1] + 95)).sum() < 3,
                                 ret=np.prod(1 + x) - 1, rvol=x.std(ddof=1) * np.sqrt(252)))
    D = pd.DataFrame(rows); M = pd.DataFrame(mech)
    out = cs.H / "iqic/out"
    D.to_parquet(out / "earnings_test.parquet"); M.to_parquet(out / "earnings_mech.parquet")
    pd.set_option("display.width", 200)

    print(f"\n창 {D.start.nunique()}개 · 유니버스 중앙 {M.n_univ.median():.0f} · 발표 완료 비율 중앙 {M.rep_share.median():.0%}"
          f" · 풀<{POOL} 창 {(M.pool < POOL).sum()}")
    print("연도별 발표 완료 비율:", {int(k): round(v_, 2) for k, v_ in M.groupby(M.start.dt.year).rep_share.mean().items()})
    print("월별 발표 완료 비율:", {int(k): round(v_, 2) for k, v_ in M.groupby(M.start.dt.month).rep_share.mean().items()})

    a1 = {}
    for fname in cs.FIELDS:
        Df = D[D.field == fname]
        S = Df.groupby("arm")[["score", "win_R70", "win_R75", "bottom25", "ret", "rvol"]].mean()
        print(f"\n=== {fname} ===")
        print(S.loc[["LV30", "E0", "E1", "E2"]].round(3).to_string())
        piv = Df.pivot(index="start", columns="arm", values="score")
        for arm in ["E1", "E0", "E2"]:
            dd = piv[arm] - piv["LV30"]
            late = dd[dd.index.year >= 2015]; early = dd[dd.index.year <= 2014]; nov = dd[dd.index.month == 11]
            print(f"  {arm}−LV30 점수차: 전체 {dd.mean():+.2f} (t {tstat(dd):+.2f}) · 2006–14 {early.mean():+.2f}"
                  f" · 2015–24 {late.mean():+.2f} (t {tstat(late):+.2f}) · 11월만 {nov.mean():+.2f} (n {len(nov)})")
            a1.setdefault(arm, []).append(dd.mean() >= -0.5 and tstat(dd) >= -1.0 and late.mean() >= -1.0)

    R = D[D.field == "학생형"].pivot(index="start", columns="arm", values="ret")
    m2 = (R.E1 - R.E2) * 100
    print(f"\nM2 발표 후 흐름 (E1−E2 15일 수익, %p): 평균 {m2.mean():+.2f} (t {tstat(m2):+.2f})"
          f" · 2006–14 {m2[m2.index.year <= 2014].mean():+.2f} · 2015–24 {m2[m2.index.year >= 2015].mean():+.2f}"
          f" (t {tstat(m2[m2.index.year >= 2015]):+.2f})")
    print(f"M1 불확실성 해소 (하위 100개, 미발표−발표완료 실현변동성 연율 %p): 평균 {M.rv_diff.mean()*100:+.2f}"
          f" (t {tstat(M.rv_diff):+.2f}) · 사전 252일 변동성 차 {M.exante_diff.mean()*100:+.2f}"
          f" · 미발표 중 창 안 발표 {M.notrep_in_win.mean():.0%} · 발표완료 중 창 안 재공시 {M.rep_in_win.mean():.0%}")

    ok1, ok0 = all(a1["E1"]), all(a1["E0"])
    m1ok = M.rv_diff.mean() > 0 and tstat(M.rv_diff) >= 2
    m2ok = m2.mean() > 0 and tstat(m2) >= 2
    print(f"\n판정: A1(E1) {'통과' if ok1 else '실패'} · A1(E0) {'통과' if ok0 else '실패'} · "
          f"M1 {'통과' if m1ok else '실패'} · M2 {'통과' if m2ok else '실패'}")
    print("→", "E1 채택" if ok1 else ("E0 채택" if ok0 else "LV30 유지"))
