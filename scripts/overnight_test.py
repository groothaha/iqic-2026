"""오버나잇 프리미엄(일반형) — PREREG_overnight.md 검정.

A = 밤에만 QQQ 50% + SPY 50%, B = E0 + 밤 오버레이(QQQ 50% + SPY 50%, 대출 8%), B50 = 오버레이 절반(강건성).
주: 2·5·8·11월 창(E0 기준) · 보조: 매월 창(A vs LV30). 출력: out/overnight_test.parquet, 표준출력 요약·판정.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import contest_sim as cs

N = 300
K = 30
COST = 0.0001          # 밤 왕복 1bp
LOAN = 0.08


def tstat(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 else np.nan


def night_day(path):
    e = pd.read_parquet(path)
    e.index = pd.to_datetime(e.index).tz_localize(None).normalize()
    f = e["Adj Close"] / e["Close"]
    night = e.Open * f / (e.Close * f).shift(1) - 1
    day = e.Close / e.Open - 1
    return night, day


if __name__ == "__main__":
    c, v, r, spy, qqq, rf, gold = cs.load()
    dates = c.index; W = cs.W; T = len(dates)
    nq, dq = night_day(cs.BT / "QQQ_daily.parquet")
    ns, ds = night_day(cs.BT / "SPY_daily_full.parquet")
    on = (0.5 * nq + 0.5 * ns).reindex(dates)
    cal = pd.Series(dates, index=dates).diff().dt.days.fillna(1).values      # 직전 종가부터 달력일
    X = r.sub(spy, axis=0).values
    col = {t: j for j, t in enumerate(c.columns)}
    ea = pd.read_parquet(cs.H / "iqic/out/ea_dates.parquet")
    ea["pos"] = dates.searchsorted(ea.filing_date)
    ea = ea[(ea.pos >= 1) & (ea.pos <= T - 2)]
    ev = {t: np.sort(g.pos.unique()) for t, g in ea.groupby("ticker") if t in col}

    print("QQQ·SPY 밤/낮 평균 (bp/일):")
    for a, b_ in [("2006–14", ("2006", "2014")), ("2015–19", ("2015", "2019")), ("2020–24", ("2020", "2024"))]:
        print(f"  {a}: QQQ 밤 {nq[b_[0]:b_[1]].mean()*1e4:.2f} 낮 {dq[b_[0]:b_[1]].mean()*1e4:.2f} · "
              f"SPY 밤 {ns[b_[0]:b_[1]].mean()*1e4:.2f} 낮 {ds[b_[0]:b_[1]].mean()*1e4:.2f}")

    starts = pd.Series(dates, index=dates).groupby(dates.to_period("M")).first()
    starts = [d for d in starts if d >= pd.Timestamp("2006-01-01") and dates.get_loc(d) + W < T]
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
        o = on.iloc[i:i + W].values
        if np.isnan(o).any():
            continue
        n = len(U)
        z = np.zeros(3)

        def eq(idx):
            w = np.zeros(n); w[idx] = 1 / len(idx); return cs.port_returns(w, z, Rs, m, q, f, g)

        ports = {"LV30": eq(ctx["lowvol_top"][:K]), "A": o - COST}
        quarterly = d.month in (2, 5, 8, 11)
        if quarterly:
            e_rep = np.full(n, np.nan)
            for j, t in enumerate(U):
                p = ev.get(t)
                if p is None:
                    continue
                k_ = np.searchsorted(p, i - 1)
                if k_ > 0 and p[k_ - 1] >= i - 30:
                    x = X[p[k_ - 1] - 1:p[k_ - 1] + 2, col[t]]
                    e_rep[j] = x.sum() if np.isfinite(x).all() else np.nan
            e0 = eq(np.array([j for j in ctx["lowvol_top"] if np.isfinite(e_rep[j])][:K], dtype=int))
            loan = LOAN * cal[i:i + W] / 365
            ports.update({"E0": e0, "B": e0 + (o - COST) - loan, "B50": e0 + 0.5 * ((o - COST) - loan)})
        mret = np.prod(1 + m) - 1
        for fname, mix in cs.FIELDS.items():
            kinds = cs.rng.choice(len(mix), N, p=[p_ for p_, _ in mix])
            P = np.column_stack([cs.port_returns(*cs.field_weights(mix[k_][1], ctx), Rs, m, q, f, g) for k_ in kinds])
            repsc = np.clip(cs.rng.normal(45, 13, N), 10, 80) + np.clip(cs.rng.normal(15, 3, N), 0, 20)
            for arm, x in ports.items():
                mt = cs.metrics(np.column_stack([P, x]), m, f)
                sc = cs.score(mt)
                rank = (sc > sc[-1]).sum() + 1
                mex = m - f; xe = x - f
                b_ = np.cov(xe, mex)[0, 1] / mex.var(ddof=1)
                rows.append(dict(field=fname, start=d, quarterly=quarterly, arm=arm, score=sc[-1],
                                 bottom25=rank > (N + 1) * 0.75,
                                 win_R70=((sc[:-1] + repsc) > (sc[-1] + 90)).sum() < 3,
                                 win_R75=((sc[:-1] + repsc) > (sc[-1] + 95)).sum() < 3,
                                 ret=np.prod(1 + x) - 1, mret=mret, beta=b_))
    D = pd.DataFrame(rows)
    D.to_parquet(cs.H / "iqic/out/overnight_test.parquet")
    pd.set_option("display.width", 200)
    Q = D[D.quarterly]
    print(f"\n창: 주 {Q.start.nunique()} · 보조(전체 월) {D.start.nunique()}")

    res = {}
    for fname in cs.FIELDS:
        Df = Q[Q.field == fname]
        S = Df.groupby("arm")[["score", "win_R70", "win_R75", "bottom25", "ret", "beta"]].mean()
        print(f"\n=== 주 · {fname}"); print(S.loc[["E0", "A", "B", "B50", "LV30"]].round(3).to_string())
        piv = Df.pivot(index="start", columns="arm", values="score")
        mr = Df[Df.arm == "E0"].set_index("start").mret
        for arm in ["A", "B", "B50"]:
            dd = piv[arm] - piv["E0"]
            p15 = dd[dd.index.year >= 2015]; p20 = dd[dd.index.year >= 2020]
            print(f"  {arm}−E0: 전체 {dd.mean():+.2f} (t {tstat(dd):+.2f}) · 2006–14 {dd[dd.index.year <= 2014].mean():+.2f}"
                  f" · 2015–24 {p15.mean():+.2f} · 2020–24 {p20.mean():+.2f} (n {len(p20)}) · 상승장 {dd[mr > 0].mean():+.2f} · 하락장 {dd[mr <= 0].mean():+.2f}")
            res.setdefault(arm, []).append((dd.mean() > 0 and p15.mean() >= -1.0 and p20.mean() >= -1.0, dd.mean()))
    for fname in cs.FIELDS:
        piv = D[D.field == fname].pivot(index="start", columns="arm", values="score")
        dd = piv["A"] - piv["LV30"]
        print(f"\n보조 · {fname}: A−LV30 전체 {dd.mean():+.2f} (t {tstat(dd):+.2f}) · 2015–24 {dd[dd.index.year >= 2015].mean():+.2f}"
              f" · 2020–24 {dd[dd.index.year >= 2020].mean():+.2f} (n {(dd.index.year >= 2020).sum()})")

    passed = {a: all(p for p, _ in res[a]) for a in ["A", "B"]}
    print(f"\n판정: A {'통과' if passed['A'] else '실패'} · B {'통과' if passed['B'] else '실패'}")
    cand = [a for a in ["A", "B"] if passed[a]]
    print("→", (max(cand, key=lambda a: np.mean([s for _, s in res[a]])) + " 채택") if cand else "E0 유지")
