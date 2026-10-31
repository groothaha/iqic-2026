"""E0 + 외가격 SPY 콜 매수 — PREREG_options.md 검정.

주: 2·5·8·11월 창(E0 기반) · 보조: 매월 창(LV30 기반). 옵션 = optionsDX SPY EOD, ask 매수·mid 평가.
출력: out/options_test.parquet, 표준출력 요약·판정·강건성 표.
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
GRID = [(b, o) for b in (0.005, 0.01, 0.02) for o in (0.0, 0.02, 0.04)]
MAIN = (0.01, 0.02)


def tstat(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 else np.nan


def load_chain():
    fs = sorted((cs.H / "options_data/data/optionsdx/parquet").glob("spy_*.parquet"))
    out = []
    for f in fs:
        d = pd.read_parquet(f, columns=["date", "spot", "expiry", "strike", "c_bid", "c_ask"])
        d = d[(d.expiry - d.date).dt.days.le(45) & (d.strike >= d.spot * 0.97) & (d.strike <= d.spot * 1.10)]
        out.append(d)
    return pd.concat(out, ignore_index=True)


def call_path(ch, f_date, end_date, wdates, otm):
    """f_date 종가에 ask 로 산 콜의 창 중 mid 가치 / 매수가. 없으면 None."""
    day = ch[ch.date == f_date]
    if day.empty:
        return None
    ex = day.expiry[day.expiry >= end_date]
    if ex.empty:
        return None
    e = ex.min(); S0 = day.spot.iloc[0]
    c = day[(day.expiry == e) & (day.strike >= S0 * (1 + otm)) & (day.c_ask > 0)].sort_values("strike")
    if c.empty:
        return None
    k = c.strike.iloc[0]; cost = c.c_ask.iloc[0]
    q = ch[(ch.expiry == e) & (ch.strike == k) & ch.date.isin(wdates)].set_index("date")
    mid = ((q.c_bid + q.c_ask) / 2).reindex(wdates).ffill()
    if mid.isna().any():
        return None
    return mid.values / cost, k / S0 - 1


if __name__ == "__main__":
    c, v, r, spy, qqq, rf, gold = cs.load()
    dates = c.index; W = cs.W; T = len(dates)
    X = r.sub(spy, axis=0).values
    col = {t: j for j, t in enumerate(c.columns)}
    ea = pd.read_parquet(cs.H / "iqic/out/ea_dates.parquet")
    ea["pos"] = dates.searchsorted(ea.filing_date)
    ea = ea[(ea.pos >= 1) & (ea.pos <= T - 2)]
    ev = {t: np.sort(g.pos.unique()) for t, g in ea.groupby("ticker") if t in col}
    ch = load_chain()
    ch = ch[ch.date.isin(dates)]
    print(f"체인 {len(ch):,}행 · {ch.date.min().date()} ~ {ch.date.max().date()}", flush=True)

    starts = pd.Series(dates, index=dates).groupby(dates.to_period("M")).first()
    starts = [d for d in starts if d >= pd.Timestamp("2010-01-01") and dates.get_loc(d) + W < T
              and dates[dates.get_loc(d) + W - 1] <= ch.date.max()]
    spx = (1 + spy.fillna(0)).cumprod()
    rows, skipped = [], []
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
        quarterly = d.month in (2, 5, 8, 11)
        if quarterly:                                      # E0 = 발표 완료 중 저변동 30 (earnings_test 와 같은 정의)
            e_rep = np.full(n, np.nan)
            for j, t in enumerate(U):
                p = ev.get(t)
                if p is None:
                    continue
                k_ = np.searchsorted(p, i - 1)
                if k_ > 0 and p[k_ - 1] >= i - 30:
                    x = X[p[k_ - 1] - 1:p[k_ - 1] + 2, col[t]]
                    e_rep[j] = x.sum() if np.isfinite(x).all() else np.nan
            base_idx = np.array([j for j in ctx["lowvol_top"] if np.isfinite(e_rep[j])][:K], dtype=int)
            base_name = "E0"
        else:
            base_idx = ctx["lowvol_top"][:K]; base_name = "LV30"
        wdates = dates[i:i + W]
        paths = {}
        for b, o in GRID:
            cp = call_path(ch, dates[i - 1], wdates[-1], wdates, o)
            if cp is None:
                break
            paths[(b, o)] = cp
        if len(paths) < len(GRID):
            skipped.append(d); continue
        w = np.zeros(n); w[base_idx] = 1 / K
        xb = cs.port_returns(w, np.zeros(3), Rs, m, q, f, g)
        vb = np.cumprod(1 + xb)
        ports = {"base": xb}
        for (b, o), (cv, _) in paths.items():
            val = (1 - b) * vb + b * cv
            ports[(b, o)] = np.diff(np.concatenate([[1.0], val])) / np.concatenate([[1.0], val[:-1]])
        mret = np.prod(1 + m) - 1
        for fname, mix in cs.FIELDS.items():
            kinds = cs.rng.choice(len(mix), N, p=[p_ for p_, _ in mix])
            P = np.column_stack([cs.port_returns(*cs.field_weights(mix[k_][1], ctx), Rs, m, q, f, g) for k_ in kinds])
            repsc = np.clip(cs.rng.normal(45, 13, N), 10, 80) + np.clip(cs.rng.normal(15, 3, N), 0, 20)
            for arm, x in ports.items():
                sc = cs.score(cs.metrics(np.column_stack([P, x]), m, f))
                rank = (sc > sc[-1]).sum() + 1
                cv, mny = paths[MAIN] if arm == MAIN else (None, None)
                rows.append(dict(field=fname, start=d, quarterly=quarterly, base=base_name,
                                 arm="base" if arm == "base" else f"{arm[0]:.3f}_{arm[1]:.2f}",
                                 score=sc[-1], bottom25=rank > (N + 1) * 0.75,
                                 win_R70=((sc[:-1] + repsc) > (sc[-1] + 90)).sum() < 3,
                                 win_R75=((sc[:-1] + repsc) > (sc[-1] + 95)).sum() < 3,
                                 ret=np.prod(1 + x) - 1, mret=mret,
                                 call_mult=cv[-1] if cv is not None else np.nan, otm_real=mny))
    D = pd.DataFrame(rows)
    D.to_parquet(cs.H / "iqic/out/options_test.parquet")
    pd.set_option("display.width", 200)
    mainarm = f"{MAIN[0]:.3f}_{MAIN[1]:.2f}"
    print(f"\n창: 주 {D[D.quarterly].start.nunique()} · 보조 {D[~D.quarterly].start.nunique()} · 건너뜀 {len(skipped)} {[s.date() for s in skipped][:6]}")

    verdict = {}
    for label, sub in [("주 (E0 기반, 2·5·8·11월)", D[D.quarterly]), ("보조 (LV30 기반, 나머지 월)", D[~D.quarterly])]:
        print(f"\n######## {label}")
        for fname in cs.FIELDS:
            Df = sub[sub.field == fname]
            piv = Df.pivot(index="start", columns="arm", values="score")
            dd = piv[mainarm] - piv["base"]
            mr = Df[Df.arm == "base"].set_index("start").mret
            up, dn, sup = dd[mr > 0], dd[mr <= 0], dd[mr > 0.02]
            early, late = dd[dd.index.year <= 2016], dd[dd.index.year >= 2017]
            S = Df[Df.arm.isin(["base", mainarm])].groupby("arm")[["score", "win_R70", "win_R75", "bottom25", "ret"]].mean()
            print(f"--- {fname}")
            print(S.round(3).to_string())
            print(f"  C1−기반 점수차: 전체 {dd.mean():+.2f} (t {tstat(dd):+.2f}) · 상승장 {up.mean():+.2f} (n {len(up)}) · "
                  f"하락장 {dn.mean():+.2f} (n {len(dn)}) · 강상승(>2%) {sup.mean():+.2f} (n {len(sup)}) · "
                  f"2010–16 {early.mean():+.2f} · 2017–23 {late.mean():+.2f}")
            verdict[(label[:1], fname)] = (dd.mean() >= -0.5 and tstat(dd) >= -1.0, up.mean() > 0)
        cm = sub[(sub.arm == mainarm) & (sub.field == "학생형")]
        print(f"  콜 수익 배수(창 끝 가치/매수가): 평균 {cm.call_mult.mean():.2f} · 중앙 {cm.call_mult.median():.2f} · "
              f"1 초과 비율 {(cm.call_mult > 1).mean():.0%} · 0.05 미만 비율 {(cm.call_mult < 0.05).mean():.0%} · 실제 외가격 평균 {cm.otm_real.mean():.2%}")
        G = sub.pivot_table(index=["field", "start"], columns="arm", values="score")
        grid = pd.DataFrame({a: (G[a] - G["base"]).groupby(level=0).mean() for a in G.columns if a != "base"}).T
        print("  강건성(점수차 평균, 행 = 예산_외가격):"); print(grid.round(2).to_string())

    a1 = all(verdict[("주", f)][0] for f in cs.FIELDS); a2 = all(verdict[("주", f)][1] for f in cs.FIELDS)
    a2_aux = all(verdict[("보", f)][1] for f in cs.FIELDS)
    print(f"\n판정: A1 {'통과' if a1 else '실패'} · A2 {'통과' if a2 else '실패'} · 보조 A2 부호 {'일치' if a2_aux else '반대'}")
    print("→", "C1 채택" if a1 and a2 else "기각 (E0 단독 유지)")
