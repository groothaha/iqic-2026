"""현재 설계(R_all) + 시장 노출 빼기 파생 오버레이 — PREREG_derivatives.md 검정.

F25·F40 = S&P 선물 매도(사전 베타 → 목표 0.25·0.40) · C50·C100 = SPY 2% 외가격 콜 매도(노셔널 50·100%, bid 체결·mid 평가).
F 는 2·5·8·11월 창 2006~2024, C 는 옵션 데이터 범위(2010~2023). 출력: out/deriv_test.parquet, 표준출력 요약·판정.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import contest_sim as cs
from risk_test import sic_sector, tstat, path, K, CAP_N, STOP, DD

N = 300
F_GRID = (0.0, 0.10, 0.25, 0.40)
C_GRID = [(nu, o) for nu in (0.5, 1.0) for o in (0.0, 0.02, 0.04)]
MAIN = {"F25": ("F", 0.25), "F40": ("F", 0.40), "C50": ("C", (0.5, 0.02)), "C100": ("C", (1.0, 0.02))}


def path_ov(R, f, m, H0=0.0, cn=0.0, cmid=None, prem=0.0):
    """R_all 경로(섹터 한도는 종목 선택에서) + 선물 매도 노셔널 H0 + 콜 매도 노셔널 비율 cn.
    cmid = 창 중 콜 중간 호가 / 진입일 SPY, prem = 진입 매도 호가 / 진입일 SPY. 낙폭 기준 발동 시 모든 노출 절반."""
    W_, k = R.shape
    val = np.full(k, 1 / k); px = np.ones(k); alive = np.ones(k, bool)
    cash = cn * prem; H = H0; c_n = cn; Vp = 1.0; out = np.zeros(W_)
    pend_stop = np.zeros(k, bool); pend_dd = False; dd_done = False
    for t in range(W_):
        val[alive] *= 1 + R[t, alive]; px *= 1 + R[t]; cash *= 1 + f[t]
        cash -= H * (m[t] - f[t]); H *= 1 + m[t]
        if pend_stop.any():
            cash += val[pend_stop].sum(); val[pend_stop] = 0; alive &= ~pend_stop; pend_stop[:] = False
        if pend_dd:
            cash += 0.5 * val[alive].sum(); val[alive] *= 0.5; H *= 0.5
            if c_n:
                cash -= 0.5 * c_n * cmid[t]; c_n *= 0.5
            pend_dd = False; dd_done = True
        V = val.sum() + cash - (c_n * cmid[t] if c_n else 0.0)
        out[t] = V / Vp - 1; Vp = V
        pend_stop |= alive & (px - 1 <= STOP) & ~pend_stop
        if not dd_done and V - 1 <= DD:
            pend_dd = True
    return out


def load_chain():
    fs = sorted((cs.H / "options_data/data/optionsdx/parquet").glob("spy_*.parquet"))
    out = []
    for fp in fs:
        d = pd.read_parquet(fp, columns=["date", "spot", "expiry", "strike", "c_bid", "c_ask"])
        d = d[(d.expiry - d.date).dt.days.le(45) & (d.strike >= d.spot * 0.99) & (d.strike <= d.spot * 1.08)]
        out.append(d)
    return pd.concat(out, ignore_index=True)


def short_call(ch, f_date, end_date, wdates, otm):
    """f_date 종가에 bid 로 판 콜: (창 중 mid / S0, bid / S0, 실제 외가격, 내가격 종료 여부). 없으면 None."""
    day = ch[ch.date == f_date]
    if day.empty:
        return None
    ex = day.expiry[day.expiry >= end_date]
    if ex.empty:
        return None
    e = ex.min(); S0 = day.spot.iloc[0]
    c = day[(day.expiry == e) & (day.strike >= S0 * (1 + otm)) & (day.c_bid > 0)].sort_values("strike")
    if c.empty:
        return None
    k = c.strike.iloc[0]; bid0 = c.c_bid.iloc[0]
    q = ch[(ch.expiry == e) & (ch.strike == k) & ch.date.isin(wdates)].set_index("date")
    mid = ((q.c_bid + q.c_ask) / 2).reindex(wdates).ffill()
    sp = q.spot.reindex(wdates).ffill()
    if mid.isna().any():
        return None
    return mid.values / S0, bid0 / S0, k / S0 - 1, bool(sp.iloc[-1] > k)


def real_beta(x, m, f):
    ex, mx = x - f, m - f
    return ((ex - ex.mean()) * (mx - mx.mean())).sum() / ((mx - mx.mean()) ** 2).sum()


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
    ch = load_chain(); ch = ch[ch.date.isin(dates)]
    print(f"체인 {len(ch):,}행 · {ch.date.min().date()} ~ {ch.date.max().date()}", flush=True)

    starts = pd.Series(dates, index=dates).groupby(dates.to_period("M")).first()
    starts = [d for d in starts if d >= pd.Timestamp("2006-01-01") and d.month in (2, 5, 8, 11)
              and dates.get_loc(d) + W < T]
    spx = (1 + spy.fillna(0)).cumprod()
    rows, info = [], []
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
        cnt, sel = {}, []
        for j in order:
            s = sector[U[j]]
            if cnt.get(s, 0) < CAP_N:
                sel.append(j); cnt[s] = cnt.get(s, 0) + 1
            if len(sel) == K:
                break
        sel = np.array(sel, dtype=int)
        Rsel = Rs[:, sel]
        bhat = float(beta.iloc[sel].mean())

        base = path_ov(Rsel, f, m)
        assert np.allclose(base, path(Rsel, f, stop=True, dd=True)[0])   # 오버레이 0 = R_all 경로 그대로
        ports = {"BASE": base}
        for tb in F_GRID:
            ports[f"F_{tb:.2f}"] = path_ov(Rsel, f, m, H0=max(0.0, bhat - tb))
        wdates = dates[i:i + W]
        c_ok = d >= pd.Timestamp("2010-01-01") and wdates[-1] <= ch.date.max()
        itm = None
        if c_ok:
            cps = {(nu, o): short_call(ch, dates[i - 1], wdates[-1], wdates, o) for nu, o in C_GRID}
            if all(cp is not None for cp in cps.values()):
                for (nu, o), (cm, pr, _, it) in cps.items():
                    ports[f"C_{nu:.1f}_{o:.2f}"] = path_ov(Rsel, f, m, cn=nu, cmid=cm, prem=pr)
                itm = cps[(0.5, 0.02)][3]
            else:
                c_ok = False
        mret = np.prod(1 + m) - 1
        info.append(dict(start=d, bhat=bhat, c_ok=c_ok, itm=itm, mret=mret))
        for fname, mix in cs.FIELDS.items():
            kinds = cs.rng.choice(len(mix), N, p=[p_ for p_, _ in mix])
            P = np.column_stack([cs.port_returns(*cs.field_weights(mix[k_][1], ctx), Rs, m, q, f, g) for k_ in kinds])
            repsc = np.clip(cs.rng.normal(45, 13, N), 10, 80) + np.clip(cs.rng.normal(15, 3, N), 0, 20)
            for arm, x in ports.items():
                mt = cs.metrics(np.column_stack([P, x]), m, f)
                sc = cs.score(mt)
                rank = (sc > sc[-1]).sum() + 1
                rows.append(dict(field=fname, start=d, arm=arm, c_ok=c_ok, score=sc[-1], bottom25=rank > (N + 1) * 0.75,
                                 win_R70=((sc[:-1] + repsc) > (sc[-1] + 90)).sum() < 3,
                                 win_R75=((sc[:-1] + repsc) > (sc[-1] + 95)).sum() < 3,
                                 ret=np.prod(1 + x) - 1, mret=mret, beta_real=real_beta(x, m, f), bhat=bhat,
                                 **{f"p_{k}": cs.pct_rank(mt[k])[-1] for k in ("sharpe", "alpha", "treynor", "ret")}))
    D = pd.DataFrame(rows); I = pd.DataFrame(info)
    D.to_parquet(cs.H / "iqic/out/deriv_test.parquet")
    pd.set_option("display.width", 220)
    print(f"\n창: F {I.start.nunique()}개 · C {int(I.c_ok.sum())}개 · 사전 베타 β̂ 중앙 {I.bhat.median():.2f} "
          f"(사분위 {I.bhat.quantile(.25):.2f}~{I.bhat.quantile(.75):.2f}) · C 콜 내가격 종료 {I[I.c_ok].itm.mean():.0%}")

    name = {"F25": "F_0.25", "F40": "F_0.40", "C50": "C_0.5_0.02", "C100": "C_1.0_0.02"}
    verdict = {}
    for key, arm in name.items():
        sub = D[D.c_ok] if key.startswith("C") else D
        print(f"\n######## {key} ({arm}) — 창 {sub.start.nunique()}개")
        for fname in cs.FIELDS:
            Df = sub[sub.field == fname]
            piv = Df.pivot(index="start", columns="arm", values="score")
            dd = piv[arm] - piv["BASE"]
            mr = Df[Df.arm == "BASE"].set_index("start").mret
            g_ = Df[Df.arm.isin(["BASE", arm])].groupby("arm")
            S = pd.DataFrame({"점수": g_.score.mean(), "수상70": g_.win_R70.mean(), "수상75": g_.win_R75.mean(),
                              "하위25%": g_.bottom25.mean(), "15일수익": g_.ret.mean(), "하위5%수익": g_.ret.quantile(0.05),
                              "최악창": g_.ret.min(), "실현β중앙": g_.beta_real.median(),
                              "β<0.1": g_.beta_real.apply(lambda b: (b < 0.1).mean()), "β<0": g_.beta_real.apply(lambda b: (b < 0).mean()),
                              "Sharpe백분위": g_.p_sharpe.mean(), "알파백분위": g_.p_alpha.mean(),
                              "Treynor백분위": g_.p_treynor.mean(), "수익백분위": g_.p_ret.mean()}).loc[["BASE", arm]]
            print(f"--- {fname}"); print(S.round(3).to_string())
            late = dd[dd.index.year >= (2017 if key.startswith("C") else 2015)]
            print(f"  {key}−BASE: 전체 {dd.mean():+.2f} (t {tstat(dd):+.2f}) · 상승장 {dd[mr > 0].mean():+.2f} (n {(mr > 0).sum()}) · "
                  f"하락장 {dd[mr <= 0].mean():+.2f} (n {(mr <= 0).sum()}) · 후기 {late.mean():+.2f}")
            b25 = S.loc[arm, "하위25%"] - S.loc["BASE", "하위25%"]
            verdict.setdefault(key, []).append((dd.mean(), dd.mean() >= 0 and b25 <= 0.05, b25))
    print("\n강건성 (점수차 평균, 학생형 / 숙련형):")
    for sub_, arms in [(D, [a for a in D.arm.unique() if a.startswith("F_")]), (D[D.c_ok], [a for a in D.arm.unique() if a.startswith("C_")])]:
        G = sub_.pivot_table(index=["field", "start"], columns="arm", values="score")
        for a in arms:
            x_ = (G[a] - G["BASE"]).groupby(level=0).mean()
            print(f"  {a:12s} {x_['학생형']:+.2f} / {x_['숙련형']:+.2f}")
    print("\n판정:")
    passed = {}
    for key, vs in verdict.items():
        ok = all(p for _, p, _ in vs)
        print(f"  {key}: 점수차 {vs[0][0]:+.2f} / {vs[1][0]:+.2f} · 하위25% 증가 {vs[0][2]:+.1%} / {vs[1][2]:+.1%} → {'통과' if ok else '실패'}")
        if ok:
            passed[key] = np.mean([x for x, _, _ in vs])
    print("→", f"{max(passed, key=passed.get)} 채택 (통과 {list(passed)})" if passed else "전부 실패 → 현재 설계(R_all) 유지")
