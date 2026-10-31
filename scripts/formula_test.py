"""채점 공식 민감도 — PREREG_formula.md 검정.

같은 창·참가자·포트에서 교과서 공식(T)과 StockTrak FAQ 문구 공식(ST, ST_a, ST_t)으로 채점.
출력: out/formula_test.parquet, 표준출력 표·경고 판정.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import contest_sim as cs
from risk_test import sic_sector, K, CAP_N
from deriv_test import path_ov

N = 300
RF_ST = 0.03 / 252
ARMS_C = ["모멘텀 10종목", "모멘텀 5종목 집중", "저변동 10종목", "저변동 5종목", "고베타 20종목"]


def metrics_st(R, m, f, alpha_st=True, treynor_st=True, sharpe_st=True):
    """StockTrak FAQ 문구 공식. 꺼진 항목은 교과서(cs.metrics) 값."""
    mt = cs.metrics(R, m, f)
    W_ = R.shape[0]
    sd = R.std(0, ddof=1)
    beta = ((R - R.mean(0)) * (m - m.mean())[:, None]).sum(0) / (W_ - 1) / m.var(ddof=1)
    b = np.where(np.abs(beta) < 0.05, np.sign(beta + 1e-12) * 0.05, beta)
    if sharpe_st:
        mt["sharpe"] = np.where(sd > 1e-9, (R.mean(0) - RF_ST) / np.maximum(sd, 1e-12), 0.0) * 252 ** 0.5
    if alpha_st:
        mt["alpha"] = R.mean(0) - (RF_ST + beta * (R.mean(0) - m.mean()))
    if treynor_st:
        mt["treynor"] = sd / b * 252 ** 0.5
    return mt


if __name__ == "__main__":
    c, v, r, spy, qqq, rf, gold = cs.load()
    dates = c.index; W = cs.W; T = len(dates)
    col = {t: j for j, t in enumerate(c.columns)}
    X = r.sub(spy, axis=0).values
    ea = pd.read_parquet(cs.H / "iqic/out/ea_dates.parquet")
    ea["pos"] = dates.searchsorted(ea.filing_date)
    ea = ea[(ea.pos >= 1) & (ea.pos <= T - 2)]
    ev = {t: np.sort(g.pos.unique()) for t, g in ea.groupby("ticker") if t in col}
    sic = pd.read_parquet(cs.H / "iqic/out/sic.parquet").set_index("ticker").sic
    sector = {t: sic_sector(sic.get(t)) for t in c.columns}

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
        cnt, sel = {}, []
        for j in order:
            s = sector[U[j]]
            if cnt.get(s, 0) < CAP_N:
                sel.append(j); cnt[s] = cnt.get(s, 0) + 1
            if len(sel) == K:
                break
        cand = cs.candidates(ctx)
        ports = {"BASE": path_ov(Rs[:, np.array(sel, dtype=int)], f, m),
                 "LV30": cs.port_returns(*cand["저변동 30종목"], Rs, m, q, f, g),
                 "S_0.6": 0.6 * m + 0.4 * f, "SPY": m.copy()}
        for a in ARMS_C:
            ports[a] = cs.port_returns(*cand[a], Rs, m, q, f, g)
        mret = np.prod(1 + m) - 1
        for fname, mix in cs.FIELDS.items():
            kinds = cs.rng.choice(len(mix), N, p=[p_ for p_, _ in mix])
            P = np.column_stack([cs.port_returns(*cs.field_weights(mix[k_][1], ctx), Rs, m, q, f, g) for k_ in kinds])
            repsc = np.clip(cs.rng.normal(45, 13, N), 10, 80) + np.clip(cs.rng.normal(15, 3, N), 0, 20)
            for arm, x in ports.items():
                Rall = np.column_stack([P, x])
                for form, kw in (("T", None), ("ST", dict()), ("ST_a", dict(treynor_st=False, sharpe_st=False)),
                                 ("ST_t", dict(alpha_st=False, sharpe_st=False))):
                    mt = cs.metrics(Rall, m, f) if kw is None else metrics_st(Rall, m, f, **kw)
                    sc = cs.score(mt)
                    rank = (sc > sc[-1]).sum() + 1
                    rows.append(dict(field=fname, start=d, arm=arm, form=form, score=sc[-1],
                                     bottom25=rank > (N + 1) * 0.75,
                                     win_R70=((sc[:-1] + repsc) > (sc[-1] + 90)).sum() < 3,
                                     win_R75=((sc[:-1] + repsc) > (sc[-1] + 95)).sum() < 3, mret=mret,
                                     **{f"p_{k}": cs.pct_rank(mt[k])[-1] for k in ("sharpe", "alpha", "treynor", "ret")}))
    D = pd.DataFrame(rows)
    D.to_parquet(cs.H / "iqic/out/formula_test.parquet")
    pd.set_option("display.width", 220)
    print(f"창 {D.start.nunique()}개")
    arms = ["BASE", "LV30", "S_0.6", "SPY"] + ARMS_C
    warn = {}
    for fname in cs.FIELDS:
        print(f"\n=== {fname}")
        for form in ("T", "ST", "ST_a", "ST_t"):
            Df = D[(D.field == fname) & (D.form == form)]
            g_ = Df.groupby("arm")
            S = pd.DataFrame({"점수": g_.score.mean(), "수상70": g_.win_R70.mean(), "수상75": g_.win_R75.mean(),
                              "하위25%": g_.bottom25.mean(), "Sharpe": g_.p_sharpe.mean(), "알파": g_.p_alpha.mean(),
                              "Treynor": g_.p_treynor.mean(), "수익": g_.p_ret.mean()}).loc[arms]
            S["순위"] = S.점수.rank(ascending=False).astype(int)
            print(f"-- 공식 {form}"); print(S.round(3).to_string())
            if form == "T":
                w70_T = S.loc["BASE", "수상70"]
            if form == "ST":
                warn[fname] = (S.loc["BASE", "순위"] > 3) or (S.loc["BASE", "수상70"] - w70_T <= -0.05)
    print("\n경고:", "걸림 → 사용자 결정 필요" if all(warn.values()) else "안 걸림 (두 모형 모두는 아님)", warn)
