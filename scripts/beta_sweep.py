"""베타 곡선 — PREREG_beta.md 검정.

종목 선택(R_all) 고정, S&P 선물로 포트 베타만 x ∈ {0 … 1.2} 로 바꿔 대회 점수 곡선. 참고: S_x = x·SPY + (1−x)·무위험(알파 0).
출력: out/beta_sweep.parquet, 표준출력 곡선·판정.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import contest_sim as cs
from risk_test import sic_sector, tstat, K, CAP_N
from deriv_test import path_ov, real_beta

N = 300
XS = (0.0, 0.10, 0.25, 0.40, 0.60, 0.80, 1.00, 1.20)

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
        Rsel = Rs[:, np.array(sel, dtype=int)]
        bhat = float(beta.iloc[sel].mean())
        ports = {"BASE": path_ov(Rsel, f, m)}
        for x_ in XS:
            ports[f"B_{x_:.2f}"] = path_ov(Rsel, f, m, H0=bhat - x_)
            ports[f"S_{x_:.2f}"] = x_ * m + (1 - x_) * f
        mret = np.prod(1 + m) - 1
        for fname, mix in cs.FIELDS.items():
            kinds = cs.rng.choice(len(mix), N, p=[p_ for p_, _ in mix])
            P = np.column_stack([cs.port_returns(*cs.field_weights(mix[k_][1], ctx), Rs, m, q, f, g) for k_ in kinds])
            repsc = np.clip(cs.rng.normal(45, 13, N), 10, 80) + np.clip(cs.rng.normal(15, 3, N), 0, 20)
            for arm, x in ports.items():
                mt = cs.metrics(np.column_stack([P, x]), m, f)
                sc = cs.score(mt)
                rank = (sc > sc[-1]).sum() + 1
                rows.append(dict(field=fname, start=d, arm=arm, score=sc[-1], bottom25=rank > (N + 1) * 0.75,
                                 win_R70=((sc[:-1] + repsc) > (sc[-1] + 90)).sum() < 3,
                                 win_R75=((sc[:-1] + repsc) > (sc[-1] + 95)).sum() < 3,
                                 ret=np.prod(1 + x) - 1, mret=mret, beta_real=real_beta(x, m, f), bhat=bhat,
                                 **{f"p_{k}": cs.pct_rank(mt[k])[-1] for k in ("sharpe", "alpha", "treynor", "ret")}))
    D = pd.DataFrame(rows)
    D.to_parquet(cs.H / "iqic/out/beta_sweep.parquet")
    pd.set_option("display.width", 220)
    print(f"창 {D.start.nunique()}개 · 사전 베타 β̂ 중앙 {D.bhat.median():.2f}")
    arms = ["BASE"] + [f"B_{x:.2f}" for x in XS] + [f"S_{x:.2f}" for x in XS]
    rej = {}
    for fname in cs.FIELDS:
        Df = D[D.field == fname]
        g_ = Df.groupby("arm")
        up = Df[Df.mret > 0].groupby("arm").score.mean(); dn = Df[Df.mret <= 0].groupby("arm").score.mean()
        S = pd.DataFrame({"점수": g_.score.mean(), "상승장": up, "하락장": dn, "수상70": g_.win_R70.mean(), "수상75": g_.win_R75.mean(),
                          "하위25%": g_.bottom25.mean(), "실현β중앙": g_.beta_real.median(),
                          "Sharpe": g_.p_sharpe.mean(), "알파": g_.p_alpha.mean(), "Treynor": g_.p_treynor.mean(),
                          "수익": g_.p_ret.mean()}).loc[arms]
        piv = Df.pivot(index="start", columns="arm", values="score")
        S["−BASE"] = [(piv[a] - piv["BASE"]).mean() for a in arms]
        S["t"] = [tstat(piv[a] - piv["BASE"]) if a != "BASE" else np.nan for a in arms]
        S["2015–24"] = [(piv[a] - piv["BASE"])[piv.index.year >= 2015].mean() for a in arms]
        print(f"\n=== {fname}"); print(S.round(3).to_string())
        for x_ in (0.8, 1.0, 1.2):
            a = f"B_{x_:.2f}"
            rej.setdefault(a, []).append(S.loc[a, "−BASE"] >= 1.0 and S.loc[a, "t"] >= 1.0)
        b = S.loc[[f"B_{x:.2f}" for x in XS], "점수"]
        print(f"  B 곡선 최고점: β {float(b.idxmax()[2:]):.2f} ({b.max():.2f}점)")
    bad = [a for a, v_ in rej.items() if all(v_)]
    print("\n판정:", f"논지 기각 ({bad} 가 BASE 보다 ≥ +1점, t ≥ 1)" if bad else "논지 지지 (β 0.8·1.0·1.2 어느 것도 BASE 를 +1점 이상 넘지 못함)")
