"""내부자 기회적 매수 신호 2015+ 재현 — PREREG.md 의 T1~T8 그대로.

입력: ~/data/us/sec_insider/ps/*.parquet (fetch_insider.py)
      ~/fundamental_ml/data/{prices_m,features,chars}.parquet
출력: ~/iqic/out/{monthly.parquet, summary.csv}
"""
import io, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm

H = Path.home()
PS = H / "data/us/sec_insider/ps"
FM = H / "fundamental_ml/data"
OUT = H / "iqic/out"


# ---------------------------------------------------------------- 1. 거래 정제
def load_trades():
    d = pd.concat([pd.read_parquet(p) for p in sorted(PS.glob("*.parquet"))], ignore_index=True)
    n0 = len(d)
    d = d[d.DOCUMENT_TYPE == "4"]
    d = d[((d.TRANS_CODE == "P") & (d.TRANS_ACQUIRED_DISP_CD == "A")) |
          ((d.TRANS_CODE == "S") & (d.TRANS_ACQUIRED_DISP_CD == "D"))]
    d = d[(d.TRANS_SHARES > 0) & (d.TRANS_PRICEPERSHARE > 0)]
    d = d.dropna(subset=["FILING_DATE", "TRANS_DATE", "RPTOWNERCIK", "ISSUERCIK"])
    # 같은 공시가 분기 파일 두 곳에 들어가는 경우 대비
    d = d.drop_duplicates(["ACCESSION_NUMBER", "RPTOWNERCIK", "TRANS_DATE", "TRANS_CODE",
                           "TRANS_SHARES", "TRANS_PRICEPERSHARE"])
    d["owner"] = d.RPTOWNERCIK.astype(int)
    d["cik"] = d.ISSUERCIK.astype(int)
    d["ty"] = d.TRANS_DATE.dt.year
    d["tm"] = d.TRANS_DATE.dt.month
    print(f"거래 {n0:,} → 정제 {len(d):,} (P {(d.TRANS_CODE=='P').sum():,} / S {(d.TRANS_CODE=='S').sum():,})")
    return d


# ---------------------------------------------------------------- 2. CMP 분류
def classify(d):
    """내부자×연도 Y: 직전 3년 모두 거래 있음 → 대상. 그중 3년 모두 같은 달 거래 → routine, 아니면 opp."""
    ym = d.groupby(["owner", "ty"]).tm.agg(lambda s: frozenset(s)).to_dict()
    lab = {}
    for (o, y) in d[["owner", "ty"]].drop_duplicates().itertuples(index=False):
        h = [ym.get((o, y - k)) for k in (1, 2, 3)]
        if any(x is None for x in h):
            continue
        lab[(o, y)] = "routine" if (h[0] & h[1] & h[2]) else "opp"
    d["cls"] = [lab.get(k, "na") for k in zip(d.owner, d.ty)]
    print("분류:", d.groupby(["TRANS_CODE", "cls"]).size().to_dict())
    return d


# ---------------------------------------------------------------- 3. 가격·시총
def load_market():
    px = pd.read_parquet(FM / "prices_m.parquet")
    px = px[px.index < "2026-07-01"]                      # 7월은 수집 시점 부분월
    ret = px.pct_change(fill_method=None)
    ret = ret.where(ret.abs() <= 1.0)
    f = pd.read_parquet(FM / "features.parquet", columns=["cik", "ticker", "filed"])
    f = f.dropna(subset=["ticker"]).sort_values("filed")
    cmap = f.groupby("cik").ticker.last()
    cmap = cmap[cmap.isin(px.columns)]
    mc = pit_mktcap(px)
    return px, ret, cmap, mc


def pit_mktcap(px):
    """PIT 시총 = 수정주가 × (그 시점 최신 공시 주식수 × 공시일 이후 분할 누적배율).
    chars.parquet 의 mktcap 은 분할 보정 없이 '분할수정 주가 × 당시 주식수'라 쓰면 안 됨
    (2016-06 CMCT $5.2조·GE $1.2조 → 유니버스 VW 연 −44%). 배당 보정분(수정주가 < 실제가)은 남음."""
    q = pd.read_parquet(FM / "features_q.parquet", columns=["ticker", "filed", "shares"])
    q = q.dropna(subset=["ticker", "shares"]).query("shares > 0").copy()
    q["filed"] = pd.to_datetime(q.filed)
    sp = pd.read_parquet(OUT / "splits.parquet")
    fac = np.ones(len(q))
    spg = {t: g.sort_values("date") for t, g in sp.groupby("ticker")}
    for t, idx in q.groupby("ticker").groups.items():
        g = spg.get(t)
        if g is None:
            continue
        d = g.date.values; r = g.ratio.values
        tail = np.append(np.cumprod(r[::-1])[::-1], 1.0)          # tail[i] = prod(r[i:])
        pos = np.searchsorted(d, q.loc[idx, "filed"].values, side="right")
        fac[q.index.get_indexer(idx)] = tail[pos]
    q["sh_adj"] = q.shares.values * fac
    L = px.stack().rename("price").reset_index()
    L.columns = ["date", "ticker", "price"]
    L["known"] = L.date + pd.offsets.MonthEnd(0)                  # 월 라벨 → 그 달 말까지 공시된 것
    L = pd.merge_asof(L.sort_values("known"), q[["ticker", "filed", "sh_adj"]].sort_values("filed"),
                      left_on="known", right_on="filed", by="ticker", direction="backward")
    L = L[(L.known - L.filed).dt.days <= 460]                     # 15개월 넘게 낡은 주식수는 버림
    mc = (L.price * L.sh_adj).groupby([L.date, L.ticker]).first().unstack()
    mc = mc.reindex(index=px.index, columns=px.columns)
    mc = mc.where((mc >= 1e7) & (mc <= 6e12))
    med = mc.expanding(min_periods=6).median().shift(1)           # 자기 과거 중앙값 10배 이탈 = 태그 오류
    ratio = mc / med
    return mc.where(med.isna() | ((ratio >= 0.1) & (ratio <= 10)))


def spy_monthly():
    import yfinance as yf
    s = yf.download("SPY", start="2011-12-01", end="2026-07-01", interval="1mo",
                    auto_adjust=True, progress=False)["Close"].squeeze()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.pct_change().rename("spy")


def ff_factors():
    base = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    def get(name, skip):
        z = zipfile.ZipFile(io.BytesIO(requests.get(base + name, timeout=60).content))
        txt = z.read(z.namelist()[0]).decode("latin1").splitlines()
        rows = []
        for line in txt[skip:]:
            p = [x.strip() for x in line.split(",")]
            if len(p) < 2 or not p[0].isdigit() or len(p[0]) != 6:
                if rows:
                    break
                continue
            rows.append(p)
        return rows
    r5 = get("F-F_Research_Data_5_Factors_2x3_CSV.zip", 0)
    f5 = pd.DataFrame([x[1:7] for x in r5], index=[x[0] for x in r5],
                      columns=["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]).astype(float) / 100
    rm = get("F-F_Momentum_Factor_CSV.zip", 0)
    mom = pd.DataFrame([x[1:2] for x in rm], index=[x[0] for x in rm], columns=["mom"]).astype(float) / 100
    f = f5.join(mom, how="inner")
    f.index = pd.to_datetime(f.index, format="%Y%m")
    return f


# ---------------------------------------------------------------- 4. 포트폴리오
def build(d, ret, px, cmap, mc):
    d = d[d.cik.isin(cmap.index)].copy()
    d["ticker"] = d.cik.map(cmap)
    d["fm"] = d.FILING_DATE.dt.to_period("M").dt.to_timestamp()    # 공시월 = 형성월
    print(f"매핑된 거래 {len(d):,} · 종목 {d.ticker.nunique():,}")
    sets = {}
    for name, q in {"opp_buy": "TRANS_CODE=='P' and cls=='opp'",
                    "opp_sell": "TRANS_CODE=='S' and cls=='opp'",
                    "rout_buy": "TRANS_CODE=='P' and cls=='routine'",
                    "all_buy": "TRANS_CODE=='P'"}.items():
        sets[name] = d.query(q).groupby("fm").ticker.agg(lambda s: set(s)).to_dict()

    rows = []
    months = [m for m in ret.index if pd.Timestamp("2012-02-01") <= m <= pd.Timestamp("2026-06-01")]
    for m in months:
        f = m - pd.DateOffset(months=1)                  # 형성월
        r = ret.loc[m]
        p0 = px.loc[f] if f in px.index else None
        if p0 is None:
            continue
        u = r.index[r.notna() & (p0 >= 5)]
        uset = set(u)
        cap = mc.loc[f].reindex(u) if f in mc.index else pd.Series(np.nan, index=u)
        big = set(cap.index[cap >= cap.median()]) if cap.notna().sum() > 50 else set()

        def ew(s):
            s = [t for t in s if t in uset]
            return (r[s].mean() if s else np.nan), len(s)

        def vw(s):
            s = [t for t in s if t in cap.index and np.isfinite(cap.get(t, np.nan))]
            if not s:
                return np.nan
            w = cap[s] / cap[s].sum()
            return float((r[s] * w).sum())

        row = {"month": m, "u_ew": r[u].mean(), "u_vw": vw(list(u)), "n_u": len(u)}
        for k in sets:
            s = sets[k].get(f, set())
            row[f"{k}_ew"], row[f"n_{k}"] = ew(s)
            row[f"{k}_vw"] = vw([t for t in s if t in uset])
        s = sets["opp_buy"].get(f, set()) & big
        row["opp_buy_big_ew"], row["n_opp_buy_big"] = ew(s)
        row["u_big_ew"] = r[list(big)].mean() if big else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("month")


# ---------------------------------------------------------------- 5. 통계
def nw_t(x, lags=3):
    x = pd.Series(x).dropna()
    m = sm.OLS(x.values, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(m.params[0]), float(m.tvalues[0]), len(x)


def summarize(M, spy, ff):
    import sys
    sys.path.insert(0, str(H / "AAAAAAbacktest/tier_a"))
    from btlab.metrics import deflated_sharpe

    series = {
        "T1 opp_buy EW - U EW": M.opp_buy_ew - M.u_ew,
        "T2 opp_buy VW - U VW": M.opp_buy_vw - M.u_vw,
        "T3 opp_buy - opp_sell EW": M.opp_buy_ew - M.opp_sell_ew,
        "T4 opp_buy - opp_sell VW": M.opp_buy_vw - M.opp_sell_vw,
        "T5 rout_buy EW - U EW": M.rout_buy_ew - M.u_ew,
        "T6 all_buy EW - U EW": M.all_buy_ew - M.u_ew,
        "T7 opp_buy big EW - U big EW": M.opp_buy_big_ew - M.u_big_ew,
    }
    out = []
    for per, (a, b) in {"2015-2026": ("2015-01", "2026-06"), "2015-2019": ("2015-01", "2019-12"),
                        "2020-2026": ("2020-01", "2026-06"), "2012-2014": ("2012-01", "2014-12")}.items():
        for k, s in series.items():
            x = s.loc[a:b].dropna()
            if len(x) < 12:
                continue
            mu, t, n = nw_t(x)
            out.append(dict(test=k, period=per, months=n, mean_bp=mu * 1e4, nw_t=t,
                            ann_sr=x.mean() / x.std() * np.sqrt(12),
                            dsr8=deflated_sharpe(x.values, n_trials=8, ppy=12) if per == "2015-2026" else np.nan))
    S = pd.DataFrame(out)

    # T8: SPY CAPM 알파, 그리고 FF5+MOM 알파 (T1 롱 포트, 무위험 초과)
    x = M.loc["2015-01":"2026-06"].join(spy, how="inner").join(ff, how="inner")
    extra = {}
    for nm, y in {"opp_buy_ew": x.opp_buy_ew, "u_ew": x.u_ew, "opp_buy_big_ew": x.opp_buy_big_ew}.items():
        ye = y - x.rf
        c = sm.OLS(ye, sm.add_constant(x.spy - x.rf), missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        f6 = sm.OLS(ye, sm.add_constant(x[["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]]),
                    missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        extra[nm] = dict(capm_alpha_bp=c.params["const"] * 1e4, capm_t=c.tvalues["const"],
                         beta=c.params.iloc[1], ff6_alpha_bp=f6.params["const"] * 1e4, ff6_t=f6.tvalues["const"],
                         **{f"b_{k}": f6.params[k] for k in ["smb", "hml", "rmw", "cma", "mom"]})
    # T1 초과(롱−유니버스)의 FF6 알파
    ex = (x.opp_buy_ew - x.u_ew)
    f6 = sm.OLS(ex, sm.add_constant(x[["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]]),
                missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    extra["T1_excess"] = dict(ff6_alpha_bp=f6.params["const"] * 1e4, ff6_t=f6.tvalues["const"],
                              **{f"b_{k}": f6.params[k] for k in ["smb", "hml", "rmw", "cma", "mom"]})
    return S, pd.DataFrame(extra).T


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    d = classify(load_trades())
    px, ret, cmap, mc = load_market()
    M = build(d, ret, px, cmap, mc)
    M.to_parquet(OUT / "monthly.parquet")
    spy, ff = spy_monthly(), ff_factors()
    S, E = summarize(M, spy, ff)
    S.to_csv(OUT / "summary.csv", index=False); E.to_csv(OUT / "alphas.csv")
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    print(S.round(3).to_string(index=False))
    print(E.round(3).to_string())
    n = M.loc["2015-01":"2026-06"]
    print("월평균 종목 수(2015+):", n[[c for c in n.columns if c.startswith("n_")]].mean().round(1).to_dict())
    print("최소:", n[[c for c in n.columns if c.startswith("n_")]].min().to_dict())
