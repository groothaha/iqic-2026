"""저변동 쏠림 민감도 — 다른 참가자도 저변동을 하면 우리 저변동 포트의 수상 확률이 얼마나 줄어드나.

contest_sim.py 와 같은 창·지표·보고서 분포. 참가자 모형은 학생형에 'lowvol'(변동성 하위 60개 중 15~30개) 비중 s 를 섞음.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import contest_sim as cs

BASE = [(0.35, "popular"), (0.25, "chase"), (0.20, "random10"), (0.10, "etf"), (0.10, "lottery")]
SHARES = [0.0, 0.1, 0.25, 0.5]
CANDS = ["저변동 30종목", "저변동 10종목", "모멘텀 10종목", "SPY 100%"]
N = 300

if __name__ == "__main__":
    c, v, r, spy, qqq, rf, gold = cs.load()
    dates = c.index; W = cs.W
    starts = pd.Series(dates, index=dates).groupby(dates.to_period("M")).first()
    starts = [d for d in starts if d >= pd.Timestamp("2006-01-01") and dates.get_loc(d) + W < len(dates)]
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
        ctx = dict(univ=U, dv_rank_top50=np.argsort(-dv.values)[:50],
                   mom_top=np.argsort(-mom3.values)[:max(20, len(U) // 10)],
                   mom12_top=np.argsort(-mom12.values), lowvol_top=np.argsort(Rh.std().values),
                   hibeta_top=np.argsort(-beta.values),
                   spy_up=bool(spx.iloc[i - 1] > spx.iloc[i - 200:i].mean()))
        Rs = r.iloc[i:i + W][U].values
        m = spy.iloc[i:i + W].values; q = qqq.iloc[i:i + W].values
        f = rf.iloc[i:i + W].values; g = gold.iloc[i:i + W].fillna(0).values
        cands = cs.candidates(ctx)
        for s in SHARES:
            mix = [(p * (1 - s), k) for p, k in BASE] + ([(s, "lowvol")] if s > 0 else [])
            kinds = cs.rng.choice(len(mix), N, p=[p for p, _ in mix])
            P = np.column_stack([cs.port_returns(*cs.field_weights(mix[k][1], ctx), Rs, m, q, f, g) for k in kinds])
            rep = np.clip(cs.rng.normal(45, 13, N), 10, 80) + np.clip(cs.rng.normal(15, 3, N), 0, 20)
            for name in CANDS:
                x = cs.port_returns(*cands[name], Rs, m, q, f, g)
                sc = cs.score(cs.metrics(np.column_stack([P, x]), m, f))
                rank = (sc > sc[-1]).sum() + 1
                rows.append(dict(share=s, cand=name, score=sc[-1], top3=rank <= 3,
                                 **{f"win_R{R0}": ((sc[:-1] + rep) > (sc[-1] + R0 + 20)).sum() < 3 for R0 in (60, 70, 75)}))
    D = pd.DataFrame(rows)
    D.to_parquet(cs.H / "iqic/out/crowding_sim.parquet")
    pd.set_option("display.width", 200)
    for col in ["score", "top3", "win_R60", "win_R70", "win_R75"]:
        print(f"\n--- {col} (열 = 저변동 참가자 비중) ---")
        print(D.pivot_table(index="cand", columns="share", values=col, aggfunc="mean").round(3).to_string())
