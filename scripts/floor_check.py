"""구조 확인 — 저변동의 대회 점수가 '가장 낮은 30개'를 정확히 골라서 나오나, 저변동 구간에 있기만 하면 나오나.

변동성 하위 K개 중 무작위 30개(창마다 5번 추첨)를 저변동 30종목과 비교. 전략 시도가 아니라 플라시보형 점검.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import contest_sim as cs

POOLS = [100, 200, 400, None]          # None = 전체 유니버스
DRAWS = 5
N = 300

if __name__ == "__main__":
    c, v, r, spy, qqq, rf, gold = cs.load()
    dates = c.index; W = cs.W
    starts = pd.Series(dates, index=dates).groupby(dates.to_period("M")).first()
    starts = [d for d in starts if d >= pd.Timestamp("2006-01-01") and dates.get_loc(d) + W < len(dates)]
    spx = (1 + spy.fillna(0)).cumprod()
    rows = []
    for fname, mix in cs.FIELDS.items():
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
            kinds = cs.rng.choice(len(mix), N, p=[p for p, _ in mix])
            P = np.column_stack([cs.port_returns(*cs.field_weights(mix[k][1], ctx), Rs, m, q, f, g) for k in kinds])
            rep = np.clip(cs.rng.normal(45, 13, N), 10, 80) + np.clip(cs.rng.normal(15, 3, N), 0, 20)
            n = len(U)
            ports = [("저변동 30종목(하위 30)", ctx["lowvol_top"][:30])]
            for K in POOLS:
                pool = ctx["lowvol_top"][:K] if K else np.arange(n)
                for _ in range(DRAWS):
                    ports.append((f"하위 {K} 중 무작위 30" if K else "전체 중 무작위 30", cs.rng.choice(pool, 30, replace=False)))
            for name, idx in ports:
                w = np.zeros(n); w[idx] = 1 / 30
                x = cs.port_returns(w, np.zeros(3), Rs, m, q, f, g)
                sc = cs.score(cs.metrics(np.column_stack([P, x]), m, f))
                rank = (sc > sc[-1]).sum() + 1
                rows.append(dict(field=fname, cand=name, score=sc[-1], bottom25=rank > (N + 1) * 0.75,
                                 win_R70=((sc[:-1] + rep) > (sc[-1] + 90)).sum() < 3,
                                 win_R75=((sc[:-1] + rep) > (sc[-1] + 95)).sum() < 3))
    D = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"유니버스 크기 중앙값 참고: 창 {D.groupby('field').size().iloc[0] // (1 + DRAWS * len(POOLS))}개")
    T = D.groupby(["field", "cand"])[["score", "bottom25", "win_R70", "win_R75"]].mean().round(3)
    print(T.to_string())
