"""그림 1: 베타별 대회 점수 — 우리 30종목(선물로 베타 조정) vs S&P 500 + 현금. 출처 out/beta_sweep.parquet."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = Path.home() / "iqic"
D = pd.read_parquet(H / "out/beta_sweep.parquet")
XS = (0.0, 0.10, 0.25, 0.40, 0.60, 0.80, 1.00, 1.20)
FIELDS = {"학생형": "Field A (student-like)", "숙련형": "Field B (experienced)"}
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e4e3df"

plt.rcParams.update({"font.family": "Helvetica", "font.size": 8.5, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.35), sharey=True)
for ax, (key, title) in zip(axes, FIELDS.items()):
    g = D[D.field == key].groupby("arm").score.mean()
    b = [g[f"B_{x:.2f}"] for x in XS]
    s = [g[f"S_{x:.2f}"] for x in XS[1:]]
    ax.plot(XS, b, color=BLUE, lw=2, marker="o", ms=4.5, mec="white", mew=1, zorder=3)
    ax.plot(XS[1:], s, color=ORANGE, lw=2, ls=(0, (4, 2)), marker="s", ms=4.2, mec="white", mew=1, zorder=3)
    base = g["BASE"]
    ax.plot([0.59], [base], marker="D", ms=6.5, color=INK, mec="white", mew=1.2, zorder=4, ls="none")
    ax.annotate("stocks alone,\nno futures", (0.59, base), xytext=(0.30, base + 1.8), fontsize=7.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    gap = g["B_0.60"] - g["S_0.60"]
    ax.annotate("", xy=(0.66, g["S_0.60"]), xytext=(0.66, g["B_0.60"]),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.8, shrinkA=1, shrinkB=1))
    ax.text(0.69, (g["S_0.60"] + g["B_0.60"]) / 2, f"+{gap:.1f}", fontsize=7.5, color=INK, va="center")
    ax.text(1.21, b[-1] + 0.5, "30 stocks\n+ S&P futures", fontsize=7.5, color=INK, ha="right", va="bottom")
    ax.text(1.21, s[-1] - 1.1, "S&P 500 + cash\n(zero alpha)", fontsize=7.5, color=INK, ha="right", va="top")
    ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    ax.set_xlabel("portfolio beta (target)")
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    ax.set_xlim(-0.05, 1.27); ax.set_ylim(46, 60)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("mean trading score (of 100)")
fig.tight_layout(w_pad=1.2)
out = H / "report/fig1_beta_curve.svg"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
print(out)
