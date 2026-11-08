"""그림 3 열화 단계별 성능 변화 + 미검출 영향 (이슈 #1, 지적 29·30·33·37, T5·T7)

출력
  ../paper/figures/fig3_성능변화.png   (a)(b) 평균 ± 95% CI, (c)(d) 영상별 점수 변화 분포
  results_n50/미검출_영향.md            조건별 미검출 비율, 처리 방식별 T 평균, 영상별 증감 수

CI는 영상 50개를 복원 추출한 부트스트랩 2,000회(seed 0)의 2.5·97.5 백분위.
실행: python make_fig3.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results_n50"
FIG = HERE.parent / "paper" / "figures" / "fig3_성능변화.png"
N_BOOT, SEED = 2000, 0

# 열화 종류: 이름, 단계, 색(검증한 categorical 1~3번), 선 모양, 마커
KINDS = [("resize", ["2", "4", "8"], "해상도 저하", "#2a78d6", "-", "o", ["1/2", "1/4", "1/8"]),
         ("h264", ["37", "42", "47"], "H.264 압축", "#eb6834", "--", "s", ["37", "42", "47"]),
         ("blur", ["6", "11", "17"], "모션 블러", "#1baf7a", ":", "^", ["6", "11", "17"])]
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"


def load(tag):
    return pd.read_csv(RES / f"molmo_{tag}.csv").drop_duplicates("path").set_index("path")


def boot_ci(x, rng):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    return np.percentile(x[idx].mean(1), [2.5, 97.5])


clean = load("clean")
runs = {(k, l): load(f"{k}_{l}").loc[clean.index] for k, ls, *_ in KINDS for l in ls}

# ---------- 그림 ----------
from matplotlib import font_manager  # noqa: E402
_have = {f.name for f in font_manager.fontManager.ttflist}
KFONT = next((n for n in ("Malgun Gothic", "Apple SD Gothic Neo", "NanumGothic", "AppleGothic", "Noto Sans CJK KR")
              if n in _have), "DejaVu Sans")  # Windows·macOS 어디서나 한글이 나오게
plt.rcParams.update({"font.family": KFONT, "font.size": 7.0, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.unicode_minus": False})
# 논문에 들어가는 실제 크기(한 단 폭 3.35 in)로 그린다. HWP에는 100%로 넣는다.
fig, axes = plt.subplots(2, 2, figsize=(3.35, 3.45), dpi=600,
                         gridspec_kw=dict(height_ratios=[1, 1.05], hspace=0.55, wspace=0.34))
rng = np.random.default_rng(SEED)
for col, (metric, name) in enumerate([("when_s1.0", "시간 유사도 T"), ("where", "공간 유사도 S")]):
    # (a)(b) 평균 ± 95% CI
    ax = axes[0, col]
    c_mean = clean[metric].mean()
    c_lo, c_hi = boot_ci(clean[metric], rng)
    ax.axhspan(c_lo, c_hi, color="#f0f0ee", zorder=0)
    ax.axhline(c_mean, color=MUTED, lw=0.8, zorder=1)
    for i, (k, ls, label, color, style, marker, ticks) in enumerate(KINDS):
        off = (i - 1) * 0.13
        ys, los, his = [c_mean], [c_lo], [c_hi]
        for l in ls:
            v = runs[(k, l)][metric]
            lo, hi = boot_ci(v, rng)
            ys.append(v.mean()); los.append(lo); his.append(hi)
        xs = np.arange(4) + off
        ax.errorbar(xs[1:], ys[1:], yerr=[np.array(ys[1:]) - los[1:], np.array(his[1:]) - ys[1:]],
                    fmt="none", ecolor=color, elinewidth=0.7, capsize=1.5, zorder=2)
        ax.plot(xs, ys, color=color, ls=style, lw=1.1, marker=marker, ms=3.6,
                mec="white", mew=0.5, label=label, zorder=3)
    ax.set_xticks(range(4), ["원본", "1", "2", "3"])
    ax.set_xlabel("열화 단계", fontsize=6.8, labelpad=1.5)
    ax.set_title(f"({'ab'[col]}) {name} 평균", fontsize=7.2, color=INK, pad=3, weight="bold")
    ax.set_ylim(0.15, 0.85) if col == 0 else ax.set_ylim(0.40, 0.85)
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # (c)(d) 영상별 점수 변화(열화 - 원본) 분포
    ax = axes[1, col]
    data, colors, xt = [], [], []
    for k, ls, label, color, style, marker, ticks in KINDS:
        for l, t in zip(ls, ticks):
            data.append((runs[(k, l)][metric] - clean[metric]).to_numpy())
            colors.append(color); xt.append(str(ls.index(l) + 1))
    pos = np.arange(len(data)) + np.repeat([0, 0.5, 1.0], 3)
    ax.axhline(0, color=MUTED, lw=0.8)
    bp = ax.boxplot(data, positions=pos, widths=0.6, whis=(5, 95), showfliers=False,
                    patch_artist=True, medianprops=dict(color=INK, lw=0.9))
    for patch, c in zip(bp["boxes"], colors):
        patch.set(facecolor=c, alpha=0.35, edgecolor=c, lw=0.7)
    for part in ("whiskers", "caps"):
        for j, line in enumerate(bp[part]):
            line.set(color=colors[j // 2], lw=0.7)
    jr = np.random.default_rng(1)
    for p, d, c in zip(pos, data, colors):
        ax.scatter(p + jr.uniform(-0.2, 0.2, len(d)), d, s=1.2, color=c, alpha=0.55, lw=0, zorder=3)
        ax.scatter(p, d.mean(), s=7, marker="D", color="white", edgecolor=INK, lw=0.6, zorder=4)
    ax.set_xticks(pos, xt)
    ax.set_xlabel("열화 단계", fontsize=6.8, labelpad=1.5)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(f"({'cd'[col]}) 영상별 {name.split()[-1]} 변화", fontsize=7.2, color=INK, pad=3, weight="bold")
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

h, lab = axes[0, 0].get_legend_handles_labels()
fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, fontsize=7.0,
           bbox_to_anchor=(0.5, 1.01), handlelength=2.4, columnspacing=1.2)
# 막대·띠·상자 설명은 그림 안이 아니라 HWP 캡션에 쓴다 (글자를 키우기 위해).
fig.subplots_adjust(left=0.115, right=0.985, top=0.905, bottom=0.095)
fig.savefig(FIG, dpi=600)

# ---------- 미검출 영향 표 ----------
rows = []
for tag, df in [("원본", clean)] + [(f"{k} {l}", runs[(k, l)]) for k, ls, *_ in KINDS for l in ls]:
    miss = df["when_failed"].astype(str).str.lower() == "true"
    t = df["when_s1.0"]
    rows.append({"조건": tag, "미검출 수": int(miss.sum()), "미검출 %": round(miss.mean() * 100),
                 "T 공식(대체값)": round(t.mean(), 3),
                 "T 미검출=0": round(t.where(~miss, 0).mean(), 3),
                 "T 미검출 제외": round(t[~miss].mean(), 3),
                 "대체값 기여": round(t[miss].sum() / len(t), 3)})
miss_tab = pd.DataFrame(rows)

chg = []
for k, ls, *_ in KINDS:
    for l in ls:
        for metric, nm in [("when_s1.0", "T"), ("where", "S")]:
            d = runs[(k, l)][metric] - clean[metric]
            chg.append({"조건": f"{k} {l}", "점수": nm, "0.1 이상 상승": int((d >= 0.1).sum()),
                        "0.1 이상 하락": int((d <= -0.1).sum()), "변화 0.1 미만": int((d.abs() < 0.1).sum())})
chg_tab = pd.DataFrame(chg)


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


(RES / "미검출_영향.md").write_text(
    "# 미검출 영향과 영상별 점수 변화 (make_fig3.py 생성)\n\n"
    "## 1. 조건별 미검출과 처리 방식에 따른 시점 점수 T\n\n"
    "- **T 공식(대체값)**: 논문 값. 미검출 영상은 ACCIDENT 공식 코드대로 예측 시각을 `정답 시각÷2`로 채워 채점\n"
    "- **T 미검출=0**: 미검출 영상을 오답(0점)으로 처리했을 때\n"
    "- **T 미검출 제외**: 미검출 영상을 빼고 평균\n"
    "- **대체값 기여**: 대체값으로 받은 점수가 평균 T에 더한 양 (= 공식 − 미검출=0)\n\n"
    + md(miss_tab) +
    "\n\n## 2. 영상별 점수 변화 (열화 − 원본, 50개 중 영상 수)\n\n" + md(chg_tab) + "\n",
    encoding="utf-8")
print(FIG)
print(md(miss_tab))
print()
print(md(chg_tab))
