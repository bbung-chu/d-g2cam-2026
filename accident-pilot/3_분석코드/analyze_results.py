#!/usr/bin/env python3
"""
조건별 결과 CSV(pilot_when_where.py 출력)를 모아 논문용 표·그림을 만든다.

    python analyze_results.py          # 기본: ../2_원자료 읽고 ../4_분석결과에 저장

출력:
  summary.csv / summary.md : 조건별 평균 점수, 원본 대비 하락률(RPD)과 95% 부트스트랩 신뢰구간
  degradation_curve.png    : 열화 단계별 성능 그래프 (When, Where)

하락률 RPD = (원본 평균 − 열화 평균) / 원본 평균 × 100 (%).
신뢰구간은 원본과 열화 결과가 모두 있는 영상만 짝지어, 영상 단위로 복원추출(기본 2,000회)해서 구한다.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = {"when_s1.0": "When (σt=1s)", "where": "Where (oracle)"}
KIND_ORDER = ["resize", "h264", "blur"]


def load(results):
    out = {}
    for f in sorted(Path(results).glob("molmo_*.csv")):
        m = re.fullmatch(r"molmo_(clean|(resize|h264|blur)_(.+))\.csv", f.name)
        if not m:
            continue  # mock 결과 등은 건너뜀
        kind, level = ("clean", "0") if m.group(1) == "clean" else (m.group(2), m.group(3))
        out[(kind, level)] = pd.read_csv(f).drop_duplicates("path").set_index("path")
    return out


def rpd(clean, deg):
    return (clean.mean() - deg.mean()) / clean.mean() * 100


def bootstrap(clean, deg, n_boot, rng):
    idx = rng.integers(0, len(clean), size=(n_boot, len(clean)))
    c, d = clean.to_numpy()[idx].mean(1), deg.to_numpy()[idx].mean(1)
    vals = (c - d) / c * 100
    return np.percentile(vals, [2.5, 97.5])


def main():
    p = argparse.ArgumentParser()
    base = Path(__file__).resolve().parents[1]
    p.add_argument("--results", type=Path, default=base / "2_원자료", help="molmo_*.csv 폴더")
    p.add_argument("--out", type=Path, default=base / "4_분석결과")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    runs = load(args.results)
    if ("clean", "0") not in runs:
        raise SystemExit("molmo_clean.csv(원본 결과)가 없습니다.")
    clean = runs[("clean", "0")]

    rows = [{"degradation": "clean", "level": "-", "n": len(clean),
             **{m: clean[m].mean() for m in METRICS if m in clean}}]
    conds = sorted([k for k in runs if k[0] != "clean"],
                   key=lambda k: (KIND_ORDER.index(k[0]), float(k[1])))
    for kind, level in conds:
        deg = runs[(kind, level)]
        common = clean.index.intersection(deg.index)
        row = {"degradation": kind, "level": level, "n": len(common)}
        for m in METRICS:
            if m not in deg or m not in clean or len(common) == 0:
                continue
            c, d = clean.loc[common, m], deg.loc[common, m]
            lo, hi = bootstrap(c, d, args.n_boot, rng)
            row.update({m: d.mean(), f"{m}_clean_paired": c.mean(),
                        f"{m}_RPD%": rpd(c, d), f"{m}_RPD_CI_low": lo, f"{m}_RPD_CI_high": hi})
        rows.append(row)

    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "summary.csv", index=False)

    lines = ["| 열화 | 단계 | n | When σt=1 | When 하락률 % [95% CI] | Where | Where 하락률 % [95% CI] |",
             "|---|---|---|---|---|---|---|"]
    for r in df.to_dict("records"):

        def cell(m):
            if f"{m}_RPD%" not in r or pd.isna(r.get(f"{m}_RPD%")):
                return "-"
            return f"{r[f'{m}_RPD%']:.1f} [{r[f'{m}_RPD_CI_low']:.1f}, {r[f'{m}_RPD_CI_high']:.1f}]"
        get = lambda m: f"{r[m]:.3f}" if m in r and not pd.isna(r[m]) else "-"  # noqa: E731
        lines.append(f"| {r['degradation']} | {r['level']} | {r['n']} | {get('when_s1.0')} | "
                     f"{cell('when_s1.0')} | {get('where')} | {cell('where')} |")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    plot(df, args.out / "degradation_curve.png")


def plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(METRICS), figsize=(4 * len(METRICS), 3.2), sharey=False)
    clean = df[df.degradation == "clean"].iloc[0]
    for ax, (m, title) in zip(axes, METRICS.items()):
        if m not in df:
            continue
        for kind in KIND_ORDER:
            sub = df[df.degradation == kind]
            if sub.empty:
                continue
            ys = [clean[m]] + sub[m].tolist()
            ax.plot(range(len(ys)), ys, marker="o", label=kind)
        n_levels = max(len(df[df.degradation == k]) for k in KIND_ORDER)
        ax.set_xticks(range(n_levels + 1), ["clean"] + [f"L{i}" for i in range(1, n_levels + 1)])
        ax.set_title(title)
        ax.set_ylabel("score")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
