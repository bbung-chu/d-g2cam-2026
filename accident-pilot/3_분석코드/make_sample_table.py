"""50개 표본 특성과 전체 2,027개 비교 (이슈 #2, 지적 9·11)

출력: ../4_분석결과/표본_특성.md (../2_원자료의 metadata-real.csv, molmo_clean.csv를 읽음)
범주형(사고 유형, 장면, 낮/밤, 날씨, 화질, 해상도 구간)은 비율 비교 + 카이제곱 적합도 검정,
연속형(영상 길이, 사고 시점)은 중앙값·사분위 비교 + Mann-Whitney U 검정.
실행: python make_sample_table.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RAW = Path(__file__).resolve().parents[1] / "2_원자료"
OUT = Path(__file__).resolve().parents[1] / "4_분석결과"
full = pd.read_csv(RAW / "metadata-real.csv")
paths = pd.read_csv(RAW / "molmo_clean.csv")["path"]
samp = full[full.path.isin(paths)].copy()
assert len(samp) == 50

for df in (full, samp):
    df["해상도"] = pd.cut(df["height"], [0, 480, 720, 10_000], labels=["480p 이하", "481~720p", "720p 초과"])

CAT = [("type", "사고 유형"), ("scene_layout", "장면"), ("day_time", "낮/밤"),
       ("weather", "날씨"), ("quality", "화질"), ("해상도", "해상도")]
CONT = [("duration", "영상 길이(초)"), ("accident_time", "사고 시점(초)")]

lines = ["# 50개 표본 특성과 전체 실제 영상(2,027개) 비교 (make_sample_table.py 생성)", "",
         "추출: 실제 영상 2,027개 전체를 seed 0으로 섞은 뒤 앞에서 50개 (`pilot_when_where.py:223`). "
         f"ACCIDENT 분할 기준 test {int((samp.split_in_distribution == 'test').sum())}개, "
         f"train {int((samp.split_in_distribution == 'train').sum())}개.", ""]
summary = []
for col, name in CAT:
    pf = full[col].astype(str).value_counts(normalize=True)
    cs = samp[col].astype(str).value_counts()
    cats = pf.index.tolist()
    obs = np.array([cs.get(c, 0) for c in cats])
    # 기대 빈도 5 미만 범주는 "기타"로 합쳐 검정
    exp = pf.values * 50
    keep = exp >= 5
    o2 = np.append(obs[keep], obs[~keep].sum()) if (~keep).any() else obs
    e2 = np.append(exp[keep], exp[~keep].sum()) if (~keep).any() else exp
    o2, e2 = o2[e2 > 0], e2[e2 > 0]
    p = stats.chisquare(o2, e2 * o2.sum() / e2.sum()).pvalue if len(o2) > 1 else float("nan")
    summary.append((name, "범주 비율", f"p = {p:.2f}"))
    lines += [f"## {name} (카이제곱 적합도 p = {p:.2f})", "", "| 값 | 전체 2,027 | 표본 50 |", "|---|---|---|"]
    for c in cats:
        lines.append(f"| {c} | {pf[c] * 100:.1f}% | {cs.get(c, 0)}개 ({cs.get(c, 0) * 2:.0f}%) |")
    lines.append("")
for col, name in CONT:
    f, s = full[col], samp[col]
    p = stats.mannwhitneyu(f, s).pvalue
    summary.append((name, "중앙값 [사분위]",
                    f"전체 {f.median():.1f} [{f.quantile(.25):.1f}, {f.quantile(.75):.1f}] / "
                    f"표본 {s.median():.1f} [{s.quantile(.25):.1f}, {s.quantile(.75):.1f}], p = {p:.2f}"))
    lines += [f"## {name} (Mann-Whitney U p = {p:.2f})", "",
              "| | 전체 2,027 | 표본 50 |", "|---|---|---|",
              f"| 중앙값 | {f.median():.1f} | {s.median():.1f} |",
              f"| 사분위 | {f.quantile(.25):.1f} ~ {f.quantile(.75):.1f} | {s.quantile(.25):.1f} ~ {s.quantile(.75):.1f} |", ""]

lines = lines[:4] + ["## 요약 (논문용)", "", "| 항목 | 비교 | 결과 |", "|---|---|---|"] + \
    [f"| {a} | {b} | {c} |" for a, b, c in summary] + [""] + lines[4:]
(OUT / "표본_특성.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[:14]))
