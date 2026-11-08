"""표 4 축소판: 원본 점수, 열화 점수, 차이, RPD(95% CI)  (지적 31, 34)

RPD는 반올림 전 평균으로 계산한 값(summary.csv)을 그대로 쓴다.
실행: python make_table4.py  → ../4_분석결과/table4_compact.md (../4_분석결과/summary.csv를 읽음)
"""
from pathlib import Path

import pandas as pd

d = Path(__file__).resolve().parents[1] / "4_분석결과"
s = pd.read_csv(d / "summary.csv")
rows = ["| 열화 | 단계 | T 원본 | T 열화 | T 차이 | RPD_T % [95% CI] | S 원본 | S 열화 | S 차이 | RPD_S % [95% CI] |",
        "|---|---|---|---|---|---|---|---|---|---|"]
for _, x in s.iloc[1:].iterrows():
    ct, cs = x["when_s1.0_clean_paired"], x["where_clean_paired"]
    rows.append(
        f"| {x.degradation} | {x.level} | {ct:.3f} | {x['when_s1.0']:.3f} | {x['when_s1.0'] - ct:+.3f} "
        f"| {x['when_s1.0_RPD%']:.1f} [{x['when_s1.0_RPD_CI_low']:.1f}, {x['when_s1.0_RPD_CI_high']:.1f}] "
        f"| {cs:.3f} | {x['where']:.3f} | {x['where'] - cs:+.3f} "
        f"| {x['where_RPD%']:.1f} [{x['where_RPD_CI_low']:.1f}, {x['where_RPD_CI_high']:.1f}] |")
(d / "table4_compact.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
print("\n".join(rows))
