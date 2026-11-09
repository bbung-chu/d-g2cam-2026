"""점수 변화의 원인 분석: 모델 응답 변동성인지, 특정 영상에 몰리는지, 영상 특성과 관련 있는지 (지적 33·50·51)

1. 응답 변동성: 원본 조건을 같은 환경(A100, 공식 설정)에서 다시 실행했을 때 예측이 같은지
   (../5_이전결과_1102/molmo_clean.csv 중 원래 A100에서 실행된 21개와 ../2_원자료/molmo_clean.csv 비교)
2. 영상 집중도: 시점 점수 T가 0.1 이상 바뀐 (영상, 조건) 이 몇 개 영상에 몰리는지
3. 영상 특성: 6개 특성(사고 유형, 장면, 낮/밤, 날씨, 화질, 해상도)별로 점수 변화(열화 − 원본)가 다른지
   Kruskal-Wallis 검정, 9개 조건 × 2개 점수(T, S) × 6개 특성 = 108개 검정
4. 다중 비교: 18개 RPD 비교(9개 조건 × T·S)에 본페로니 보정을 하면 결론이 바뀌는지
   짝지은 부트스트랩 200,000회(seed 0), 보정 구간 = (1 − 0.05/18) 백분위 구간
출력: ../4_분석결과/점수변화_원인.md
실행: python make_variability.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parents[1]
RAW, OUT, OLD = BASE / "2_원자료", BASE / "4_분석결과", BASE / "5_이전결과_1102"
TAGS = ["resize_2", "resize_4", "resize_8", "h264_37", "h264_42", "h264_47", "blur_6", "blur_11", "blur_17"]
CHAR = [("type", "사고 유형"), ("scene_layout", "장면"), ("day_time", "낮/밤"),
        ("weather", "날씨"), ("quality", "화질"), ("해상도", "해상도")]

clean = pd.read_csv(RAW / "molmo_clean.csv").set_index("path")
runs = {t: pd.read_csv(RAW / f"molmo_{t}.csv").set_index("path").loc[clean.index] for t in TAGS}
lines = ["# 점수 변화의 원인 분석 (make_variability.py 생성)", ""]

# 1. 응답 변동성
old = pd.read_csv(OLD / "molmo_clean.csv")
fast = old.iloc[list(range(20)) + [49]].set_index("path")      # 원래 A100에서 실행된 21개 (처리 시간 기준)
new = clean.loc[fast.index]
same_t = (fast["pred_frame"].fillna(-1) == new["pred_frame"].fillna(-1))
same_s = (fast["pred_x"] - new["pred_x"]).abs().lt(1e-9) & (fast["pred_y"] - new["pred_y"]).abs().lt(1e-9)
lines += ["## 1. 같은 환경에서 다시 실행했을 때 예측이 같은가", "",
          f"- 원래 A100에서 실행된 원본 {len(fast)}개: 시점 예측 프레임 같음 {int(same_t.sum())}개, "
          f"위치 예측 좌표 같음 {int(same_s.sum())}개", ""]

# 2. 영상 집중도
dT = pd.DataFrame({t: runs[t]["when_s1.0"] - clean["when_s1.0"] for t in TAGS})
big = dT.abs() >= 0.1
n_cond = big.sum(axis=1)
lines += ["## 2. 시점 점수 T가 0.1 이상 바뀐 영상", "",
          f"- 9개 조건 중 하나 이상에서 바뀐 영상: {int((n_cond >= 1).sum())}개 / 50개",
          f"- 그중 5개 이상 조건에서 바뀐 영상: {int((n_cond >= 5).sum())}개",
          f"- 한 번도 0.1 이상 바뀌지 않은 영상: {int((n_cond == 0).sum())}개", "",
          "| 바뀐 조건 수 | 영상 수 |", "|---|---|"]
lines += [f"| {k} | {int(v)} |" for k, v in n_cond.value_counts().sort_index().items()]
lines.append("")

# 3. 영상 특성
meta = pd.read_csv(RAW / "metadata-real.csv").set_index("path").loc[clean.index]
meta["해상도"] = pd.cut(meta["height"], [0, 480, 720, 10_000], labels=["480p 이하", "481~720p", "720p 초과"])
rows = []
for t in TAGS:
    for metric, nm in [("when_s1.0", "T"), ("where", "S")]:
        d = runs[t][metric] - clean[metric]
        for col, cname in CHAR:
            groups = [d[meta[col] == g].to_numpy() for g in meta[col].dropna().unique()]
            groups = [g for g in groups if len(g) >= 2]
            if len(groups) < 2 or np.ptp(np.concatenate(groups)) == 0:
                p = 1.0
            else:
                p = stats.kruskal(*groups).pvalue
            rows.append({"조건": t, "점수": nm, "특성": cname, "p": round(p, 3)})
res = pd.DataFrame(rows)
n_sig = int((res.p < 0.05).sum())
lines += ["## 3. 영상 특성별 점수 변화 차이 (Kruskal-Wallis)", "",
          f"- 검정 수: {len(res)}개 (9개 조건 × T·S × 6개 특성), 각 특성에서 영상 2개 이상인 값만 비교",
          f"- p < 0.05: {n_sig}개 (우연으로 기대되는 수 약 {len(res) * 0.05:.1f}개)", "",
          "| 조건 | 점수 | 특성 | p |", "|---|---|---|---|"]
lines += [f"| {r.조건} | {r.점수} | {r.특성} | {r.p} |" for r in res[res.p < 0.05].itertuples()]
lines.append("")
# 4. 다중 비교 (본페로니)
B, ALPHA = 200_000, 0.05 / 18
rng = np.random.default_rng(0)
idx = rng.integers(0, len(clean), size=(B, len(clean)))
mc = []
for t in TAGS:
    for metric, nm in [("when_s1.0", "T"), ("where", "S")]:
        x, y = clean[metric].to_numpy(), runs[t][metric].to_numpy()
        r = (x[idx].mean(1) - y[idx].mean(1)) / x[idx].mean(1) * 100
        lo, hi = np.percentile(r, [2.5, 97.5])
        blo, bhi = np.percentile(r, [ALPHA / 2 * 100, 100 - ALPHA / 2 * 100])
        mc.append({"조건": t, "점수": nm, "RPD": round((x.mean() - y.mean()) / x.mean() * 100, 1),
                   "95% CI": f"[{lo:.1f}, {hi:.1f}]", "본페로니 CI": f"[{blo:.2f}, {bhi:.1f}]",
                   "0 제외(보정 전)": lo > 0 or hi < 0, "0 제외(보정 후)": blo > 0 or bhi < 0})
mc = pd.DataFrame(mc)
lines += ["## 4. 다중 비교 보정 (본페로니, 18개 비교)", "",
          f"- 보정 전 0을 포함하지 않는 비교: {int(mc['0 제외(보정 전)'].sum())}개, 보정 후: {int(mc['0 제외(보정 후)'].sum())}개", "",
          "| 조건 | 점수 | RPD % | 95% CI | 본페로니 CI |", "|---|---|---|---|---|"]
lines += [f"| {r.조건} | {r.점수} | {r.RPD} | {r._4} | {r._5} |" for r in mc.itertuples()]
lines.append("")
(OUT / "점수변화_원인.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
