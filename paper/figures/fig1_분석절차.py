"""그림 1 전체 분석 절차 (11/9 수정본).

11/2본 대비: 미검출·무응답 처리 경로 추가(지적 29, T2), 용어를 GLOSSARY.md에 맞춤(시점 판별 / 조건부 위치 추정).
11/9 재작성: 논문에 들어가는 실제 크기(한 단 폭 3.35 in)로 그림. 교수님 요청(화살표 정리, 인쇄 가독성)에 따라
  - 화살표는 모두 곧은 세로선, 같은 길이, 좌우 대칭
  - 글자는 검정·중간 굵기 이상, 선은 1pt 이상 (HWP 뷰어가 화면 크기로 줄여도 흐려지지 않게)
HWP에는 100% 크기로 넣는다.
실행: python fig1_분석절차.py  → 같은 폴더에 fig1_분석절차.png (600 dpi)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

_have = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next((n for n in ("Malgun Gothic", "Apple SD Gothic Neo", "NanumGothic", "AppleGothic")
                                    if n in _have), "DejaVu Sans")

W, H = 3.35, 3.07            # 한 단 폭 (inch), 실제 인쇄 크기
INK = "#000000"
MISS = "#b0261c"             # 미검출·무응답 경로
FT, FS = 7.6, 7.0            # 제목, 설명 글자 크기 (pt)
LW, ALW = 1.0, 1.15          # 상자 선, 화살표 굵기 (pt)
SUBW = "medium"              # 설명 글자 굵기

fig = plt.figure(figsize=(W, H), dpi=600)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def rbox(x, y, w, h, fc="#ffffff", ec=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.035",
                                fc=fc, ec=ec, lw=LW))


def box(x, y, w, h, title, sub=None, fc="#ffffff", ec=INK, color=INK, ft=FT, fs=FS):
    rbox(x, y, w, h, fc, ec)
    if sub:
        ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=ft, weight="bold", color=color)
        ax.text(x + w / 2, y + h * 0.29, sub, ha="center", va="center", fontsize=fs, weight=SUBW, color=color)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=ft, weight="bold", color=color)


def down(x, y_top, y_bot, color=INK):
    """곧은 세로 화살표 (y_top → y_bot)."""
    ax.annotate("", xy=(x, y_bot), xytext=(x, y_top),
                arrowprops=dict(arrowstyle="-|>", lw=ALW, color=color, mutation_scale=9, shrinkA=0, shrinkB=0))


m = 0.05
fw = W - 2 * m
G = 0.07                     # 상자 사이 화살표 길이 (모두 같게)

# ① 입력
h1 = 0.27
y1 = H - 0.03 - h1
box(m, y1, fw, h1, "① ACCIDENT 실제 CCTV 사고 영상 50개 (시드 0)", fc="#eaf1fb")
down(W / 2, y1, y1 - G)

# ② 열화
h2 = 0.58
y2 = y1 - G - h2
rbox(m, y2, fw, h2, fc="#fdf1e2")
ax.text(W / 2, y2 + h2 - 0.10, "② 열화 적용: 원본 + 3종 × 3단계 = 10조건",
        ha="center", va="center", fontsize=FT, weight="bold")
items = [("해상도 저하", "1/2, 1/4, 1/8"), ("H.264 압축", "CRF 37, 42, 47"), ("모션 블러", "6, 11, 17 px")]
gi = 0.05
cw = (fw - 2 * gi - 2 * 0.045) / 3
for i, (a, b) in enumerate(items):
    box(m + gi + i * (cw + 0.045), y2 + 0.045, cw, 0.33, a, b)
down(W / 2, y2, y2 - G)

# ③ 모델
h3 = 0.27
y3 = y2 - G - h3
box(m, y3, fw, h3, "③ Molmo-7B-D (ACCIDENT 공식 질의·채점 코드)", fc="#e9f5ea")

hw = (fw - 0.10) / 2
cols = [m, W - m - hw]                     # 왼쪽 시점, 오른쪽 위치
for x0 in cols:
    down(x0 + hw / 2, y3, y3 - G)

# 과제
h4 = 0.36
y4 = y3 - G - h4
box(cols[0], y4, hw, h4, "시점 판별", "0.5초 간격 프레임마다 질의")
box(cols[1], y4, hw, h4, "조건부 위치 추정", "정답 프레임에 사고 지점 질의")

# 분기: 왼쪽 검은 화살표(응답 있음) ↓, 오른쪽 빨간 상자(응답 없음) ↓ ↓  — 두 열 모두 같은 배치
h5, gb = 0.36, 0.08
y5 = y4 - gb - h5
h6 = 0.36
y6 = y5 - gb - h6
xa = 0.17                                  # 검은 화살표 위치 (열 왼쪽에서)
mx0 = 0.42                                 # 빨간 상자 시작 위치 (열 기준)
mw = hw - mx0


def branch(x0, ok, miss1, miss2, score, sub):
    down(x0 + xa, y4, y6 + h6)
    ax.text(x0 + xa + 0.03, y5 + h5 / 2, ok, ha="left", va="center", fontsize=6.8, weight=SUBW, linespacing=1.1)
    mx = x0 + mx0
    box(mx, y5, mw, h5, miss1, miss2, fc="#fde9e7", ec=MISS, color=MISS, ft=7.0, fs=6.8)
    down(mx + mw / 2, y4, y5 + h5, color=MISS)
    down(mx + mw / 2, y5, y6 + h6, color=MISS)
    box(x0, y6, hw, h6, score, sub)
    down(x0 + hw / 2, y6, y6 - G)          # ④로 곧게


branch(cols[0], "'예'\n있음", "모두 '아니오': 미검출", "정답 시각÷2의 정수 부분", "시간 유사도 T", "가우시안, σt = 1초")
branch(cols[1], "점\n1개", "점 1개가 아님: 무응답", "화면 중앙으로 채점", "공간 유사도 S", "2차원 가우시안")

# ④ 비교
h7 = 0.36
y7 = y6 - G - h7
box(m, y7, fw, h7, "④ 원본 대비 비교 (같은 영상끼리 짝지음)",
    "점수 차이 · 상대 하락률(RPD) · 부트스트랩 95% 신뢰구간", fc="#f1ecfa")
assert y7 > 0.02, y7

out = Path(__file__).with_suffix(".png")
fig.savefig(out, dpi=600)
print(out, f"bottom margin {y7:.2f} in")
