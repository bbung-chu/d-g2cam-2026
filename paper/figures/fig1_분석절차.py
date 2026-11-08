"""그림 1 전체 분석 절차 (11/9 수정본).

11/2본 대비: 미검출·무응답 처리 경로 추가(지적 29, T2), 용어를 GLOSSARY.md에 맞춤(시점 판별 / 조건부 위치 추정).
11/9 재작성: 논문에 들어가는 실제 크기(한 단 폭 3.35 in)로 그리고 글자를 6.5~7.5 pt로 키움.
HWP에는 100% 크기로 넣는다 (줄이면 글자가 다시 작아진다).
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

W, H = 3.35, 3.27          # 한 단 폭 (inch), 실제 인쇄 크기
INK, SUB = "#1a1a1a", "#333333"
MISS = "#b23a2e"           # 미검출·무응답 경로
FT, FS = 7.5, 6.8          # 제목, 설명 글자 크기 (pt)

fig = plt.figure(figsize=(W, H), dpi=600)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def rbox(x, y, w, h, fc="#ffffff", ec=INK, lw=0.8, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.035",
                                fc=fc, ec=ec, lw=lw, ls=ls))


def box(x, y, w, h, title, sub=None, fc="#ffffff", ec=INK, color=INK, ft=FT, fs=FS, ls="-"):
    rbox(x, y, w, h, fc, ec, ls=ls)
    if sub:
        ax.text(x + w / 2, y + h * 0.69, title, ha="center", va="center", fontsize=ft, weight="bold", color=color)
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=fs,
                color=color if color != INK else SUB)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=ft, weight="bold", color=color)


def arrow(x1, y1, x2, y2, color=INK, label=None, lx=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=color, mutation_scale=8, shrinkA=0, shrinkB=0))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2, label, fontsize=6.5, ha="left", va="center", color=color)


m = 0.05
fw = W - 2 * m
G = 0.085                  # 상자 사이 화살표 길이

# ① 입력
h1 = 0.28
y1 = H - 0.03 - h1
box(m, y1, fw, h1, "① ACCIDENT 실제 CCTV 사고 영상 50개 (시드 0)", fc="#eef3fa")
arrow(W / 2, y1, W / 2, y1 - G)

# ② 열화
h2 = 0.60
y2 = y1 - G - h2
rbox(m, y2, fw, h2, fc="#fdf3e7")
ax.text(W / 2, y2 + h2 - 0.10, "② 열화 적용: 원본 + 3종 × 3단계 = 10조건",
        ha="center", va="center", fontsize=FT, weight="bold")
items = [("해상도 저하", "1/2, 1/4, 1/8"), ("H.264 압축", "CRF 37, 42, 47"), ("모션 블러", "6, 11, 17 px")]
gap = 0.05
cw = (fw - 2 * gap - 2 * 0.04) / 3
for i, (a, b) in enumerate(items):
    box(m + gap + i * (cw + 0.04), y2 + 0.04, cw, 0.33, a, b)
arrow(W / 2, y2, W / 2, y2 - G)

# ③ 모델
h3 = 0.28
y3 = y2 - G - h3
box(m, y3, fw, h3, "③ Molmo-7B-D (ACCIDENT 공식 질의·채점 코드)", fc="#eef7ee")

hw = (fw - 0.08) / 2
xl, xr = m, W - m - hw
for x0 in (xl, xr):
    arrow(x0 + hw / 2, y3, x0 + hw / 2, y3 - G)

# 과제
h4 = 0.36
y4 = y3 - G - h4
box(xl, y4, hw, h4, "시점 판별", "0.5초 간격 프레임마다 질의")
box(xr, y4, hw, h4, "조건부 위치 추정", "정답 프레임에 사고 지점 질의")

# 분기: 왼쪽 = 응답 있음(검은 화살표), 오른쪽 = 응답 없음(빨간 상자)
h5 = 0.36
y5 = y4 - 0.13 - h5
h6 = 0.36
y6 = y5 - 0.13 - h6
mw = hw * 0.64


def branch(x0, miss1, miss2, score, sub):
    ax_ = x0 + 0.17
    arrow(ax_, y4, ax_, y6 + h6, label="응답 있음", lx=0.03)
    mx = x0 + hw - mw
    box(mx, y5, mw, h5, miss1, miss2, fc="#fdecea", ec=MISS, color=MISS, ft=6.8, fs=6.5)
    arrow(mx + mw / 2, y4, mx + mw / 2, y5 + h5, color=MISS)
    arrow(mx + mw / 2, y5, mx + mw / 2, y6 + h6, color=MISS)
    box(x0, y6, hw, h6, score, sub)


branch(xl, "응답 없음: 미검출", "정답 시각÷2로 채점", "시간 유사도 T", "가우시안, σt = 1초")
branch(xr, "점 없음: 무응답", "화면 중앙으로 채점", "공간 유사도 S", "2차원 가우시안")

# ④ 비교
h7 = 0.36
y7 = y6 - G - h7
for x0, dx in ((xl, -0.10), (xr, 0.10)):
    arrow(x0 + hw / 2, y6, W / 2 + dx, y7 + h7)
box(m, y7, fw, h7, "④ 원본 대비 비교 (같은 영상끼리 짝지음)",
    "점수 차이 · 상대 하락률(RPD) · 부트스트랩 95% 신뢰구간", fc="#f3eefa")
assert y7 > 0.02, y7

out = Path(__file__).with_suffix(".png")
fig.savefig(out, dpi=600)
print(out, f"bottom margin {y7:.2f} in")
