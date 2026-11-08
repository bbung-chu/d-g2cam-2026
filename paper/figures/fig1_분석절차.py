"""그림 1 전체 분석 절차 (11/9 수정본).

11/2본 대비: 미검출·무응답 처리 경로 추가(지적 29, T2), 글자·화살표 키움, 600 dpi,
용어를 GLOSSARY.md에 맞춤(시점 판별 / 조건부 위치 추정).
실행: python fig1_분석절차.py  → 같은 폴더에 fig1_분석절차.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

plt.rcParams["font.family"] = "Malgun Gothic"

W, H = 3.4, 4.5          # 2단 논문 한 단 너비(약 86 mm)
INK = "#222222"
fig = plt.figure(figsize=(W, H), dpi=600)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def box(x, y, w, h, title, sub=None, fc="#ffffff", fs=7.0, lw=0.9, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.04",
                                fc=fc, ec=INK, lw=lw, ls=ls))
    if sub:
        ax.text(x + w / 2, y + h - 0.055, title, ha="center", va="top", fontsize=fs, weight="bold")
        ax.text(x + w / 2, y + (h - 0.17) / 2, sub, ha="center", va="center",
                fontsize=fs - 1.0, color="#333333", linespacing=1.2)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs, weight="bold")


def arrow(x1, y1, x2, y2, label=None, lx=0.0, ly=0.0, color=INK):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color=color, mutation_scale=9,
                                shrinkA=0, shrinkB=0))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, fontsize=5.8,
                ha="center", va="center", color=color, weight="bold")


def diamond(cx, cy, w, h, text):
    ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)],
                         closed=True, fc="#fff8d6", ec=INK, lw=0.9))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=5.8, weight="bold", linespacing=1.1)


m = 0.08
fw = W - 2 * m
MISS = "#b23a2e"   # 미검출·무응답 경로 색

# ① 입력
box(m, 4.0, fw, 0.44, "① 입력 영상",
    "ACCIDENT 실제 CCTV 사고 영상 50개 (seed 0 무작위 추출)", fc="#eef3fa")
arrow(W / 2, 4.0, W / 2, 3.86)

# ② 열화
y2, h2 = 3.06, 0.80
ax.add_patch(FancyBboxPatch((m, y2), fw, h2, boxstyle="round,pad=0.015,rounding_size=0.04",
                            fc="#fdf3e7", ec=INK, lw=0.9))
ax.text(W / 2, y2 + h2 - 0.11, "② 열화 적용 (원본 + 3종 × 3단계 = 10조건)",
        ha="center", va="center", fontsize=7.0, weight="bold")
labels = [("원본", "열화 없음"), ("해상도 저하", "1/2, 1/4, 1/8"),
          ("H.264 압축", "CRF 37, 42, 47"), ("모션 블러", "6, 11, 17 px")]
cw = (fw - 0.10 - 3 * 0.05) / 4
for i, (a, b) in enumerate(labels):
    box(m + 0.05 + i * (cw + 0.05), y2 + 0.07, cw, 0.48, a, b, fs=6.4)
arrow(W / 2, y2, W / 2, 2.92)

# ③ 모델
box(m, 2.52, fw, 0.40, "③ Molmo-7B-D-0924 (zero-shot)",
    "ACCIDENT 공식 질의·채점 코드 사용", fc="#eef7ee")

hw = (fw - 0.10) / 2
xl, xr = m, W - m - hw            # 왼쪽(시점), 오른쪽(위치) 열
cl, cr = xl + hw / 2, xr + hw / 2
arrow(cl, 2.52, cl, 2.40); arrow(cr, 2.52, cr, 2.40)

# 시점 판별
box(xl, 1.88, hw, 0.52, "시점 판별 (프레임 단위)", "0.5초 간격 프레임마다\n사고 여부 질의", fs=6.6)
# 위치 추정
box(xr, 1.88, hw, 0.52, "조건부 위치 추정", "정답 사고 프레임을 주고\n사고 지점(점) 질의", fs=6.6)

# 분기: 각 열 왼쪽에 판단 마름모, 오른쪽에 미검출·무응답 처리 상자
DW, DH, DY = 0.66, 0.40, 1.56        # 마름모 너비·높이·중심 y
MW, MH = 0.82, 0.50                  # 처리 상자
TY, TH = 0.80, 0.40                  # T·S 상자


def branch(x0, question, miss_title, miss_sub, score_title, score_sub):
    dcx = x0 + 0.06 + DW / 2
    arrow(dcx, 1.88, dcx, DY + DH / 2)
    diamond(dcx, DY, DW, DH, question)
    mx = x0 + hw - MW
    ax.add_patch(FancyBboxPatch((mx, DY - MH / 2), MW, MH, boxstyle="round,pad=0.015,rounding_size=0.04",
                                fc="#fdecea", ec=MISS, lw=0.9))
    ax.text(mx + MW / 2, DY + MH / 2 - 0.05, miss_title, ha="center", va="top",
            fontsize=6.0, weight="bold", color=MISS)
    ax.text(mx + MW / 2, DY - 0.07, miss_sub, ha="center", va="center",
            fontsize=5.4, color=MISS, linespacing=1.15)
    arrow(dcx + DW / 2, DY, mx, DY, color=MISS)
    arrow(dcx, DY - DH / 2, dcx, TY + TH, "예", lx=0.08)
    arrow(mx + MW / 2, DY - MH / 2, mx + MW / 2, TY + TH, color=MISS)
    box(x0, TY, hw, TH, score_title, score_sub, fs=6.4)


branch(xl, "'예' 응답\n있음?", "아니오: 미검출",
       "공식 대체값\n(정답 시각÷2) 사용,\n비율 따로 보고",
       "시간 유사도 T", "가우시안, σt = 1초")
branch(xr, "점 좌표\n있음?", "아니오: 무응답",
       "화면 중앙 좌표로\n처리",
       "공간 유사도 S", "2차원 가우시안")

# ④ 비교
arrow(cl, TY, W / 2 - 0.08, 0.62); arrow(cr, TY, W / 2 + 0.08, 0.62)
box(m, 0.06, fw, 0.56, "④ 원본 대비 비교",
    "원본 점수 · 열화 점수 · 차이 · 상대 하락률(RPD)\n영상 단위 짝지은 부트스트랩 2,000회 → 95% 신뢰구간",
    fc="#f3eefa")

out = Path(__file__).with_suffix(".png")
fig.savefig(out, dpi=600)
print(out)
