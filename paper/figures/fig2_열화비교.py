#!/usr/bin/env python3
"""그림 2: 같은 정답 사고 프레임에 10개 조건(원본 + 열화 9개)을 적용한 비교 (이슈 #3, 지적 35·36·38).

- 열화는 accident-pilot/pilot_when_where.py 의 degrade(), h264_video() 소스를 그대로 읽어 실행한다
  (논문 실험과 같은 코드. 모듈을 import하면 ACCIDENT 저장소의 dataset 패키지가 필요해서 함수만 꺼내 쓴다).
- 정답 위치: metadata-real.csv 의 x1, y1, x2, y2 (초록 상자)
- 모델 예측: results_n50/molmo_<조건>.csv 의 pred_x, pred_y (빨간 ×). 모델은 다시 돌리지 않는다.
- 사고 영역이 작아서 정답 상자 중심으로 프레임 일부를 잘라 보여준다 (--crop: 프레임 너비 대비 비율).

영상 mp4는 저장소에 넣지 않는다 (저작권·용량). 기본 경로는 저장소 밖 ../사례영상/real_videos/.

실행 (저장소 루트에서):
    python paper/figures/fig2_열화비교.py --video BwSlVnQTtNo_00
필요: opencv-python, numpy, pandas, pillow, matplotlib, ffmpeg(libx264). ffmpeg가 PATH에 없으면 --ffmpeg로 경로 지정.
"""
import argparse
import ast
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PILOT = REPO / "accident-pilot" / "pilot_when_where.py"
RESULTS = REPO / "accident-pilot" / "results_n50"

CONDS = [("clean", "0", "원본"),
         ("resize", "2", "해상도 1/2"), ("resize", "4", "해상도 1/4"), ("resize", "8", "해상도 1/8"),
         ("h264", "37", "H.264 CRF 37"), ("h264", "42", "H.264 CRF 42"), ("h264", "47", "H.264 CRF 47"),
         ("blur", "6", "블러 6 px"), ("blur", "11", "블러 11 px"), ("blur", "17", "블러 17 px")]
# 2×5 배치: 1행 원본 + 해상도 3 + CRF 37, 2행 CRF 42·47 + 블러 3
ORDER = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def load_pilot_functions():
    """pilot_when_where.py에서 degrade, h264_video 함수 정의만 꺼내 실행한다."""
    tree = ast.parse(PILOT.read_text(encoding="utf-8"))
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("degrade", "h264_video")]
    assert len(keep) == 2, "degrade/h264_video를 찾지 못함"
    ns = {"np": np, "Image": Image, "Path": Path, "subprocess": __import__("subprocess")}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(PILOT), "exec"), ns)
    return ns["degrade"], ns["h264_video"]


def read_frame(path, i):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
    ok, f = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {i} of {path}")
    return Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))


def set_korean_font():
    for name in ("AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"):
        if any(name == f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="BwSlVnQTtNo_00")
    ap.add_argument("--video-dir", type=Path, default=REPO.parent / "사례영상" / "real_videos")
    ap.add_argument("--meta", type=Path, default=RESULTS / "metadata-real.csv")
    ap.add_argument("--crop", type=float, default=0.36, help="잘라 보여줄 너비 (프레임 너비 대비)")
    ap.add_argument("--ffmpeg", default=None, help="ffmpeg 실행 파일 경로 (PATH에 없을 때)")
    ap.add_argument("--blur-regions", default="", help="비식별화: 'x1,y1,x2,y2;...' (정규화 좌표, 원본 프레임 기준)")
    ap.add_argument("--out", type=Path, default=REPO / "paper" / "figures" / "fig2_열화비교.png")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    degrade, h264_video = load_pilot_functions()
    if args.ffmpeg:  # h264_video()는 "ffmpeg"를 PATH에서 찾으므로 임시 폴더에 링크를 둔다
        tmpbin = Path(tempfile.mkdtemp())
        (tmpbin / "ffmpeg").symlink_to(Path(args.ffmpeg).resolve())
        os.environ["PATH"] = f"{tmpbin}{os.pathsep}{os.environ['PATH']}"
    assert shutil.which("ffmpeg"), "ffmpeg가 필요하다 (--ffmpeg로 지정)"

    rel = f"real_videos/{args.video}.mp4"
    src = args.video_dir / f"{args.video}.mp4"
    meta = pd.read_csv(args.meta).set_index("path").loc[rel]
    fr = int(meta["accident_frame"])
    cache = Path(tempfile.mkdtemp())

    blur_boxes = [tuple(map(float, b.split(","))) for b in args.blur_regions.split(";") if b.strip()]

    def anonymize(img):
        if not blur_boxes:
            return img
        a = np.asarray(img).copy()
        h, w = a.shape[:2]
        for x1, y1, x2, y2 in blur_boxes:
            xa, xb, ya, yb = int(x1 * w), int(x2 * w), int(y1 * h), int(y2 * h)
            a[ya:yb, xa:xb] = cv2.GaussianBlur(a[ya:yb, xa:xb], (0, 0), 3)
        return Image.fromarray(a)

    frames, preds = [], []
    for kind, level, _ in CONDS:
        video = h264_video(src, level, cache) if kind == "h264" else src
        img = degrade(anonymize(read_frame(video, fr)), kind, level)
        frames.append(img)
        tag = "clean" if kind == "clean" else f"{kind}_{level}"
        r = pd.read_csv(RESULTS / f"molmo_{tag}.csv").set_index("path").loc[rel]
        preds.append((float(r["pred_x"]), float(r["pred_y"]), float(r["where"])))

    W, H = frames[0].size
    cw = args.crop * W
    ch = cw * H / W
    cx, cy = meta["center_x"] * W, meta["center_y"] * H
    x0 = min(max(cx - cw / 2, 0), W - cw)
    y0 = min(max(cy - ch / 2, 0), H - ch)

    set_korean_font()
    fig, axes = plt.subplots(2, 5, figsize=(6.7, 1.95))
    for ax, k in zip(axes.flat, ORDER):
        img = np.asarray(frames[k])
        ax.imshow(img)
        ax.set_xlim(x0, x0 + cw)
        ax.set_ylim(y0 + ch, y0)
        bx1, by1, bx2, by2 = (meta[c] for c in ("x1", "y1", "x2", "y2"))
        ax.add_patch(Rectangle((bx1 * W, by1 * H), (bx2 - bx1) * W, (by2 - by1) * H,
                               fill=False, ec="#00e000", lw=0.8))
        px, py, s = preds[k]
        ax.plot(px * W, py * H, marker="x", color="#ff2020", ms=4.5, mew=1.1)
        ax.set_title(f"{CONDS[k][2]}  S={s:.2f}", fontsize=5.5, pad=1.5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.3)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.93, bottom=0.01, wspace=0.03, hspace=0.16)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"saved {args.out}  frame={fr}  size={W}x{H}  crop=({x0:.0f},{y0:.0f},{cw:.0f}x{ch:.0f})")
    for (kind, level, name), (px, py, s) in zip(CONDS, preds):
        print(f"  {name:14s} pred=({px:.3f},{py:.3f}) S={s:.3f}")


if __name__ == "__main__":
    main()
