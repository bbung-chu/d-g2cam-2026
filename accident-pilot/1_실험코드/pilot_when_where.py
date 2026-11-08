#!/usr/bin/env python3
"""
ACCIDENT 예비 실험: Molmo-7B-D zero-shot, When(시점) + Where(위치, 시점 오라클)만.

공식 코드(baselines/llm)의 로직을 그대로 따르되, 유형(What) 분류 호출은 뺐다.
  - When : baselines/temporal/main.py + temporal/analysis.ipynb 와 동일
  - Where: baselines/spatial/molmo.ipynb (oracle) + spatial/analysis.ipynb 와 동일
논문 비교 기준: 시점 when_s1.0 (Molmo 0.343), 위치 where (Molmo 0.596, 오라클).
논문 수치는 Zero-shot 묶음(실제 영상 2,027개 전체) 기준으로 추정된다.

위치: ACCIDENT 저장소의 baselines/llm/ 폴더에 복사해서 실행.
    cd ACCIDENT/baselines/llm
    python pilot_when_where.py --mock --n 3                 # GPU 없이 데이터·열화 파이프라인만 확인
    python pilot_when_where.py --n 5                        # 모델까지 동작 확인
    python pilot_when_where.py --n 20 --seed 0              # 소요 시간 측정
    python pilot_when_where.py --n 20 --degrade resize --level 4
    python pilot_when_where.py --n 20 --degrade h264   --level 42
    python pilot_when_where.py --n 20 --degrade blur   --level 15

열화 (--level 값은 팀에서 정하는 수치를 그대로 넣는다):
  resize : 1/level 로 축소 후 원래 크기로 다시 확대 (예: 2, 4, 8)
  h264   : ffmpeg libx264 로 영상을 CRF=level 로 재인코딩 (예: 32, 37, 42)
  blur   : 가로 방향 선형 모션 블러, 길이 level 픽셀 (예: 5, 15, 25)
"""
import argparse
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

LLM_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LLM_DIR.parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(LLM_DIR))

from dataset.accident_dataset import get_dataset_paths, resolve_dataset_root  # noqa: E402

# decord가 설치되지 않는 환경(예: 최신 코랩)에서는 OpenCV로 읽는다.
# 공식 reasoning.utils가 decord를 import하므로 빈 모듈로 대신 채워 둔다.
try:
    import decord  # noqa: F401
    HAS_DECORD = True
except ImportError:
    import types
    HAS_DECORD = False
    _stub = types.ModuleType("decord")
    _stub.VideoReader = _stub.cpu = None
    sys.modules["decord"] = _stub

SIGMAS = [0.5, 1.0, 2.0]


# ---------- 프레임 읽기 (decord가 없으면 OpenCV로) ----------
def _reader(path):
    if HAS_DECORD:
        os.environ["DECORD_VERBOSE"] = "0"
        from decord import VideoReader, cpu
        vr = VideoReader(str(path), ctx=cpu(0))
        return len(vr), lambda i: Image.fromarray(vr[i].asnumpy()).convert("RGB")
    else:
        import cv2
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        def get(i):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            if not ok:
                raise RuntimeError(f"cannot read frame {i} of {path}")
            return Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        return n, get


TEMPORAL_PROMPT = "Is there a traffic accident or collision? Yes or No answer only."  # 공식 molmo.py와 동일


def first_yes_frame(model, path, step, kind, level, full_scan=False):
    """0.5초 간격 프레임에 차례로 질의해 처음 "yes"인 프레임 번호를 돌려준다 (공식 방식과 같은 예측).
    공식 코드는 모든 프레임에 질의한 뒤 첫 "yes"를 고르지만, 프레임마다 독립적인 질의(greedy)라
    첫 "yes"에서 멈춰도 예측은 똑같다. full_scan=True면 공식처럼 끝까지 질의한다."""
    total, get = _reader(path)
    first, n = None, 0
    for i in range(0, total, step):
        answer = model.generate_text(degrade(get(i), kind, level), TEMPORAL_PROMPT)
        n += 1
        if "yes" in answer.strip().lower():
            if first is None:
                first = i
            if not full_scan:
                break
    return first, n


def load_frame(path, i):
    total, get = _reader(path)
    return get(min(int(i), total - 1))


# ---------- 열화 ----------
def degrade(img, kind, level):
    """프레임 단위 열화. h264는 영상 단위라 여기서는 그대로 통과."""
    if kind in ("clean", "h264"):
        return img
    if kind == "resize":
        f = float(level)
        w, h = img.size
        small = img.resize((max(1, round(w / f)), max(1, round(h / f))), Image.BICUBIC)
        return small.resize((w, h), Image.BICUBIC)
    if kind == "blur":
        k = int(level)
        if k <= 1:
            return img
        a = np.asarray(img).astype(np.float32)
        pad = np.pad(a, ((0, 0), (k // 2, k - 1 - k // 2), (0, 0)), mode="edge")
        c = np.cumsum(pad, axis=1)
        c = np.concatenate([np.zeros_like(c[:, :1]), c], axis=1)
        out = (c[:, k:] - c[:, :-k]) / k
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    raise ValueError(f"unknown degradation: {kind}")


def h264_video(src, crf, cache_dir):
    """원본 영상을 CRF로 재인코딩한 복사본 경로 (한 번 만들면 재사용). 프레임 수·fps 유지."""
    dst = Path(cache_dir) / f"crf{crf}" / Path(src).name
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
             "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
             "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", "-an", str(dst)],
            check=True)
    return dst


# ---------- 모델 ----------
def load_model(mock, bf16):
    if mock:
        return MockReasoner()
    from reasoning.molmo import MolmoReasoner
    if not bf16:
        return MolmoReasoner()  # 공식 코드 그대로

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

    class MolmoBF16(MolmoReasoner):
        """GPU 메모리가 부족할 때만. 공식 설정과 수치가 조금 다를 수 있다."""
        def __init__(self):
            name = "allenai/Molmo-7B-D-0924"
            self.processor = AutoProcessor.from_pretrained(
                name, trust_remote_code=True, torch_dtype="auto", device_map="auto")
            self.model = AutoModelForCausalLM.from_pretrained(
                name, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda")

        def generate_text(self, img, prompt):
            inputs = self.processor.process(images=[img], text=prompt)
            inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
            inputs["images"] = inputs["images"].to(torch.bfloat16)
            out = self.model.generate_from_batch(
                inputs, GenerationConfig(max_new_tokens=512, stop_strings="<|endoftext|>"),
                tokenizer=self.processor.tokenizer)
            return self.processor.tokenizer.decode(
                out[0, inputs["input_ids"].size(1):], skip_special_tokens=True)

    return MolmoBF16()


class MockReasoner:
    """GPU 없이 파이프라인만 확인할 때 쓰는 가짜 모델 (점수·시간은 의미 없음)."""
    def generate_text(self, img, prompt):
        return "Yes" if random.random() < 0.1 else "No"

    def accident_spatial_reasoning(self, img):
        w, h = img.size
        return (random.randint(0, w - 1), random.randint(0, h - 1)), "mock"


# ---------- 실행 ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-path", type=Path, default=PROJECT_ROOT / "dataset")
    p.add_argument("--metadata", type=Path, default=None,
                   help="다른 메타데이터 CSV를 쓸 때 (기본: dataset/metadata-real.csv)")
    p.add_argument("--split-col", default=None)
    p.add_argument("--split-value", default=None)
    p.add_argument("--n", type=int, default=20, help="샘플 수 (0이면 전체)")
    p.add_argument("--seed", type=int, default=0, help="같은 seed면 조건이 달라도 같은 영상")
    p.add_argument("--degrade", default="clean", choices=["clean", "resize", "h264", "blur"])
    p.add_argument("--level", default="0")
    p.add_argument("--skip-when", action="store_true")
    p.add_argument("--skip-where", action="store_true")
    p.add_argument("--mock", action="store_true", help="모델 없이 파이프라인만 확인")
    p.add_argument("--bf16", action="store_true", help="GPU 메모리가 40GB 미만일 때")
    p.add_argument("--full-scan", action="store_true",
                   help="첫 yes 이후 프레임도 공식 코드처럼 모두 질의 (예측은 같고 시간만 더 걸림)")
    p.add_argument("--full-n", type=int, default=2027, help="소요 시간 추정용 전체 영상 수")
    p.add_argument("--n-conditions", type=int, default=10,
                   help="소요 시간 추정용 조건 수 (원본 1 + 열화 3종×3단계)")
    p.add_argument("--out", type=Path, default=LLM_DIR / "results" / "pilot")
    args = p.parse_args()

    random.seed(args.seed)
    root = resolve_dataset_root(args.dataset_path)
    _, meta_path = get_dataset_paths(root, kind="real")
    full = pd.read_csv(args.metadata or meta_path)
    full["video_path"] = full["path"].apply(lambda x: os.path.join(root, x))

    # 위치 점수의 sigma: 공식 analysis와 같이 전체 메타데이터의 평균 bbox 크기
    sigma_x = (full["x2"] - full["x1"]).abs().mean()
    sigma_y = (full["y2"] - full["y1"]).abs().mean()

    df = full
    if args.split_col:
        df = df[df[args.split_col].astype(str) == str(args.split_value)]
    # 같은 seed로 섞은 뒤 앞에서 n개 → n을 20에서 50으로 늘려도 앞의 20개는 그대로(이어서 실행)
    df = df.sample(frac=1, random_state=args.seed)
    if args.n:
        df = df.head(args.n)

    tag = "clean" if args.degrade == "clean" else f"{args.degrade}_{args.level}"
    if args.mock:
        tag = "mock_" + tag
    args.out.mkdir(parents=True, exist_ok=True)
    out_csv = args.out / f"molmo_{tag}.csv"
    done = set(pd.read_csv(out_csv)["path"]) if out_csv.exists() else set()
    print(f"[{tag}] videos: {len(df)} (already done: {len(done & set(df['path']))})")

    t0 = time.perf_counter()
    model = load_model(args.mock, args.bf16)
    print(f"model load: {time.perf_counter() - t0:.1f}s")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        if row["path"] in done:
            continue
        video = row["video_path"]
        rec = {"path": row["path"], "type": row.get("type"), "duration": row["duration"],
               "accident_time": row["accident_time"]}

        t = time.perf_counter()
        if args.degrade == "h264":
            video = h264_video(video, args.level, root / "degraded")
        rec["prep_sec"] = time.perf_counter() - t

        if not args.skip_when:
            t = time.perf_counter()
            fps = row["no_frames"] / row["duration"]
            frame_id, n_q = first_yes_frame(model, video, round(fps * 0.5),
                                            args.degrade, args.level, args.full_scan)
            # 공식 analysis.ipynb의 대체값을 그대로 사용 (주의: accident_time//2는 정답값을 씀)
            pred_ts = frame_id / fps if frame_id is not None else row["accident_time"] // 2
            rec.update(pred_frame=frame_id, pred_ts=pred_ts, when_failed=frame_id is None,
                       n_queries=n_q, when_sec=time.perf_counter() - t)
            for s in SIGMAS:
                rec[f"when_s{s}"] = float(np.exp(-(pred_ts - row["accident_time"]) ** 2 / (2 * s ** 2)))

        if not args.skip_where:
            t = time.perf_counter()
            img = degrade(load_frame(video, row["accident_frame"]), args.degrade, args.level)
            (x, y), raw = model.accident_spatial_reasoning(img)
            w, h = img.size
            if x is None or y is None:
                x, y = w / 2, h / 2
            xn, yn = x / w, y / h
            rec.update(pred_x=xn, pred_y=yn, where_raw=raw, where_sec=time.perf_counter() - t)
            rec["where"] = float(np.exp(-((xn - row["center_x"]) ** 2 / (2 * sigma_x ** 2)
                                          + (yn - row["center_y"]) ** 2 / (2 * sigma_y ** 2))))

        pd.DataFrame([rec]).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)

    summarize(pd.read_csv(out_csv), tag, args, full)


def summarize(res, tag, args, full):
    print(f"\n== 점수 [{tag}] n={len(res)} ==")
    for c in [c for c in res.columns if c[:6] == "when_s" and c[6:7].isdigit() or c == "where"]:
        print(f"  {c:10s} {res[c].mean():.3f}")
    if "when_failed" in res:
        print(f"  시점 예측 실패(대체값 사용) 비율: {res['when_failed'].mean():.1%}")

    print("\n== 소요 시간 ==")
    sec_cols = [c for c in ["prep_sec", "when_sec", "where_sec"] if c in res]
    per_video = res[sec_cols].sum(axis=1)
    print(f"  영상당 평균 {per_video.mean():.1f}s  (" +
          ", ".join(f"{c} {res[c].mean():.1f}s" for c in sec_cols) + ")")
    for n in (50, 100):
        print(f"  추정: 영상 {n}개 × {args.n_conditions}조건 ≈ "
              f"{per_video.mean() * n * args.n_conditions / 3600:.1f}시간")
    if "when_sec" in res and res["n_queries"].sum() > 0:
        per_q = res["when_sec"].sum() / res["n_queries"].sum()
        print(f"  시점 질의 1회당 {per_q:.2f}s, 영상당 질의 평균 {res['n_queries'].mean():.0f}회")
        # 시점 시간은 영상 길이에 비례 → 영상 1초당 시점 시간 × 전체 메타데이터의 길이로 보정
        when_per_sec = res["when_sec"].sum() / res["duration"].sum()
        when_full = when_per_sec * full["duration"].sum() * args.full_n / len(full)
        other = res[[c for c in ("prep_sec", "where_sec") if c in res]].sum(axis=1).mean()
        one = when_full + other * args.full_n
        print(f"  추정: 영상 {args.full_n}개 × 1조건 ≈ {one / 3600:.1f}시간, "
              f"× {args.n_conditions}조건 ≈ {one * args.n_conditions / 3600:.1f}시간")


if __name__ == "__main__":
    main()
