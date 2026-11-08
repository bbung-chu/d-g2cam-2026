# 실험 코드·원자료·분석 결과

**논문:** CCTV 영상 열화에 따른 교통사고 탐지 AI의 성능 저하 및 취약성 분석 (이우석, 안한석, 안동현, 2026 융복합지식학회 추계학술발표대회)

ACCIDENT 벤치마크의 실제 CCTV 사고 영상 50개에 해상도 저하, H.264 압축, 모션 블러를 각각 3단계로 적용하고, Molmo-7B-D의 시점 판별·조건부 위치 추정 점수를 같은 영상의 원본과 짝지어 비교한 실험의 전체 자료입니다.

## 먼저 보실 곳

1. **코드 출처 구분** → 이 문서의 [3. 공식 코드와 직접 작성한 코드](#3-공식-코드와-직접-작성한-코드)
2. **실험 코드** → `1_실험코드/pilot_when_where.py` (모델 실행과 열화 적용이 모두 이 파일 하나에 있습니다)
3. **논문 수치의 원자료** → `2_원자료/molmo_*.csv`, 이를 계산한 결과 → `4_분석결과/`

## 1. 폴더 구성

| 폴더 | 내용 |
|---|---|
| `1_실험코드/` | `pilot_when_where.py`: 모델 질의, 열화 적용, 채점. `colab/`: 이 스크립트를 Google Colab GPU에서 실행한 노트북 |
| `2_원자료/` | 조건별 실험 결과 CSV 10개(원본 1 + 열화 9)와 ACCIDENT 정답 메타데이터 |
| `3_분석코드/` | 원자료에서 논문의 표·그림을 만드는 스크립트 5개 (모델을 다시 돌리지 않음) |
| `4_분석결과/` | `3_분석코드/` 실행 결과. 논문 표 3·4, 그림 2·3, 미검출 분석, 표본 특성의 출처 |
| `5_이전결과_1102/` | 원본 조건 재실행(11/9) 전의 결과. 비교용으로 남김 ([6. 알려진 사항](#6-알려진-사항)) |

## 2. 논문의 표·그림과 파일

| 논문 | 파일 | 만드는 코드 |
|---|---|---|
| 3.1 표본 대표성 (p ≥ 0.12) | `4_분석결과/표본_특성.md` | `3_분석코드/make_sample_table.py` |
| 표 3 원본 점수 (0.370, 0.656) | `4_분석결과/summary.csv`의 clean 행 | `3_분석코드/analyze_results.py` |
| 표 4 조건별 점수·차이·RPD·95% CI | `4_분석결과/table4_compact.md` (`summary.csv`에서 생성) | `analyze_results.py` → `make_table4.py` |
| 4.1 미검출 8개(16%), 0점 처리 시 0.341 | `4_분석결과/미검출_영향.md` | `3_분석코드/make_fig3.py` |
| 그림 2 같은 프레임의 조건별 비교 | `4_분석결과/fig2_열화비교_단폭.png` | `3_분석코드/make_fig2.py` (영상 파일 필요) |
| 그림 3 평균·CI·영상별 변화 | `4_분석결과/fig3_성능변화.png` | `3_분석코드/make_fig3.py` |

원자료 CSV 한 행은 영상 하나입니다. 주요 열: `pred_frame`(처음 사고라고 답한 프레임), `when_failed`(미검출 여부), `when_s1.0`(시점 점수 T), `pred_x`·`pred_y`(예측 위치), `where`(위치 점수 S), `when_sec`·`where_sec`(처리 시간).

## 3. 공식 코드와 직접 작성한 코드

실험은 ACCIDENT 공식 저장소(https://github.com/accidentbench/ACCIDENT)의 `baselines/llm`을 바탕으로 하였습니다. 공식 저장소에는 열화를 적용하는 코드가 없으며, **열화 적용과 결과 분석은 모두 직접 작성**하였습니다.

### 파일 단위

| 파일 | 구분 |
|---|---|
| `1_실험코드/pilot_when_where.py` | 공식 코드 사용 + 수정 + 직접 작성 (아래 표) |
| `1_실험코드/colab/*.ipynb` | 직접 작성. Colab에서 패키지 설치, 데이터 다운로드, `pilot_when_where.py` 호출, 결과 저장만 함 |
| `3_분석코드/*.py` | 모두 직접 작성 |

### `pilot_when_where.py` 안의 구분

| 부분 | 구분 | 공식 코드와의 관계 |
|---|---|---|
| `MolmoReasoner` (모델 불러오기, 질의, 위치 응답 해석) | 공식 코드 그대로 | `reasoning/molmo.py`를 import하여 수정 없이 사용 |
| 시점 질의 문장, 0.5초 간격 프레임 선택 | 공식 코드 그대로 | `reasoning/molmo.py`, `baselines/temporal/main.py`와 같음 |
| `first_yes_frame()` | 공식 코드 수정 | 공식 코드는 모든 프레임에 질의한 뒤 첫 "yes"를 고름. 프레임마다 독립 질의이므로 첫 "yes"에서 멈춰도 예측은 같고 시간만 줄어듦 (`--full-scan`이면 공식처럼 끝까지 질의) |
| 채점 (가우시안 유사도, 위치 σ = 전체 사고 상자 평균 크기, 미검출 시 정답 시각÷2 대체값, 점 없음 → 화면 중앙) | 공식 코드 옮김 | `baselines/temporal/analysis.ipynb`, `baselines/spatial/analysis.ipynb`의 채점 로직을 스크립트로 옮김 |
| 유형 분류(What) 호출 | 삭제 | 시점·위치와 별도 질의라 빼도 두 점수는 같음 |
| `degrade()` (해상도 축소·재확대, 수평 선형 모션 블러) | **직접 작성** | 공식 저장소에 없음 |
| `h264_video()` (FFmpeg libx264 재인코딩, preset medium, yuv420p) | **직접 작성** | 공식 저장소에 없음 |
| 표본 추출(시드 0으로 섞은 뒤 앞에서 50개), 이어서 실행, 시간 측정 | 직접 작성 | |
| `MolmoBF16` (`--bf16`) | 직접 작성 | GPU 메모리가 부족할 때만 쓰는 선택 기능. **최종 결과 10개 조건은 모두 이 기능 없이 공식 설정으로 실행** |

## 4. 결과 다시 계산하기 (모델 실행 없이, 1분 이내)

```bash
pip install numpy pandas matplotlib scipy
cd 3_분석코드
python analyze_results.py      # → 4_분석결과/summary.csv, summary.md, degradation_curve.png
python make_table4.py          # → 4_분석결과/table4_compact.md
python make_fig3.py            # → 4_분석결과/fig3_성능변화.png, 미검출_영향.md
python make_sample_table.py    # → 4_분석결과/표본_특성.md
# 그림 2는 영상 파일이 필요합니다 (아래)
python make_fig2.py --video-dir <real_videos 폴더>   # → 4_분석결과/fig2_열화비교_단폭.png
```

위 스크립트를 다시 실행하면 `4_분석결과/`의 표·수치 파일이 그대로 다시 만들어지는 것을 확인하였습니다(2026-11-09). `make_fig2.py`에는 FFmpeg(libx264)와 `opencv-python`, `pillow`가 추가로 필요합니다.

## 5. 실험을 처음부터 실행하기 (GPU 필요)

```bash
git clone https://github.com/accidentbench/ACCIDENT
cd ACCIDENT
pip install -r baselines/llm/requirements.txt     # transformers 4.53.2 등
bash dataset/download_dataset.sh                  # Kaggle 인증 필요
cp <이 폴더>/1_실험코드/pilot_when_where.py baselines/llm/
cd baselines/llm
python pilot_when_where.py --n 50 --seed 0                            # 원본
python pilot_when_where.py --n 50 --seed 0 --degrade resize --level 4  # 해상도 1/4
python pilot_when_where.py --n 50 --seed 0 --degrade h264   --level 42 # CRF 42
python pilot_when_where.py --n 50 --seed 0 --degrade blur   --level 11 # 블러 11 px
# 전체 단계: resize 2·4·8, h264 37·42·47, blur 6·11·17
```

같은 시드를 쓰면 모든 조건에서 같은 영상 50개가 선택됩니다. 실험은 Google Colab A100 GPU에서 실행하였습니다(`1_실험코드/colab/`). 조건 하나에 약 45분이 걸립니다.

## 6. 알려진 사항

- **원본 조건 재실행 (11/9):** 처음 실행한 원본 조건 50개 중 29개는 다른 조건과 다른 GPU에서 실행되었습니다(질의 1회당 7~15초, 다른 조건은 약 2.5초). 2026-11-09에 Colab A100에서 공식 설정으로 원본 50개를 다시 실행하여 `2_원자료/molmo_clean.csv`를 교체하였습니다(질의 1회당 2.46초). 원래 A100에서 실행된 21개는 예측이 모두 같았고, 나머지 29개 중 3개가 달라졌습니다. 원본 점수는 시점 0.375 → 0.370, 위치 0.655 → 0.656이 되었고, 신뢰구간 기준 결론은 같습니다. 이전 결과는 `5_이전결과_1102/`에 있습니다.
- **FFmpeg 버전:** 실험 당시 기록하지 않았습니다. 2026-11-08에 Colab 새 노트북에서 확인한 값은 FFmpeg 6.1.1-3ubuntu5, libx264 core 164 r3108입니다. 이는 Ubuntu 24.04 패키지 버전이라 같은 OS 이미지에서는 바뀌지 않습니다. 실험 코드가 쓰는 `-fps_mode` 옵션은 FFmpeg 5.1 이상에만 있어 실험 때도 5.1 이상이었습니다.
- **영상 미포함:** 영상 파일(mp4)은 저작권과 용량 때문에 넣지 않았습니다. Kaggle `picekl/accident`의 `real_videos/`에서 받을 수 있습니다(CC BY-NC-SA 4.0). `2_원자료/metadata-real.csv`는 같은 곳에서 받은 원본 그대로입니다.
