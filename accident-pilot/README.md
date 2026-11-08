# 실험 코드·원자료·분석 결과 (CCTV 영상 열화와 VLM 사고 판단)

논문 「CCTV 영상 열화에 따른 교통사고 탐지 AI의 성능 저하 및 취약성 분석」(이우석, 안한석, 안동현)의 실험 코드와 결과 묶음이다.
ACCIDENT 실제 CCTV 사고 영상 50개에 해상도 저하, H.264 압축, 모션 블러를 각 3단계로 적용하고, Molmo-7B-D의 시점 판별과 조건부 위치 추정 점수를 원본과 비교하였다.

## 1. 공식 코드와 직접 작성한 코드의 구분

실험은 ACCIDENT 공식 저장소(https://github.com/accidentbench/ACCIDENT)의 `baselines/llm`을 바탕으로 하였다.
공식 저장소에는 열화를 적용하는 코드가 없으며, 열화 적용과 결과 분석은 모두 직접 작성하였다.

| 파일 | 구분 | 내용 |
|---|---|---|
| `pilot_when_where.py` | **공식 코드 사용 + 수정 + 직접 작성** | 아래 표 참고. 공식 저장소의 `baselines/llm/`에 복사해서 실행한다 |
| `analyze_results.py` | 직접 작성 | 조건별 평균, RPD, 짝지은 부트스트랩 95% 신뢰구간 → `summary.csv`, `summary.md`, `degradation_curve.png` |
| `make_table4.py` | 직접 작성 | 표 4 축소판(원본·열화·차이·RPD) → `results_n50/table4_compact.md` |
| `make_fig3.py` | 직접 작성 | 그림 3(평균·CI·영상별 분포), 미검출 처리 방식별 점수 → `../paper/figures/fig3_성능변화.png`, `results_n50/미검출_영향.md` |
| `make_sample_table.py` | 직접 작성 | 50개 표본과 전체 2,027개의 특성 비교·검정 → `results_n50/표본_특성.md` |
| `*.ipynb`, `where_split/*.ipynb` | 직접 작성 (실행 래퍼) | Google Colab에서 설치, 데이터 다운로드, `pilot_when_where.py` 호출, 결과 저장만 한다 |
| `*.md` (실행가이드 등) | 직접 작성 | 팀 내부 실행 안내 문서 |

### `pilot_when_where.py` 안의 구분

| 부분 | 구분 | 공식 코드와의 관계 |
|---|---|---|
| `MolmoReasoner` (모델 불러오기, `generate_text`, `accident_spatial_reasoning`) | 공식 코드 그대로 | `reasoning/molmo.py`를 import해서 수정 없이 사용 |
| 시점 질의 문장 `TEMPORAL_PROMPT` | 공식 코드 그대로 | `reasoning/molmo.py`의 `accident_temporal_reasoning` 기본 질의와 같음 |
| 0.5초 간격 프레임 선택 | 공식 코드 그대로 | `baselines/temporal/main.py`의 `n=round(fps * 0.5)` |
| `first_yes_frame()` | 공식 코드 수정 | 공식 코드는 모든 프레임에 질의한 뒤 첫 "yes"를 고른다. 프레임마다 독립 질의이므로 첫 "yes"에서 멈춰도 예측은 같아 시간만 줄였다(`--full-scan`이면 공식처럼 끝까지 질의) |
| 점수 계산 (가우시안 유사도, 위치 σ = 전체 사고 상자 평균 크기, 미검출 시 `accident_time // 2` 대체값, 점 없음 → 화면 중앙) | 공식 코드 옮김 | `baselines/temporal/analysis.ipynb`, `baselines/spatial/analysis.ipynb`의 채점 로직을 스크립트로 옮김 |
| 유형 분류(What) 호출 | 삭제 | 시점·위치와 별도 질의이므로 빼도 두 점수에 영향 없음 |
| `degrade()` (해상도 축소·재확대, 수평 선형 모션 블러) | **직접 작성** | 공식 저장소에 없음 |
| `h264_video()` (FFmpeg libx264 재인코딩, preset medium, yuv420p) | **직접 작성** | 공식 저장소에 없음 |
| `MolmoBF16` (`--bf16`) | 직접 작성 | GPU 메모리 40GB 미만일 때만 쓰는 bfloat16 버전. 질의·생성 설정은 공식 코드와 같고, 모델 가중치와 입력 이미지 텐서를 bfloat16으로 바꿈(공식은 `torch_dtype='auto'`) |
| 표본 추출(seed 0으로 섞은 뒤 앞에서 n개), 이어서 실행, 시간 측정, `--mock` | 직접 작성 | |

## 2. 원자료와 분석 결과 (`results_n50/`)

| 파일 | 내용 |
|---|---|
| `molmo_clean.csv`, `molmo_<열화>_<단계>.csv` (10개) | **원자료.** 조건별 영상 50개의 예측 프레임, 예측 위치, 점수, 미검출 여부, 처리 시간 |
| `metadata-real.csv` | ACCIDENT 실제 영상 2,027개의 정답(사고 시각·프레임·상자)과 영상 특성. Kaggle `picekl/accident`에서 받은 원본 그대로 (CC BY-NC-SA 4.0) |
| `summary.csv`, `summary.md`, `degradation_curve.png` | `analyze_results.py` 출력 |
| `table4_compact.md` | `make_table4.py` 출력 |
| `미검출_영향.md` | `make_fig3.py` 출력 |
| `표본_특성.md` | `make_sample_table.py` 출력 |
| `결과해석_4장초안.md` | 팀 내부 결과 해석 메모 |

영상 파일(mp4)은 저작권과 용량 때문에 포함하지 않는다. Kaggle `picekl/accident`의 `real_videos/`에서 받을 수 있다.

## 3. 다시 계산하는 방법

모델을 다시 돌리지 않고 원자료 CSV에서 표·그림을 다시 만들 수 있다.

```bash
pip install numpy pandas matplotlib scipy
python analyze_results.py --results results_n50   # summary.csv, summary.md, degradation_curve.png
python make_table4.py
python make_fig3.py
python make_sample_table.py
```

2026-11-08에 `analyze_results.py`를 원자료 복사본에 다시 실행하여 `summary.csv`와 같은 값이 나오는 것을 확인하였다(최대 차이 3×10⁻¹⁴).

## 4. 실험을 처음부터 실행하는 방법

```bash
git clone https://github.com/accidentbench/ACCIDENT
cd ACCIDENT
pip install -r baselines/llm/requirements.txt     # transformers 4.53.2 등
bash dataset/download_dataset.sh                  # Kaggle 인증 필요
cp <이 폴더>/pilot_when_where.py baselines/llm/
cd baselines/llm
python pilot_when_where.py --n 50 --seed 0                          # 원본
python pilot_when_where.py --n 50 --seed 0 --degrade resize --level 4
python pilot_when_where.py --n 50 --seed 0 --degrade h264   --level 42
python pilot_when_where.py --n 50 --seed 0 --degrade blur   --level 11
# 단계: resize 2·4·8, h264 37·42·47, blur 6·11·17
```

같은 seed를 쓰면 모든 조건에서 같은 영상 50개가 선택된다. 실험은 Google Colab GPU에서 실행하였다(노트북 참고).

## 5. 알려진 사항

- 처음 실행한 원본 조건 50개 중 29개는 다른 조건과 다른 GPU에서 실행되었다(질의 1회당 7~15초, 다른 조건은 약 2.5초). 2026-11-09에 Colab A100에서 공식 설정(`--bf16` 없이)으로 원본 50개를 다시 실행하여 `results_n50/molmo_clean.csv`를 교체하였다(질의 1회당 2.46초). 원래 A100에서 실행된 21개는 예측이 모두 같았고, 나머지 29개 중 3개가 달라졌다. 이전 결과는 `results_n50_old_1102/`에 있다.
- H.264 재인코딩에 쓴 FFmpeg 버전은 실험 당시 기록하지 않았다. 2026-11-08에 Colab 새 노트북에서 확인한 값은 FFmpeg 6.1.1-3ubuntu5, libx264 core 164 r3108이다. 이 FFmpeg는 Ubuntu 24.04 패키지 버전이라 같은 OS 이미지에서는 바뀌지 않는다. 실험 코드가 쓰는 `-fps_mode` 옵션은 FFmpeg 5.1 이상에만 있으므로 실험 때도 5.1 이상이었다.
