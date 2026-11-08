# ACCIDENT 베이스라인 직접 돌려보기 (Molmo-7B-D, 시점·위치)

정리: 2026-10-13 · 공식 저장소 https://github.com/accidentbench/ACCIDENT (커밋 38b2297, 2026-05-07 기준)

목표는 두 단계다.
1. **원본 재현**: 영상 50~100개로 Molmo 시점·위치 점수가 논문 수치(시점 0.343 @σt=1초, 위치 0.596 @시점 오라클)와 비슷한지 확인. 출력의 `when_s1.0`과 `where`를 보면 된다.
2. **열화 적용**: 같은 영상에 해상도 저하·H.264 압축·모션 블러를 걸고 다시 측정

---

## 0. 필요한 환경

| 항목 | 내용 |
|---|---|
| GPU | **NVIDIA GPU 필수.** 공식 코드가 `.to('cuda')`로 고정돼 있음. Molmo-7B-D를 코드 그대로(`torch_dtype='auto'`) 올리면 VRAM 약 30GB 이상이 필요한 것으로 추정 → A100 40GB/80GB 권장 (Colab Pro A100, 학교 GPU 서버 등) |
| 맥북 | 현재 이 노트북(Apple Silicon)으로는 코드 수정 없이 못 돌림. CUDA가 없고, 영상 로더 `decord`도 맥 ARM용 설치가 까다로움 |
| Python | 3.10~3.11 권장, `torch==2.4.0`, `transformers==4.53.2` (requirements.txt에 고정) |
| 디스크 | 데이터셋(실제 영상) + 모델 가중치 약 30GB |
| 계정 | Kaggle 계정과 API 토큰 (데이터 다운로드용) |

## 1. 코드와 데이터 받기

```bash
git clone https://github.com/accidentbench/ACCIDENT
cd ACCIDENT

# Kaggle CLI 설치 후 인증 (~/.kaggle/kaggle.json 또는 kaggle auth login)
pip install kaggle
bash dataset/download_dataset.sh
```

다운로드가 끝나면 이 구조가 되어야 한다.
```text
dataset/
  metadata-real.csv      ← 실제 영상 메타데이터 (정답 시점·bbox·유형)
  real_videos/...
```
이미 데이터를 받아 뒀다면 다운로드는 건너뛰고, 그 폴더를 `dataset/` 위치에 두거나 `--dataset-path`로 지정한다.

## 2. 환경 설치

```bash
cd baselines/llm
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 3. 공식 스크립트로 동작 확인 (영상 3개)

```bash
python baselines/temporal/main.py --model molmo --dataset-path ../../dataset --range 0:3
```
모델이 처음 로드될 때 Hugging Face에서 `allenai/Molmo-7B-D-0924` 가중치를 받는다. 결과는 `results/molmo_temporal_parts/0.pkl ...`에 저장된다. 여기까지 되면 환경은 끝.

## 4. 오늘 예비 실험 순서

**코랩으로 할 때는 [ACCIDENT_예비실험_colab.ipynb](ACCIDENT_예비실험_colab.ipynb)를 위에서부터 실행하면 된다** (A100 런타임, 아래 명령이 그대로 들어 있음).

목적: (1) 제대로 돌아가는지, (2) 전체 실험에 얼마나 걸릴지, (3) 열화까지 포함한 실험 틀이 동작하는지 확인.
스크립트: [pilot_when_where.py](pilot_when_where.py) (공식 로직 그대로, 유형 분류만 뺌. 시간 측정과 열화 3종 포함)

```bash
cp /경로/pilot_when_where.py ACCIDENT/baselines/llm/
cd ACCIDENT/baselines/llm
```

| 단계 | 명령 | 확인할 것 |
|---|---|---|
| 0. 파이프라인 (GPU 없이, 맥에서도 가능) | `python pilot_when_where.py --mock --n 3` | 데이터 읽기·열화·저장이 오류 없이 끝나는지. 점수는 의미 없음 |
| 1. 모델 동작 | `python pilot_when_where.py --n 5` | Molmo가 로드되고 점수가 나오는지 |
| 2. 원본 + 시간 측정 | `python pilot_when_where.py --n 20 --seed 0` | 출력의 `when_s1.0`, `where`와 "소요 시간" 추정치 |
| 3. 열화 틀 | `--degrade resize --level 4` / `--degrade h264 --level 42` / `--degrade blur --level 15` 각각 `--n 5 --seed 0` | 조건마다 결과 CSV가 따로 생기는지, 점수가 원본보다 떨어지는 방향인지 |

- 0단계를 맥에서 할 때는 `pip install pandas numpy pillow tqdm opencv-python`만 있으면 된다 (h264는 ffmpeg 필요: `brew install ffmpeg`).
- 같은 `--seed`면 조건이 달라도 같은 영상이 뽑혀서 원본과 짝지어 비교할 수 있다.
- 결과: `results/pilot/molmo_<조건>.csv` (영상별 한 줄, 끊겨도 이어서 실행). 끝에 점수, 시점 예측 실패 비율, 소요 시간이 출력된다.
- 소요 시간 추정은 "영상 50·100개 × 10조건"과 "전체 2,027개 × 1조건·10조건"을 함께 보여준다. 시점 질의 수는 영상 길이에 비례하므로 전체 메타데이터의 영상 길이로 보정한다.
- GPU 메모리가 40GB 미만이면 `--bf16`을 붙인다 (공식 설정과 수치가 조금 다를 수 있음).
- 20개 2단계가 오래 걸리면 `--n 10`으로 줄여도 시간 추정에는 충분하다.

### 공식 코드가 하는 일 (논문 설명용)

| 과제 | 입력 | 질의 | 예측 | 점수 |
|---|---|---|---|---|
| 시점(When) | 0.5초 간격 프레임 전부 | 프레임마다 "Is there a traffic accident or collision? Yes or No answer only." | 처음 "yes"가 나온 프레임의 시각 | exp(−(예측−정답)² / 2σ²), σ = 0.5, 1, 2초 |
| 위치(Where), 시점 오라클 | 정답 사고 프레임 1장 | "The scene depicts a traffic accident with one or more cars colliding. Point to the car accident." | Molmo가 찍은 점(x, y) | 2차원 가우시안, σ = 전체 데이터 평균 bbox 너비·높이(정규화 좌표) |

추론량: 시점은 영상 길이(초) × 2번, 위치는 영상당 1번. 100개 pilot이면 대략 수천 번 호출이라 A100 기준 수십 분~1시간대로 예상(추정치).

## 4-B. 본 실험과 결과 정리

pilot이 아니라 실제 영상 실험 결과로 결론을 내기로 했다 (2026-10-13). 노트북 6·7절이 이 단계다.

- 6절: 원본 1 + 열화 3종 × 3단계 = 10조건을 같은 영상 `N`개로 돌린다. `N`은 5-3에서 잰 소요 시간으로 정한다.
- 7절: [analyze_results.py](analyze_results.py)가 조건별 평균, 원본 대비 하락률(RPD), 영상 단위 부트스트랩 95% 신뢰구간을 표(`summary.md`)와 그래프(`degradation_curve.png`)로 만든다. 논문의 결과 표 1개와 그림 1개에 그대로 쓸 수 있다.
- 비용 감각(추정): 영상 중앙값 26.8초(논문 부록 A) → 영상당 시점 질의 약 54회. 2,027개 전체면 조건 하나에 시점 질의 약 11만 회, 10조건이면 100만 회 이상이라 코랩으로는 마감 전에 끝나기 어렵다. 위치는 영상당 1회라 전체도 부담이 작다.

## 5. 열화 구현 방식

| 열화 | `--level` 의미 | 구현 |
|---|---|---|
| 해상도 저하 `resize` | 축소 배율 (예: 2, 4, 8) | 프레임을 1/배율로 줄였다가 원래 크기로 다시 키움 |
| H.264 압축 `h264` | CRF 값 (예: 32, 37, 42) | ffmpeg libx264로 영상 자체를 재인코딩 (프레임 수·fps 유지, `dataset/degraded/crf<값>/`에 캐시) |
| 모션 블러 `blur` | 블러 길이(픽셀) (예: 5, 15, 25) | 가로 방향 선형 평균 필터 |

논문 초안 v2 표 I의 잠정값(팀 검토 전): 해상도 1/2·1/4·1/8, CRF 37·42·47, 블러 6·11·17px. 압축은 O'Byrne 등(2022)의 CRF {22, 32, 37, 42, 47}을 참고할 수 있다.

## 6. 논문 원문(arXiv 2604.09819v1)으로 확인한 점

1. **비교 기준 수치**: 시점 0.343은 σt=1초 기준(Table 3: σt=0.5/1/2 → 0.207/0.343/0.539), 위치 0.596은 정답 사고 프레임을 주는 오라클 보조 방식, σ는 전체 사고 bbox의 평균 정규화 너비·높이(Sec. 5.2 각주 4, Table 5). 스크립트의 `when_s1.0`, `where`가 각각 이 값에 대응한다.
2. **평가 묶음**: 논문은 벤치마크를 IID(테스트 1,520), Geo-aware OOD(테스트 1,573), Zero-shot(2,027 전체를 테스트) 세 가지로 정의한다(Table 2). Molmo 수치가 어느 묶음 기준인지 논문에 명시돼 있지는 않지만, 학습 없는 방법이고 Zero-shot 묶음은 "실제 영상 전체에서 바로 평가"라고 설명하며(Sec. 4), 공식 코드도 2,027개 전체를 돌린다. 그래서 **0.343/0.596은 Zero-shot 묶음(2,027개) 기준으로 보는 것이 맞다(추론)**. 우리가 IID 테스트 1,520개를 쓰면 논문 수치와 평가 묶음이 달라진다.
3. **시점 실패 시 대체값**: Molmo가 한 번도 "yes"를 안 하면 공식 분석 노트북은 예측 시각을 `정답시각 // 2`로 채운다. 논문 본문에는 이 처리가 나오지 않는다. 정답을 쓰는 대체값이라 엄밀하지 않지만 재현을 위해 스크립트도 그대로 따랐다. 예측 실패 비율(`pred_frame`이 비어 있는 행)을 따로 세어 둘 것.
4. **압축 단계 참고**: O'Byrne 등(2022)은 H.264 CRF {22, 32, 37, 42, 47}을 썼고 CRF 42부터 탐지 성능이 뚜렷하게 떨어졌다고 보고했다.
