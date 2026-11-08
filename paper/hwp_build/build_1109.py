#!/usr/bin/env python3
"""11/2 제출 HWP를 읽어 11/9 수정본 레코드(DocInfo, Section0, Section1, 그림)를 만든다.

원본 HWP는 읽기만 한다. 결과 스트림을 OUT 폴더에 쓰고, pack_1109.js로 새 HWP를 만든다.
문장은 paper/HWP_반영순서_1109.md 와 같다 (FFmpeg 6.1.1, libx264 core 164: 11/8 Colab에서 확인).

    python build_1109.py --src 융복합지식학회_논문11_02.hwp --out <폴더> [--fig1-width 0.8] [--drop-table4]
"""
import argparse
import io
import re
import struct as S
import sys
import zlib
from pathlib import Path

import olefile
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hwplib import comp, enc, recs  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FIG = REPO / "paper" / "figures"
P_HDR, P_TXT, P_CS, CTRL, LIST, TABLE = 66, 67, 68, 71, 72, 77
EXT = (1, 2, 3, 11, 12, 14, 15, 16, 17, 18, 21, 22, 23)

# ---------------------------------------------------------------- 문장
ABSTRACT = ("본 논문은 CCTV 영상 열화가 비전-언어 모델(Vision-Language Model, VLM)의 교통사고 시점 판별과 조건부 위치 추정에 미치는 영향을 분석하였다. "
            "ACCIDENT 벤치마크의 실제 CCTV 사고 영상 50개에 해상도 저하, H.264 압축, 모션 블러(Motion Blur)를 각각 3단계로 적용하고, Molmo-7B-D의 점수를 같은 영상의 원본과 짝지어 비교하였다. "
            "원본 점수는 시점 0.370, 위치 0.656이었으며, 95% 신뢰구간 기준으로 원본과 차이가 확인된 결과는 해상도 1/4의 위치 점수(상대 하락률 8.3%) 하나였다. "
            "나머지 조건에서는 영상별 점수 변화가 엇갈려 평균 차이가 작았다.")
INTRO1 = ("도로 CCTV 영상에서 교통사고를 탐지하는 연구는 주로 객체 탐지 모델을 학습시키는 방식으로 진행되어 왔다[1]. "
          "최근에는 VLM에 질의하여 별도의 학습 없이 사고 발생 시점과 위치를 답하게 하는 방식이 제안되었다[2]. "
          "객체 탐지가 차량의 위치와 종류를 찾는 것이라면, 이 방식은 장면을 보고 사고 여부와 사고 지점을 직접 판단한다는 점에서 다르다. "
          "한편 실제 CCTV 영상은 원거리 촬영, 저장과 전송을 위한 압축[3], 차량 움직임에 따른 모션 블러로 화질이 낮은 경우가 많으며, "
          "ACCIDENT 데이터셋[2]의 실제 영상 2,027개 중 53.0%가 화질 등급 Poor 또는 Very Poor로 분류되어 있다. "
          "영상 열화의 영향은 주로 객체 탐지에서 분석되었으며[3][4][5], VLM의 사고 판단에서의 영향은 아직 충분히 알려져 있지 않다.")
INTRO2 = ("이에 본 연구는 영상 열화의 종류와 정도에 따라 VLM의 사고 시점 판별과 조건부 위치 추정 점수가 원본과 달라지는지를 연구 질문으로 하였다. "
          "ACCIDENT의 실제 CCTV 사고 영상 50개에 해상도 저하, H.264 압축, 모션 블러를 각각 3단계로 적용하고, Molmo-7B-D[6]의 두 점수를 같은 영상의 원본과 짝지어 비교하였다. "
          "모델과 질의·채점 절차는 ACCIDENT 공식 코드를 사용하였으며, 열화 적용, 짝지은 부트스트랩 신뢰구간, 미검출 처리 방식에 따른 점수 비교, 영상별 점수 변화 분석은 본 연구에서 추가하였다.")
REL21 = ("ACCIDENT[2]는 실제 도로 CCTV 사고 영상 2,027개와 합성 영상 2,211개로 구성된 벤치마크로, 사고의 발생 시점(When), 위치(Where), 유형(What)을 묻는 세 과제를 정의하였다. "
         "예측은 정답과의 차이가 클수록 점수가 낮아지는 가우시안 유사도로 평가하며, 위치 과제는 정답 사고 프레임을 모델에 주고 위치만 묻는 방식으로 평가하였다. "
         "별도의 학습 없이 사용한 VLM 중에서는 Molmo-7B-D의 점수가 시점 0.343, 위치 0.596으로 가장 높았다[2]. "
         "Molmo[6]는 이미지 속 대상을 좌표로 가리키는(pointing) 데이터로 학습된 공개 VLM이다.")
REL22 = ("영상 열화가 AI 성능에 미치는 영향은 주로 객체 탐지를 대상으로 분석되었다. "
         "O’Byrne 등[3]은 감시 영상을 H.264 코덱의 CRF(Constant Rate Factor) 22~47로 압축하여 객체 탐지 성능을 측정하였고, "
         "Cai 등[4]은 모션 블러를 포함한 악조건에서 차량 탐지 모델의 원본 대비 성능 손실률을 보고하였다. "
         "윤영두·김호[5]는 해상도 축소와 압축에 따른 탐지 신뢰도 변화를, 권오설[7]은 저해상도 영상의 차량 검출 문제를 다루었다. "
         "Poyser 등[8]은 행동 인식에서 압축이 강해질수록 움직임 정보를 쓰는 경로의 성능이 외형 정보를 쓰는 경로보다 크게 낮아짐을 보고하였다. "
         "이들 연구에서 확인된 것은 객체 탐지와 행동 인식에서의 영향이며, 같은 열화가 VLM의 사고 시점 판별과 조건부 위치 추정에 미치는 영향은 본 연구에서 측정한다.")
EQ_TAIL = "본 연구는 식 (1)에 가정한 속도와 노출 시간을 넣어 모션 블러 길이를 정하였다(3.2절)."
METHOD0 = ("본 연구의 전체 분석 절차는 (그림 1)과 같다. 실제 CCTV 사고 영상에 열화를 적용하고, "
           "각 조건에서 Molmo-7B-D의 시점 판별과 조건부 위치 추정 점수를 측정한 뒤 원본 대비 상대 하락률로 비교한다.")
SEC31 = [
    ("실험 영상은 ACCIDENT의 실제 CCTV 영상 2,027개 전체를 난수 시드 0으로 섞은 뒤 앞에서부터 50개를 사용하였다. "
     "50개 영상은 사고 유형, 장면, 낮과 밤, 날씨, 화질, 해상도, 영상 길이, 사고 시점의 분포에서 전체 영상과 통계적으로 유의한 차이를 보이지 않았다(카이제곱·Mann-Whitney 검정, ⟦p9⟧). "
     "측정한 처리 시간으로 추정하면 2,027개 전체로 10개 조건을 실행하는 데 약 290시간이 걸려, 같은 영상 50개로 10개 조건을 모두 실행하는 것을 우선하였다."),
    ("모델은 ACCIDENT에서 별도의 학습 없이 가장 높은 점수를 보인 공개 VLM인 Molmo-7B-D[2][6]를 사용하였다. "
     "본 연구는 모델 간 비교가 아니라 같은 모델에서 열화 조건 간 점수 차이를 보는 것을 목적으로 하므로 다른 VLM은 비교하지 않았다. "
     "질의와 채점은 ACCIDENT 공식 코드(github.com/accidentbench/ACCIDENT)의 baselines/llm을 가져와 사용하였고, 유형 분류 호출을 빼고 열화 적용 단계를 추가하였다. "
     "모델은 Hugging Face의 allenai/Molmo-7B-D-0924 체크포인트를 transformers 4.53.2로 불러와 별도의 학습 없이 사용하였으며, 응답은 최대 512토큰까지 생성하였다."),
    ("시점 판별은 영상에서 0.5초 간격으로 뽑은 프레임마다 사고 여부를 묻고, 처음 사고라고 답한 프레임의 시각을 예측 시점으로 하는 프레임 단위 판별이다. "
     "시점 판별에서 모든 프레임에 사고가 없다고 답한 경우는 미검출로 기록하고, ACCIDENT 공식 코드와 같이 정답 사고 시각의 절반을 예측 시각으로 넣어 채점에 포함하였다. "
     "조건부 위치 추정은 정답 사고 프레임 한 장을 주고 사고 지점을 점으로 가리키게 하는 것으로, 사고 시점을 이미 아는 조건에서의 위치 추정이다. "
     "이는 ACCIDENT의 위치 평가 방식[2]을 따른 것으로, 시점 판별의 오류가 위치 점수에 섞이지 않게 한다. "
     "위치 추정에서 점을 제시하지 않은 경우는 화면 중앙을 예측 위치로 하였다(그림 1). "
     "두 과제는 따로 평가하므로 결과를 처음부터 사고를 찾아 위치까지 추정하는 시스템의 성능으로 해석하지 않는다."),
    ("유형 분류(What)는 다루지 않았다. ACCIDENT에서 Molmo-7B-D의 유형 정확도(0.271)는 가장 많은 유형만 답하는 기준선(0.335)보다 낮아[2], "
     "열화 전후의 변화를 비교하기 어렵다고 판단하였다. 정확도가 낮은 원인은 본 연구에서 분석하지 않았다. "
     "유형 분류는 시점·위치와 별도의 질의로 수행되므로 이를 빼도 두 점수는 달라지지 않는다."),
]
SEC32 = [
    ("열화는 CCTV 영상의 촬영과 전송 과정에서 생기는 해상도 저하, H.264 압축, 모션 블러의 세 가지로 하고, 각각 3단계를 적용하였다(표 1). "
     "CRF는 감시 영상의 객체 탐지 실험[3]에서 사용한 다섯 단계(22, 32, 37, 42, 47) 중 높은 세 단계로 정하였다. "
     "이는 실험 범위를 정하기 위해 참고한 것이며, [3]의 결과가 VLM의 사고 판단에 그대로 적용된다고 보지는 않는다."),
    ("해상도 저하는 각 프레임을 1/2, 1/4, 1/8 크기로 바이큐빅 보간하여 축소한 뒤 같은 보간법으로 원래 크기까지 다시 확대한다. "
     "따라서 모델에 들어가는 영상의 크기는 같고 세부 정보만 줄어든다. "
     "이 방식은 저해상도 카메라로 직접 촬영한 영상과 달리 센서 잡음이나 렌즈 특성을 포함하지 않는다. "
     "1/8 단계는 축소된 영상에서 사고 영역이 가로 약 12, 세로 약 7픽셀(중앙값)까지 줄어드는 극단 조건으로 추가하였다."),
    ("H.264 압축은 Google Colab의 FFmpeg 6.1.1과 libx264(core 164) 인코더로 영상 전체를 다시 인코딩하며, 프리셋은 medium, 화소 형식은 yuv420p로 고정하고 CRF만 37, 42, 47로 바꾼다. "
     "원본 영상도 이미 온라인 배포를 위해 압축된 상태이므로, 이 조건은 기존 압축 위에 더해진 재압축이다."),
    ("모션 블러는 길이 ⟦L9⟧ 픽셀의 수평 방향 균일 커널을 각 프레임에 적용한다. "
     "⟦L9⟧은 차량 속도 30, 60, 90 km/h, 노출 시간 1/30 s, 픽셀당 도로 거리 0.05 m를 가정하여 식 (1)로 구한 5.6, 11.1, 16.7을 반올림한 값이다. "
     "노출 시간은 초당 30프레임 영상에서 가능한 최대값이며, 픽셀당 거리는 영상마다 다르므로 대표값으로 정하였다. "
     "이 가정은 차량의 실제 이동 방향과 카메라와의 거리를 반영하지 않으며, 같은 길이의 블러가 저해상도 영상에는 상대적으로 더 강하게 작용한다. "
     "세 열화는 한 번에 한 종류씩만 적용하고, 원본을 포함한 10개 조건에서 같은 영상 50개를 평가한다."),
]
SEC33_OLD_TAIL = "조건 간 차이는 같은 영상의 원본·열화 결과를 짝지은 부트스트랩으로 95% 신뢰구간을 구하여 확인한다."
SEC33_NEW_TAIL = ("95% 신뢰구간은 같은 영상의 원본 결과와 열화 결과를 한 쌍으로 묶은 채 영상 50개를 복원 추출하는 부트스트랩을 2,000회 반복하고, "
                  "반복마다 계산한 RPD 분포의 2.5번째와 97.5번째 백분위로 구하였다. 신뢰구간이 0을 포함하지 않는 경우만 원본과 차이가 있다고 해석한다.")
NOTE_T2 = "위치 점수의 ⟦sx8⟧, ⟦sy8⟧는 ACCIDENT 공식 코드와 같이 실제 영상 2,027개 사고 상자의 평균 너비와 높이(정규화 좌표 0.127, 0.119)이다."
SEC41 = ("원본 영상 50개에서 Molmo-7B-D의 점수는 시점 0.370, 위치 0.656이었다(표 3). "
         "(표 3)의 ACCIDENT 보고값은 같은 모델과 공식 질의·채점 코드를 사용했지만 실제 영상 2,027개 전체에 대한 값이므로, 표본이 다른 참고 비교로만 제시한다. "
         "원본에서 미검출 영상은 8개(16%)였다. 이 영상들이 공식 대체값으로 받은 점수는 평균 시점 점수에 0.030을 더하였으며, 미검출을 0점으로 처리하면 시점 점수는 0.341이 된다.")
NOTE_T3 = "ACCIDENT 보고값은 실제 영상 2,027개 전체 기준이며, 본 연구 값은 그중 50개 기준인 참고 비교이다."
SEC42 = [
    ("(그림 3)은 열화 단계별 평균 점수와 영상별 점수 변화를, (표 4)는 원본 점수, 열화 점수, 차이, RPD를 나타낸다. "
     "(표 4)의 차이와 RPD는 반올림 전 평균으로 계산하였다. "
     "(그림 2)는 한 영상의 정답 사고 프레임에 10개 조건을 적용한 예이다. 이 영상에서는 해상도 1/8과 CRF 42에서 예측 위치가 정답 영역을 벗어났다."),
    ("95% 신뢰구간이 0을 포함하지 않은 결과는 해상도 1/4의 위치 점수(0.656 → 0.602, RPD 8.3%) 하나였다. "
     "나머지 17개 결과는 영상 50개에서 원본과의 차이를 확인할 수 없었다."),
    ("평균만 보면 위치 점수는 9개 조건 중 8개에서 원본보다 0.017~0.055 낮았다. "
     "시점 점수는 해상도 1/8(0.298)과 블러 17 px(0.325)에서 가장 낮았고, 같은 두 조건에서 미검출이 12개(24%)로 늘었다. "
     "그러나 이 두 조건의 신뢰구간도 0을 포함한다. H.264 압축 조건의 시점 점수 평균은 CRF 47까지 원본과 0.02 이내였다."),
    ("열화 단계가 강해져도 점수가 일정하게 낮아지지는 않았다. 예를 들어 해상도 1/4의 시점 점수는 1/2보다 높았다. "
     "(그림 3)의 (c)를 보면 이 조건에서 시점 점수가 0.1 이상 오른 영상은 8개, 내린 영상은 9개로 증감이 상쇄되어 평균 변화가 작았다. "
     "영상별 점수는 0이나 1 근처에 몰려 있어, 한 영상의 판별 결과가 바뀌면 평균이 크게 움직인다. "
     "어떤 영상 특성이 이러한 증감을 만드는지는 이번 실험에서 확인하지 않았다."),
]
NOTE_T4 = "괄호 안은 원본과의 차이. 차이와 RPD는 반올림 전 평균으로 계산. [ ]는 짝지은 부트스트랩 95% 신뢰구간."
CAP_FIG2 = "(그림 2) 같은 사고 프레임의 조건별 비교 (사고 영역 주변을 잘라 표시, 번호판 흐림 처리. 초록 상자: 정답 영역, ×: 예측 위치, ⟦S9⟧: 공간 유사도. 출처: ACCIDENT[2], CC BY-NC-SA 4.0)"
CAP_FIG3 = "(그림 3) 열화 단계별 ⟦T9⟧와 ⟦S9⟧. (a)(b) 평균과 95% 신뢰구간(회색 띠: 원본의 신뢰구간), (c)(d) 영상별 점수 변화(열화 − 원본). 상자: 사분위, 수염: 5~95%, ◇: 평균"
CONCL = ("본 논문은 ACCIDENT의 실제 CCTV 사고 영상 50개에 해상도 저하, H.264 압축, 모션 블러를 각각 3단계로 적용하고, "
         "Molmo-7B-D의 프레임 단위 시점 판별과 조건부 위치 추정 점수를 같은 영상의 원본과 짝지어 비교하였다. "
         "평균 하락률과 함께 미검출 처리 방식과 영상별 점수 분포를 보고하여, 평균만으로는 드러나지 않는 변화를 확인할 수 있게 하였다. "
         "그 결과 95% 신뢰구간 기준으로 원본과 차이가 확인된 것은 해상도 1/4의 위치 점수뿐이었고, 다른 조건에서는 영상별 증감이 엇갈려 평균 차이가 작았다. "
         "본 연구는 열화를 한 번에 한 종류씩만 적용하였으므로, 여러 열화가 함께 나타나는 실제 CCTV 환경으로 결과를 그대로 일반화할 수는 없다. "
         "향후에는 복합 열화 조건에서 같은 방법으로 비교를 수행할 계획이다.")


# ---------------------------------------------------------------- 레코드 도구
def txt(b):
    out, i = [], 0
    while i < len(b):
        c = S.unpack('<H', b[i:i + 2])[0]
        if c < 32 and c not in (13, 10, 9):
            out.append('\x00'); i += 16 if c in EXT else 2; continue
        out.append(chr(c)); i += 2
    return ''.join(out)


def span(R, i):
    j = i + 1
    while j < len(R) and R[j][1] > R[i][1]:
        j += 1
    return j


def find(R, needle, level=0, start=0):
    for i in range(start, len(R)):
        if R[i][0] == P_HDR and R[i][1] == level and i + 1 < len(R) and R[i + 1][0] == P_TXT \
                and needle in txt(R[i + 1][2]):
            return i
    raise KeyError(needle)


def hdr_with(tmpl, nch, mask=0, ncs=1, last=None):
    h = bytearray(tmpl)
    old = S.unpack('<I', h[:4])[0]
    lastbit = (old & 0x80000000) if last is None else (0x80000000 if last else 0)
    h[0:4] = S.pack('<I', nch | lastbit)
    h[4:8] = S.pack('<I', mask)
    h[12:14] = S.pack('<H', ncs)
    h[14:16] = S.pack('<H', 0)
    h[16:18] = S.pack('<H', 0)  # 줄 배치 정보 없음 → 한글이 다시 배치
    return bytes(h)


EQ = __import__('json').loads((Path(__file__).resolve().parent / 'eqsizes_1109.json').read_text(encoding='utf-8'))
EQ_MARK = re.compile('⟦(\\w+)⟧')


def new_para(tmpl_hdr, text, cs, level=0, last=False):
    """text 안의 ⟦키⟧는 수식 입력기 개체(eqsizes_1109.json, @rhwp/core로 크기 계산)로 넣는다 (교수님 피드백: 기호는 수식 입력기)."""
    if EQ_MARK.search(text):
        parts = EQ_MARK.split(text)
        body, n, kids = bytearray(), 0, []
        for i, s in enumerate(parts):
            if i % 2 == 0:
                body += s.encode('utf-16le'); n += len(s)
            else:
                e = EQ[s]
                body += S.pack('<H', 11) + b'deqe' + b'\0' * 8 + S.pack('<H', 11); n += 8
                c = bytearray(bytes.fromhex(e['ctrl'])); c[36:40] = S.pack('<I', nid())
                kids += [[CTRL, level + 1, bytes(c)], [88, level + 2, bytes.fromhex(e['eq'])]]
        body += '\r'.encode('utf-16le'); n += 1
        return [[P_HDR, level, hdr_with(tmpl_hdr, n, mask=1 << 11, last=last)],
                [P_TXT, level + 1, bytes(body)],
                [P_CS, level + 1, S.pack('<II', 0, cs)]] + kids
    body = (text + '\r').encode('utf-16le')
    return [[P_HDR, level, hdr_with(tmpl_hdr, len(body) // 2, last=last)],
            [P_TXT, level + 1, body],
            [P_CS, level + 1, S.pack('<II', 0, cs)]]


def first_cs(R, i):
    j = i + 1
    while R[j][0] != P_CS:
        j += 1
    return S.unpack('<I', R[j][2][4:8])[0]


def replace_paras(R, i, texts, indent="  "):
    """문단 i(머리 레코드)를 texts의 문단들로 바꾼다. 문단 모양·글자 모양은 원래 문단 것."""
    tmpl, cs, L = R[i][2], first_cs(R, i), R[i][1]
    new = []
    for t in texts:
        new += new_para(tmpl, indent + t, cs, L)
    R[i:span(R, i)] = new


def set_text_keep_prefix(R, i, text, nprefix_bytes):
    """앞의 컨트롤(쪽 번호 등)은 그대로 두고 문장만 바꾼다."""
    t = R[i + 1][2]
    body = t[:nprefix_bytes] + (text + '\r').encode('utf-16le')
    R[i + 1][2] = body
    mask = S.unpack('<I', R[i][2][4:8])[0]
    R[i][2] = hdr_with(R[i][2], len(body) // 2, mask=mask)
    j = i + 2
    cs = S.unpack('<I', R[j][2][4:8])[0]
    R[j][2] = S.pack('<II', 0, cs)


def txt_hdr(R, needle):
    k = next(k for k, r in enumerate(R) if r[0] == P_TXT and needle in txt(r[2]))
    assert R[k - 1][0] == P_HDR
    return k - 1


def table_paras(R):
    """본문 표 1~4가 든 문단 머리 (제목·저자 블록의 표는 제외: '(표 1)' 캡션 뒤만)."""
    start = find(R, '(표 1)')
    return [k for k in range(start, len(R)) if R[k][0] == P_HDR and R[k][1] == 0
            and any(R[m][0] == CTRL and R[m][2][:4] == b' lbt' for m in range(k, span(R, k)))]


def insert_after(R, i, recs_):
    j = span(R, i)
    R[j:j] = recs_


# ---------------------------------------------------------------- 그림
def png_bytes(path):
    im = Image.open(path).convert('RGBA')
    bg = Image.new('RGB', im.size, 'white')
    bg.paste(im, mask=im.split()[3])
    bio = io.BytesIO()
    bg.save(bio, 'PNG', optimize=True, dpi=(600, 600))  # 그림은 실제 인쇄 크기로 그렸다
    return bio.getvalue(), im.size


INST = [0x5A100000]


def nid():
    INST[0] += 13
    return INST[0]


def picture_recs(R, pic_hdr, binid, W, H):
    """그림 문단(pic_hdr부터)을 복제해 binid, 크기 W×H(HWPUNIT)로 바꾼다."""
    grp = [list(x) for x in R[pic_hdr:span(R, pic_hdr)]]
    for r in grp:
        t, p = r[0], bytearray(r[2])
        if t == CTRL and p[:4] == b' osg':
            p[16:20] = S.pack('<I', W); p[20:24] = S.pack('<I', H); p[36:40] = S.pack('<I', nid())
        elif t == 76:
            p[20:36] = S.pack('<IIII', W, H, W, H); p[42:50] = S.pack('<ii', W // 2, H // 2)
            m = list(S.unpack('<18d', p[52:196])); m[6] = 1.0; m[10] = 1.0; p[52:196] = S.pack('<18d', *m)
        elif t == 85:
            p[12:44] = S.pack('<8i', 0, 0, W, 0, W, H, 0, H); p[44:60] = S.pack('<4i', 0, 0, W, H)
            p[71:73] = S.pack('<H', binid); p[74:78] = S.pack('<I', nid()); p[82:90] = S.pack('<II', W, H)
        r[2] = bytes(p)
    return grp


# ---------------------------------------------------------------- 표
def table_span(R, para_hdr):
    return para_hdr, span(R, para_hdr)


def cells(R, a, b):
    """표 문단 범위 [a,b)에서 셀(LIST_HEADER) 묶음 (시작, 끝, col, row)."""
    ctrl_lvl = next(R[k][1] for k in range(a, b) if R[k][0] == CTRL)
    idx = [k for k in range(a, b) if R[k][0] == LIST and R[k][1] == ctrl_lvl + 1]
    out = []
    for n, k in enumerate(idx):
        end = idx[n + 1] if n + 1 < len(idx) else b
        col, row = S.unpack('<HH', R[k][2][8:12])
        out.append((k, end, col, row))
    return out


def drop_table1_basis(R, para_hdr):
    a, b = table_span(R, para_hdr)
    cs_ = cells(R, a, b)
    for k, end, col, row in reversed(cs_):
        if col == 4:
            del R[k:end]
    a, b = table_span(R, para_hdr)
    widths = {0: 10200, 1: 4500, 2: 4500, 3: 4500}
    for k, end, col, row in cells(R, a, b):
        p = bytearray(R[k][2]); p[16:20] = S.pack('<I', widths[col]); p[34:38] = S.pack('<I', widths[col]); R[k][2] = bytes(p)
    t = next(k for k in range(a, b) if R[k][0] == TABLE)
    p = bytearray(R[t][2]); nr = S.unpack('<H', p[4:6])[0]
    p[6:8] = S.pack('<H', 4)
    p = p[:18] + b''.join(S.pack('<H', 4) for _ in range(nr)) + p[18 + 2 * nr:]
    R[t][2] = bytes(p)


def add_table4_diffs(R, para_hdr, diffs):
    """표 4의 T·S 점수 칸(1열, 3열)에 원본과의 차이를 둘째 줄로 넣는다."""
    a, b = table_span(R, para_hdr)
    for k, end, col, row in reversed(cells(R, a, b)):
        if col not in (1, 3) or row < 2:
            continue
        d = diffs[row - 2][0 if col == 1 else 1]
        h = k + 1  # 셀 안 첫 문단 머리
        tmpl, cs, L = R[h][2], first_cs(R, h), R[h][1]
        val = txt(R[h + 1][2]).rstrip('\r')
        new = new_para(tmpl, val, cs, L, last=False) + new_para(tmpl, f"({d})", cs, L, last=True)
        R[h:end] = new
        p = bytearray(R[k][2]); p[0:2] = S.pack('<H', 2); R[k][2] = bytes(p)


def set_cells(R, para_hdr, want):
    """표의 (row, col) 칸 글자를 lines(문단 목록)로 바꾼다. 첫 문단의 모양을 따른다."""
    a, b = table_span(R, para_hdr)
    for k, end, col, row in reversed(cells(R, a, b)):
        if (row, col) not in want:
            continue
        lines = want[(row, col)]
        h = k + 1
        tmpl, cs, L = R[h][2], first_cs(R, h), R[h][1]
        new = []
        for n, s in enumerate(lines):
            new += new_para(tmpl, s, cs, L, last=n == len(lines) - 1)
        R[h:end] = new
        p = bytearray(R[k][2]); p[0:2] = S.pack('<H', len(lines)); R[k][2] = bytes(p)


def set_table4(R, para_hdr):
    """표 4의 점수·차이·RPD·CI 칸을 results_n50/table4_compact.md 값으로 다시 쓴다 (원본 HWP의 옛 값 대체)."""
    fix = lambda s: s.replace('-', '−')  # noqa: E731
    rows = [[x.strip() for x in line.strip('|').split('|')]
            for line in (REPO / "accident-pilot" / "results_n50" / "table4_compact.md").read_text(encoding="utf-8").splitlines()[2:]]
    assert len(rows) == 9
    split = lambda s: [fix(s.split(' [')[0]), fix('[' + s.split(' [')[1])]  # noqa: E731  "4.9 [-2.4, 14.6]"
    want = {1: [[rows[0][2]], ['–'], [rows[0][6]], ['–']]}
    for i, c in enumerate(rows):
        want[i + 2] = [[c[3], f"({fix(c[4])})"], split(c[5]), [c[7], f"({fix(c[8])})"], split(c[9])]
    a, b = table_span(R, para_hdr)
    for k, end, col, row in reversed(cells(R, a, b)):
        if row not in want or col == 0:
            continue
        lines = want[row][col - 1]
        h = k + 1
        tmpl, cs, L = R[h][2], first_cs(R, h), R[h][1]
        new = []
        for n, s in enumerate(lines):
            new += new_para(tmpl, s, cs, L, last=n == len(lines) - 1)
        R[h:end] = new
        p = bytearray(R[k][2]); p[0:2] = S.pack('<H', len(lines)); R[k][2] = bytes(p)


def table4_diffs():
    rows = []
    for line in (REPO / "accident-pilot" / "results_n50" / "table4_compact.md").read_text(encoding="utf-8").splitlines()[2:]:
        c = [x.strip() for x in line.strip('|').split('|')]
        fix = lambda s: s.replace('-', '−')  # noqa: E731
        rows.append((fix(c[4]), fix(c[8])))
    assert len(rows) == 9
    return rows


NOTE_PREFIX = ('위치 점수의', 'ACCIDENT 보고값은', '괄호 안은')
GAP_PS, GAP_CS = 21, 22        # 양쪽 정렬 줄간격 100%, 한양신명조 7pt → 그림·표 앞뒤 작은 빈 줄
NOTE_PS, NOTE_CS = 20, 4       # 왼쪽 정렬 130%(표 캡션과 같은 모양, 별도 번호), 한양중고딕 8pt → 표 아래 주석
KEEP_NEXT_PS = (8, 12)         # 표 캡션, 그림·표·수식 문단: '다음 문단과 함께'(속성1 bit 17) → 캡션과 표·그림이 떨어지지 않게
HEAD_PS = 13                   # 소제목 문단 모양 (4.1만 원본 양식에서 10이었음)


def tidy_layout(R):
    """그림·표·수식 앞뒤 빈 줄을 같은 규칙으로 맞춘다.
    - 블록(그림+캡션, 표 캡션+표+주석, 수식) 앞뒤에 7pt 빈 줄 하나. 장 구분 빈 줄(ps 3) 옆에는 넣지 않는다.
    - 그 밖의 본문 빈 줄은 지운다. 표 주석은 8pt 중고딕, 4.1 소제목은 다른 소제목과 같은 문단 모양.
    """
    def info(k):
        e = span(R, k)
        ps = S.unpack('<H', R[k][2][8:10])[0]
        t = next((txt(R[j][2]) for j in range(k + 1, e) if R[j][0] == P_TXT and R[j][1] == 1), '')
        ctrls = {R[j][2][:4][::-1] for j in range(k + 1, e) if R[j][0] == CTRL and R[j][1] == 1}
        vis = t.replace('\x00', '').strip()
        if b'gso ' in ctrls and not vis:
            kind = 'pic'
        elif b'tbl ' in ctrls and not vis:
            kind = 'tbl'
        elif ps == 12 and vis.endswith('(1)'):
            kind = 'eq'
        elif vis.startswith('(그림'):
            kind = 'fcap'
        elif vis.startswith('(표'):
            kind = 'tcap'
        elif vis.startswith(NOTE_PREFIX):
            kind = 'note'
        elif not vis and not ctrls:
            kind = 'sec' if ps == 3 else 'blank'
        else:
            kind = 'text'
        return kind, e, ps, vis

    tops = [k for k, r in enumerate(R) if r[0] == P_HDR and r[1] == 0]
    items = [(k,) + info(k) for k in tops[1:]]          # 0번은 제목 표
    gap_src = next(k for k, kind, e, ps, vis in items if kind == 'blank')
    g_end = span(R, gap_src)
    gh = bytearray(R[gap_src][2]); gh[8:10] = S.pack('<H', GAP_PS); gh[16:18] = S.pack('<H', 0)
    gh[0:4] = S.pack('<I', S.unpack('<I', gh[0:4])[0] & 0x7FFFFFFF)
    gap = [[P_HDR, 0, bytes(gh)]] + [[r[0], r[1], S.pack('<II', 0, GAP_CS) if r[0] == P_CS else r[2]]
                                     for r in R[gap_src + 1:g_end] if r[0] != 69]

    # 블록 묶기
    seq, i = [], 0
    while i < len(items):
        k, kind, e, ps, vis = items[i]
        if kind == 'pic' and i + 1 < len(items) and items[i + 1][1] == 'fcap':
            seq.append(('block', [items[i], items[i + 1]])); i += 2; continue
        if kind == 'tcap' and i + 1 < len(items) and items[i + 1][1] == 'tbl':
            blk = [items[i], items[i + 1]]; i += 2
            if i < len(items) and items[i][1] == 'note':
                blk.append(items[i]); i += 1
            seq.append(('block', blk)); continue
        if kind == 'eq':
            seq.append(('block', [items[i]])); i += 1; continue
        if kind != 'blank':
            seq.append((kind, [items[i]]))
        i += 1

    out = [list(r) for r in R[:items[0][0]]]
    prev = 'sec'
    for n, (kind, its) in enumerate(seq):
        if kind == 'block' and prev not in ('sec', 'gap'):
            out += [list(r) for r in gap]
        for k, knd, e, ps, vis in its:
            recs_ = [list(r) for r in R[k:e]]
            if knd == 'note':
                h = bytearray(recs_[0][2]); h[8:10] = S.pack('<H', NOTE_PS); recs_[0][2] = bytes(h)
                for r in recs_:
                    if r[0] == P_CS and r[1] == 1:
                        b = bytearray(r[2])
                        for q in range(0, len(b), 8):
                            b[q + 4:q + 8] = S.pack('<I', NOTE_CS)
                        r[2] = bytes(b)
            if knd == 'text' and ps == 10 and vis.startswith('4.1'):
                h = bytearray(recs_[0][2]); h[8:10] = S.pack('<H', HEAD_PS); recs_[0][2] = bytes(h)
            out += recs_
        prev = kind
        if kind == 'block':
            nxt = seq[n + 1][0] if n + 1 < len(seq) else 'sec'
            if nxt != 'sec':
                out += [list(r) for r in gap]
                prev = 'gap'
    out += [list(r) for r in R[span(R, items[-1][0]):]]
    R[:] = out


def drop_blank_runs(R, min_run=3):
    """글자도 컨트롤도 없는 빈 문단이 min_run개 이상 이어진 곳을 지운다 (원본에 엔터로 띄운 자리)."""
    tops = [k for k, r in enumerate(R) if r[0] == P_HDR and r[1] == 0]
    blank = [all(R[j][0] not in (P_TXT, CTRL) for j in range(k + 1, span(R, k))) for k in tops]
    cut, i = [], 0
    while i < len(tops):
        j = i
        while j < len(tops) and blank[j]:
            j += 1
        if j - i >= min_run:
            cut.append((tops[i], tops[j] if j < len(tops) else len(R), j - i))
        i = max(j, i + 1)
    for a, b, _ in reversed(cut):
        del R[a:b]
    return [n for _, _, n in cut]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--fig1-width', type=float, default=1.0, help='그림 1 폭 (기존 그림 폭 대비)')
    ap.add_argument('--fig2-width', type=float, default=1.0)
    ap.add_argument('--fig3-width', type=float, default=1.0)
    ap.add_argument('--drop-table4', action='store_true')
    ap.add_argument('--trim-blank', action='store_true', help='엔터로 띄운 빈 문단 묶음(3개 이상 연속) 삭제')
    ap.add_argument('--tidy', action='store_true', help='그림·표·수식 앞뒤 빈 줄 규칙 통일, 표 주석 8pt')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    f = olefile.OleFileIO(str(args.src))
    rd = lambda n: [list(x) for x in recs(zlib.decompress(f.openstream(n).read(), -15))]  # noqa: E731
    R, R0, D = rd('BodyText/Section1'), rd('BodyText/Section0'), rd('DocInfo')

    # 신청서 요약 (Section0)
    i = txt_hdr(R0, '본 논문은 CCTV 영상의')
    set_text_keep_prefix(R0, i, ' ' + ABSTRACT, 0)

    # 논문 요약 (앞의 쪽 번호 컨트롤 16바이트 유지)
    i = txt_hdr(R, '본 논문은 CCTV 영상의')
    set_text_keep_prefix(R, i, ' ' + ABSTRACT, 16)

    replace_paras(R, find(R, '교통사고를 신속하게 파악하는'), [INTRO1])
    replace_paras(R, find(R, '이에 본 연구는 ACCIDENT 벤치마크'), [INTRO2])
    replace_paras(R, find(R, 'ACCIDENT[2]는 실제 도로 CCTV에서'), [REL21])
    replace_paras(R, find(R, '영상 열화가 AI 성능에 미치는 영향은'), [REL22])

    # 식 (1) 아래 문단: 수식 컨트롤은 두고 마지막 문장만
    i = find(R, '본 연구는 식 (1)을 이용하여')
    t = R[i + 1][2]
    cut = t.find('본 연구는 식 (1)을'.encode('utf-16le'))
    body = t[:cut] + (EQ_TAIL + '\r').encode('utf-16le')
    R[i + 1][2] = body
    R[i][2] = hdr_with(R[i][2], len(body) // 2, mask=S.unpack('<I', R[i][2][4:8])[0],
                       ncs=S.unpack('<H', R[i][2][12:14])[0])

    replace_paras(R, find(R, '본 연구의 전체 분석 절차는'), [METHOD0])
    replace_paras(R, find(R, '실험에는 ACCIDENT의 실제 CCTV 사고 영상 중'), SEC31)
    replace_paras(R, find(R, '열화는 카메라 촬영과 영상 전송 과정에서'), SEC32)
    drop_table1_basis(R, table_paras(R)[0])

    i = find(R, '시점과 위치 판단은 ACCIDENT의 평가 방식')
    old = txt(R[i + 1][2]).rstrip('\r')
    assert SEC33_OLD_TAIL in old
    replace_paras(R, i, [old.strip().replace(SEC33_OLD_TAIL, SEC33_NEW_TAIL)])

    body_tmpl, body_cs = R[find(R, '본 연구의 전체 분석 절차는')][2], 3
    tbl_paras = table_paras(R)
    # 표 순서: 0 표1, 1 표2, 2 표3, 3 표4. 뒤에서부터 넣어야 앞 인덱스가 안 바뀐다
    set_table4(R, tbl_paras[3])
    set_cells(R, tbl_paras[2], {(3, 2): ['0.370'], (3, 3): ['0.656']})  # 표 3 본 연구 행 (A100 재실행)
    insert_after(R, tbl_paras[3], new_para(body_tmpl, '  ' + NOTE_T4, body_cs))
    insert_after(R, tbl_paras[2], new_para(body_tmpl, '  ' + NOTE_T3, body_cs))
    insert_after(R, tbl_paras[1], new_para(body_tmpl, '  ' + NOTE_T2, body_cs))

    i = find(R, '4.1 원본 성능')
    replace_paras(R, i, ['4.1 원본 성능'], indent='')
    replace_paras(R, find(R, '열화를 적용하기 전에 원본 영상에서'), [SEC41])
    replace_paras(R, find(R, '표 4는 영상 50개에 대해 원본과'), SEC42)
    replace_paras(R, find(R, '본 논문은 실제 CCTV 사고 영상에 해상도 저하'), [CONCL])

    # ---- 그림
    pic_hdr = [k for k in range(len(R)) if R[k][0] == P_HDR and R[k][1] == 0
               and any(R[m][0] == CTRL and R[m][2][:4] == b' osg' for m in range(k, span(R, k)))][0]
    W0 = S.unpack('<I', next(r for r in R[pic_hdr:span(R, pic_hdr)] if r[0] == CTRL)[2][16:20])[0]
    cap_hdr = find(R, '(그림 1) 전체 분석 절차')
    cap_tmpl, cap_cs = R[cap_hdr][2], first_cs(R, cap_hdr)

    figs = {}
    for key, name, width in (('fig1', 'fig1_분석절차.png', args.fig1_width),
                             ('fig2', 'fig2_열화비교_단폭.png', args.fig2_width),
                             ('fig3', 'fig3_성능변화.png', args.fig3_width)):
        data, (w, h) = png_bytes(FIG / name)
        Wd = int(W0 * width)
        figs[key] = (data, Wd, int(Wd * h / w))

    # 그림 1: 같은 BinData(2)에 새 그림, 크기만 바꿈
    R[pic_hdr:span(R, pic_hdr)] = picture_recs(R, pic_hdr, 2, figs['fig1'][1], figs['fig1'][2])
    pic_tmpl_hdr = pic_hdr

    def fig_block(binid, key, cap):
        return picture_recs(R, pic_tmpl_hdr, binid, figs[key][1], figs[key][2]) + new_para(cap_tmpl, cap, cap_cs)

    tbl_paras = table_paras(R)
    t4_note = span(R, tbl_paras[3])  # 표 4 주석 문단 머리
    insert_after(R, t4_note, fig_block(4, 'fig3', CAP_FIG3))
    last42 = find(R, '열화 단계가 강해져도 점수가 일정하게')
    insert_after(R, last42, fig_block(3, 'fig2', CAP_FIG2))

    if args.drop_table4:
        cap4 = find(R, '(표 4) 열화 조건별')
        tp = [k for k in range(cap4, len(R)) if R[k][0] == P_HDR and R[k][1] == 0][1]
        note = span(R, tp)
        del R[cap4:span(R, note)]
        replace_paras(R, find(R, '그림 3)은 열화 단계별 평균 점수와'),
                      [SEC42[0].replace('(그림 3)은 열화 단계별 평균 점수와 영상별 점수 변화를, (표 4)는 원본 점수, 열화 점수, 차이, RPD를 나타낸다. '
                                        '(표 4)의 차이와 RPD는 반올림 전 평균으로 계산하였다. ',
                                        '(그림 3)은 열화 단계별 평균 점수와 95% 신뢰구간, 영상별 점수 변화를 나타낸다. RPD와 차이는 반올림하기 전의 평균으로 계산하였다. ')])

    if args.trim_blank:
        print('빈 문단 삭제:', drop_blank_runs(R))
    if args.tidy:
        tidy_layout(R)

    if args.tidy:
        pss = [k for k, r in enumerate(D) if r[0] == 25]
        for n in KEEP_NEXT_PS:
            b = bytearray(D[pss[n]][2]); b[0:4] = S.pack('<I', S.unpack('<I', b[0:4])[0] | (1 << 17)); D[pss[n]][2] = bytes(b)

    # ---- DocInfo: BinData 3, 4 추가
    bins = [k for k, r in enumerate(D) if r[0] == 18]
    lvl = D[bins[-1]][1]
    newb = [[18, lvl, S.pack('<HHH', 0x0001, n, 3) + 'png'.encode('utf-16le')] for n in (3, 4)]
    D[bins[-1] + 1:bins[-1] + 1] = newb
    idm_i = next(k for k, r in enumerate(D) if r[0] == 17)
    p = bytearray(D[idm_i][2]); p[0:4] = S.pack('<I', S.unpack('<I', p[0:4])[0] + 2); D[idm_i][2] = bytes(p)

    out = args.out
    (out / 'DocInfo.bin').write_bytes(comp(enc(D)))
    (out / 'Section0.bin').write_bytes(comp(enc(R0)))
    (out / 'Section1.bin').write_bytes(comp(enc(R)))
    for key, n in (('fig1', 2), ('fig2', 3), ('fig3', 4)):
        (out / f'BIN000{n}.png.bin').write_bytes(comp(figs[key][0]))
    print('ok records', len(R), {k: (v[1], v[2]) for k, v in figs.items()})


if __name__ == '__main__':
    main()
