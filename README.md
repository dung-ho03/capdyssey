# Capdyssey | 손목 PPG 심방세동 의심 구간 선별 시제품

한림대학교 캡스톤디자인 팀 **캡디세이**는 손목에서 측정한 맥파(PPG)와 움직임을 PC로 전송하고, 10초마다 심방세동(AF) 의심 구간을 표시하는 연구용 시제품을 만들었습니다. 실제 장치에서 센서 수집, Colab 분석, 화면 표시, 기록 저장까지 확인했습니다.

## 바로 실행하기

| 하고 싶은 일 | 시작 위치 |
|---|---|
| 센서를 연결해 실시간으로 측정 | [Windows 앱 설치·센서 연결 안내](models/Capdyssey_realtime_v1/README.md) · [Arduino 펌웨어](models/Capdyssey_realtime_v1/firmware/capdyssey_usb/capdyssey_usb.ino) · [실시간 분석 서버를 Colab에서 열기](https://colab.research.google.com/github/dung-ho03/capdyssey/blob/main/models/Capdyssey_realtime_v1/Colab_realtime_server.ipynb) |
| 저장된 PPG로 통합 AI 실행 | [팀원용 최종본 Colab 열기](https://colab.research.google.com/github/dung-ho03/capdyssey/blob/main/notebooks/12_quality_af_v2_final%28%EC%B5%9C%EC%A2%85%EB%B3%B8%29.ipynb) · [학습된 모델과 실행 코드](models/quality_af_v2_final/) · [모델 ZIP](<models/Capdyssey_quality_af_v2(최종본).zip>) |

실시간 측정은 Windows에서 `setup_windows.bat`로 최초 설치 후 `run_desktop.bat`로 시작합니다. Arduino 시리얼 모니터를 닫고 앱에서 예시 모드를 해제한 뒤 장치의 USB 포트를 선택하세요. 분석 주소를 비우면 파형 수신과 저장부터 확인할 수 있습니다. AF 판정까지 보려면 실시간 패키지 안내에 따라 Colab GPU 서버를 실행하고 출력된 주소와 비밀번호를 앱에 입력하세요.

## 10초 구간을 판정하는 방법

XIAO nRF52840 Sense와 Expansion Base에 연결한 SparkFun MAX30101에서 적색광 PPG를 100 Hz로 수집합니다. 보드는 PPG와 내장 가속도계의 3축 측정값을 USB로 전송하고, PC 앱은 겹치지 않는 10초 구간을 분석 서버로 보냅니다.

| 순서 | 검사 | 구간별 출력 |
|---|---|---|
| 1 | IMU로 움직임 확인 | 움직임이 크면 **움직임 제외**. 움직임 기준은 시제품용 잠정 설정입니다. |
| 2 | BUT PPG로 학습한 Random Forest(RF)로 PPG 품질 확인 | 품질 점수가 0.27395(반올림) 미만이면 **재측정** |
| 3 | MIMIC과 6종 리듬 PPG로 학습한 AF 모델 실행 | AF 점수가 0.286 이상이면 **AF 의심**, 미만이면 **비AF** |

움직임 제외·재측정 구간은 AF 점수 없이 기록합니다. 비AF에는 다른 부정맥도 포함될 수 있습니다. PC의 기본 저장 위치는 사용자 홈 폴더의 `Capdyssey_recordings`이며, 세션별 `raw.csv`에는 센서 원본, `events.jsonl`에는 판정·점수·사유가 남습니다.

## 검증 결과

**공개 데이터에서 품질 필터를 적용한 효과.** 같은 시험 참가자를 개발 과정에서 다시 사용한 평가입니다. 전체 AF 검출률과 전체 비AF 오탐률은 재측정 구간을 분모에 포함하며, 판정 구간 특이도는 실제로 판정한 비AF 구간에서 계산합니다.

| 자료 | 품질 필터 | 전체 AF 검출률 | 오탐 수 | 전체 비AF 오탐률 | 판정 구간 특이도 | 판정 비율 |
|---|---|---:|---:|---:|---:|---:|
| MIMIC | 없음 | 98.52% | 91 | 30.03% | 69.97% | 100.00% |
| MIMIC | 최종 RF | 96.62% | 51 | 16.83% | 79.84% | 92.27% |
| 6종 리듬 PPG | 없음 | 90.74% | 1,164 | 19.20% | 80.80% | 100.00% |
| 6종 리듬 PPG | 최종 RF | 90.63% | 1,163 | 19.19% | 80.74% | 99.69% |

MIMIC에서는 오탐이 40구간 줄고 전체 AF 검출률이 1.90%p 낮아졌습니다. 6종에서는 오탐이 1구간 줄었습니다. MIMIC의 판정 구간 특이도는 목표 80%에 미달했습니다. [평가 방법과 한계](docs/QUALITY_AF_V2.md)

**실제 시제품 연결 확인(2026-11-19).** 한 실측 세션에서 PPG·가속도 12,241행을 저장했고, Colab에서 반환한 12구간은 움직임 제외 5개, 품질 미달 재측정 2개, AF 의심 4개, 비AF 1개였습니다. AF 의심 4개는 연속된 약 40초 동안의 모델 출력입니다. 동시 측정한 심전도가 없어 이 기록으로 실측 정확도나 오탐률을 산출할 수 없습니다. [실측 및 테스트 기록](models/Capdyssey_realtime_v1/검증결과.md)

## 개발 과정과 재현 자료

- **초기 모델과 데이터 선택** — MIMIC 단독 모델에서 과적합 징후와 오탐을 확인한 뒤, 6종 리듬 PPG와 입력 길이를 10초로 맞추고 참가자 단위로 나누어 학습했습니다. [초기 모델 결과](docs/AF_MODEL_V1.md) · [두 자료로 학습한 근거](docs/AF_MAIN_TWO_SOURCES.md) · [오탐 분석](docs/AF_FALSE_POSITIVE_AUDIT.md)
- **장기 손목 PPG·ECG 39명 자료** — 잠정 라벨로 탐색 실험을 했고, 주석 코드의 의미와 시간 정렬에 대한 불확실성이 남아 최종 모델 학습·평가에서 제외했습니다. [주석 조사](docs/WRIST_SOURCE_AUDIT.md) · [잠정 실험](docs/AF_PROVISIONAL_EXPERIMENT.md)
- **품질 모델과 재현** — 품질 모델을 고정된 AF 모델 앞에 연결해 비교하고, 최종 RF·AF 모델을 다시 불러와 저장된 16개 구간의 점수가 재현되는지 확인했습니다. [품질 모델 비교](docs/QUALITY_AF_INTEGRATION.md) · [최종 모델 패키지](models/quality_af_v2_final/README.md)
- **데이터와 코드** — [데이터 목록](docs/DATASETS.md) · [이용 조건과 인용](docs/DATA_LICENSES.md) · [데이터 준비](data/README.md) · [실험 노트북](notebooks/) · [테스트](tests/) · [실시간 앱 테스트](models/Capdyssey_realtime_v1/tests/)
