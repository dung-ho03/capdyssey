# capdyssey
Capstone Design in hallym university

PPG에서 심방세동 의심 구간을 탐지하고, 움직임 정보를 활용한 신호 품질 처리를 연구합니다.

## 세 자료 통합 학습 진행

[세 자료 통합 코드·Colab 실행 범위](docs/AF_INTEGRATION_IMPLEMENTATION.md): 공통 10초 데이터 준비와 통합 학습기를 구현했습니다. 손목 정답 미확정 등 실행 조건이 남아 있으며 통합 성능 결과는 아직 없습니다.

**10초 실행 확인:** MIMIC만 사용한 비교 기준을 실제 학습했습니다. 민감도 29.4%, 특이도 69.0%로 성능이 부족하며, 세 자료 통합 모델은 아직 아닙니다. [실측 결과·미완료 항목](docs/AF_10S_RUN.md).

MIMIC·6종 리듬·장기 손목 PPG의 확인된 정답으로 학습하여 새로운 사람의 PPG만으로 AF/비AF를 예측하는 것이 목표입니다. 초기 모델의 과적합 징후와 높은 비AF 오분류율 때문에 참가자·리듬·측정 환경을 확대하며, 개선 여부는 별도 시험으로 검증합니다.

현재 **1단계 조사 결과를 기록했으며 통합 학습은 아직 실행하지 않았습니다.** MIMIC과 6종의 정답 매핑을 확인했고, 6종의 구체적인 사용 범위와 손목 자료의 숫자 주석 정의는 확인 대기입니다. 6종은 10초 구간이므로 기존 30초 입력 계획을 수정해야 합니다. [단계별 진행 이유·검사 결과·남은 항목](docs/AF_INTEGRATED_TRAINING.md)을 확인하세요.

## 과적합 징후에 따른 데이터 확장

AI 학습에서 과적합 징후와 낮은 일반화 성능을 확인하여 더 많은 참가자의 AF·비AF 데이터가 필요하다고 판단했습니다. 학습 손실이 감소하는 동안 검증 손실이 증가한 실제 Colab 기록을 근거로, 39명의 장기 손목 PPG·ECG·AF 주석을 제공하는 공개 v3 자료를 추가 확보 대상으로 선정했습니다. 데이터 부족이 유일한 원인으로 확정된 것은 아니며 새 자료로 재학습한 결과는 아직 없습니다.

[확장 이유·자료 이용 조건·확보 상태](docs/AF_DATA_EXPANSION.md) · [추가 자료 Colab 다운로드](https://colab.research.google.com/github/dung-ho03/capdyssey/blob/main/notebooks/02_additional_af_data_colab.ipynb)

## 학습 완료된 AF/비AF v1

[실측 결과·사용법·한계](docs/AF_MODEL_V1.md) · [저장된 모델](models/af_mimic_v1/best_model.pt)

30초 PPG를 입력받아 AF/non-AF 점수를 반환하는 연구용 v1을 학습하고 저장했습니다. 시험 7명/241구간에서 민감도 75.3%, 특이도 47.1%, 균형 정확도 61.2%입니다. 오탐이 많아 제품 배포나 진단에 사용할 성능은 아닙니다. 저장 모델은 재학습 없이 사용할 수 있으며, Colab 마지막 부분에 예측 예제가 있습니다.

## Colab에서 첫 모델 학습

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dung-ho03/capdyssey/blob/main/notebooks/01_af_baseline_colab.ipynb)

위 버튼에서 열고 GPU 런타임을 선택한 뒤 셀을 위에서 아래로 실행하세요. MIMIC PERform AF 다운로드, 사람별 분할, 작은 CNN 학습, 오탐·미탐 평가, 모델 저장까지 수행합니다. GPU가 없으면 CPU를 사용합니다. 결과는 기본적으로 본인 Google Drive에 저장하며 Drive 연결 승인이 필요합니다. 무료 Colab GPU와 실행 시간은 보장되지 않습니다.

이것은 AF 기초 모델이며 BUT 품질 모델과 IMU 융합은 포함하지 않습니다. 35명의 소규모 침상 자료이므로 손목 장치의 성능 검증을 대신하지 않습니다. 저장된 v1 결과와 별도로 재학습 결과는 실행 후 생성됩니다. Colab 계정에서의 전체 실행은 팀원이 직접 시작합니다.

## 팀 연구 자료

- [데이터 6종의 용도·정답·한계](docs/DATASETS.md)
- [다운로드 및 실행 방법](data/README.md)
- [이용 조건과 인용](docs/DATA_LICENSES.md)
- [참고 논문과 읽는 순서](docs/PAPERS.md)
- [2026-10-05 로컬 파일 검증 결과](docs/verification-2026-10-05.json)

공식 출처에서 받은 자료를 재현할 수 있도록 다운로드 도구와 파일 검증 정보를 제공합니다. 원본 신호·가공 데이터·논문 PDF는 이 저장소에 포함하지 않습니다. 6종 리듬 PPG는 연구 사용 안내가 있으나 라이선스 범위가 미확정이어서 자동 다운로드 대상에서 제외했습니다.

```sh
python -m pip install -r requirements.txt
python scripts/download_data.py --list
python scripts/download_data.py mimic_af
```

데이터 확보·검증과 학습·추론 코드가 준비돼 있습니다. 데이터는 사람별로 분할하고, 비심방세동을 모두 정상이라고 표시하지 않습니다. 서로 다른 데이터의 PPG와 움직임을 임의로 짝지어 실제 동시 측정 자료처럼 사용하지 않습니다. 실제 학습 결과와 한계는 위 v1 보고서에 공개했습니다.
