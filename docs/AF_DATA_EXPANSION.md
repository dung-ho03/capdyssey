# 과적합 징후에 따른 AF·비AF 데이터 확장

확인일: 2026-10-10. 목표는 단순한 구간 수 증가가 아니라 새로운 참가자와 측정 환경의 확보입니다.

## 추가 데이터가 필요한 이유

AI 학습에서 과적합 징후와 낮은 일반화 성능을 확인하여 더 많은 참가자의 AF·비AF 데이터가 필요하다고 판단했습니다.

실제 Colab 기록에서 3→8 epoch 사이 학습 손실은 0.6462→0.4586으로 감소했지만 검증 손실은 0.6737→0.8395로 증가했습니다. 이는 과적합과 일치하는 경향입니다. 학습 손실에는 클래스 가중치가 있고 검증 손실에는 없으므로 두 손실의 절대 크기를 직접 비교하지 않고 각각의 시간 변화로 판단합니다. 데이터 부족이 유일한 원인이라는 인과관계가 입증된 것은 아닙니다.

검증 손실이 가장 낮았던 3 epoch 모델도 시험 특이도 46.0%, 균형 정확도 61.9%에 그쳤습니다. 더 긴 학습만으로 해결하지 않고 데이터 확장·라벨 점검·오류 분석을 함께 진행합니다. 근거: [Colab 학습 기록](../models/af_mimic_v1_colab/history.json), [평가](../models/af_mimic_v1_colab/metrics.json).

## 우선 확보 자료: 장기 손목 PPG v3

[공식 Zenodo v3](https://zenodo.org/records/23187681), DOI 10.5281/zenodo.23187681, 2026-10-06 공개. Bacevičius, Pluščiauskaitė 외 저자들의 *Long-term electrocardiogram and wrist-based photoplethysmogram recordings with annotated atrial fibrillation episodes*.

- 공식 설명: 39명, 261일의 장기 기록. 손목 PPG 100Hz, ECG 500Hz, 손목 가속도 50Hz 및 ECG 장치 가속도 25Hz.
- ECG 기반 AF 구간을 전문가가 검토·수정했으며 v3에서 AF 주석을 추가 정제했습니다. AF 및 비AF 구간의 실제 수는 파일을 읽어 별도로 집계해야 합니다.
- 구버전 v1/v2는 현재 제한 접근입니다. v3는 공식 API에서 공개 접근이며 006·039번 파일 다운로드와 검증을 완료했습니다. 구버전의 45명 수치를 v3의 참가자 수로 사용하지 않습니다.
- v3는 참가자 ID를 재배정했으므로 구버전과 ID만으로 연결하거나 별도 참가자로 중복 합산하면 안 됩니다.
- 메타데이터는 `other-nc`로 표시하지만 첨부 `LICENSE.txt` 원문은 **CC BY-NC-SA 4.0**입니다. 비영리 캡스톤 연구 범위에서 사용하고 저자·출처·라이선스·변경 내용을 표시합니다. 파생 자료 공유 시 동일조건을 지켜야 하며 상업적 사용 허용으로 해석하지 않습니다.
- 전체 파일 용량은 63,713,598,681 bytes입니다. 처음에는 총용량이 가장 작은 006·039 참가자의 PPG/ECG 쌍을 선택했습니다. 리듬이나 모델 성능을 기준으로 고른 표본이 아닙니다. 이 표본을 전체 39명의 대표성 있는 평가 집합으로 간주하지 않습니다.

재현 가능한 전체 파일 목록·공식 MD5는 [manifest](../data/longterm_af_v3_manifest.json)에 있습니다. 원본 신호와 참가자 임상정보 파일은 Git에서 제외합니다.

### 실제 확보·검증 결과

2026-10-10 로컬에서 **2명(006·039)의 PPG/ECG 4개 파일과 라이선스·참가자 설명 파일, 총 1,418,539,826 bytes**를 확보했습니다. 6개 파일 모두 공식 MD5와 일치하며 SHA256도 기록했습니다. 전체 39명을 다운로드한 것은 아닙니다. [검증 기록](verification-longterm-af-2026-10-10.json)에 파일 크기·해시·구조 검사 결과를 보관합니다.

| 참가자 | PPG 구간 수 | 샘플 수를 100Hz로 환산한 시간 | 비유한 샘플 수 |
|---|---:|---:|---:|
| 006 | 4 | 9.10시간 | 0 |
| 039 | 12 | 116.93시간 | 0 |

합계 약 126.04시간은 원시 PPG 샘플 분량이며 품질 검사·ECG 정답 정렬 후 학습 가능한 시간은 아직 계산하지 않았습니다. 두 참가자 모두 네 종류의 주석 값(0, 0.25, 0.5, 1)을 확인했습니다. 라벨 코드 의미를 확인하기 전까지 **자료 확보 완료 / 학습 입력 준비 미완료**로 구분합니다.

```sh
python -m pip install -r requirements.txt
python scripts/download_longterm_af.py --subjects 006 039 --plan
python scripts/download_longterm_af.py --subjects 006 039
```

[Colab 다운로드·구조 확인 노트북](https://colab.research.google.com/github/dung-ho03/capdyssey/blob/main/notebooks/02_additional_af_data_colab.ipynb)도 제공합니다. 이 신규 노트북의 코드 문법은 검사했으며, 실제 다운로드·파일 검증은 로컬에서 수행합니다. 이전 AF 모델의 Colab 학습 검증과 신규 데이터 노트북의 검증 범위를 혼동하지 않습니다.

```sh
python -m pip install -r requirements-data.txt
python scripts/audit_longterm_af.py --subjects 006 039 --output data/raw/longterm_af_v3/structure-audit.json
```

이 검사는 전체 ECG를 메모리에 적재하지 않고 HDF5 구조, PPG 샘플 수·비유한 값 수, AF 주석 값별 개수와 RR/QRS 관계를 확인합니다. 확인된 구조는 주석 수 = RR 간격 수 = QRS 수 − 1이며, QRS 차이를 500Hz로 환산한 값과 RR 값도 대조합니다. 주석 개수는 30초 학습 구간 개수가 아닙니다.

**라벨 해석은 아직 확정하지 않았습니다.** 006번의 `AF_annotation`에는 0, 0.25, 0.5, 1이 존재합니다. 공식 설명만으로 각 숫자의 의미와 경계 처리 규칙까지 확인할 수 없어 이진 라벨 변환을 적용하지 않았습니다. 특히 0.25·0.5를 임의로 AF 또는 비AF로 바꾸면 안 됩니다. PPG 시작 시각과 ECG 시작 시각도 달라 시각별 정렬이 필요합니다. 261일은 전체 기록 기간이며 끊김 없는 PPG 261일을 뜻하지 않습니다. 비유한 값이 없더라도 움직임 잡음이나 기록 사이 공백이 없다는 의미는 아닙니다.

기본 저장 위치는 저장소의 `data/raw/longterm_af_v3/`입니다. MD5 및 파일 크기를 검사하고 SHA256 기록도 생성합니다. 동일한 검증 완료 파일은 재다운로드하지 않습니다. 큰 파일은 공식 서버의 HTTP Range를 확인해 4개 연결로 받고 `.part`에서 이어받습니다. 최종 체크섬 실패 파일은 사용하지 않습니다. 다른 참가자는 `--subjects 001 002`처럼 지정합니다. 참가자별 ECG와 PPG를 함께 받으며 처음부터 전체 64GB를 다운로드하지 않습니다.

## 추가 후보와 현재 접근 상태

| 후보 | 가치 | 현재 상태 |
|---|---|---|
| [MIMIC-III-Ext-PPG v1.1](https://physionet.org/content/mimic-iii-ext-ppg/1.1.0/) | 6,189명, AF 597,769개 30초 구간, 다양한 비AF 리듬 | 연구자 인증·CITI 교육·프로젝트 DUA 필요. 팀에 승인 계정 없음. 미확보 |
| [DeepBeat](https://github.com/AshleyLab/deepbeat), Synapse syn21985690 | AF 및 품질 분류용 손목 PPG | 저자가 비상업적 DUA 명시. GitHub 코드 GPL과 데이터 이용 조건은 별개. 미확보 |
| [Pulsewatch](https://www.synapse.org/Synapse:syn23565056) | 손목 AF·비AF 및 다른 리듬 후보 | 프로젝트 존재 확인. 다운로드 권한과 이용 조건 해결 전 미확보 |
| [6종 리듬 저자 저장소](https://github.com/zdzdliu/PPGArrhythmiaDetection) | AF 외 PAC/PVC 등을 포함 | 기존 수집분과 중복. 학습·가공·재배포 범위 확인 전 신규 허가 자료로 집계하지 않음 |

MIMIC-III-Ext-PPG의 라벨은 기록된 임상 리듬 전 최대 15분 구간에 연결된 것이므로 전 구간 전문가 ECG 판독과 같지 않습니다. 기존 MIMIC PERform과 원 환자가 중복될 수 있어 환자 ID를 대조해야 합니다. WESAD·DaLiA를 AF 주석 없이 비AF 정답으로 추가하지 않습니다. MIT-BIH AFDB는 ECG 자료이므로 PPG 학습량으로 세지 않습니다.

처음 공유받았던 [Zenodo 21095509](https://zenodo.org/records/21095509)는 공식 API 확인 결과 ECG 결정트리 논문 PDF 1개이며 PPG AF·비AF 원본 데이터셋이 아닙니다. 학습 데이터 후보에서 제외합니다.

## 학습에 넣기 전 필요한 처리

1. 파일 구조, ECG AF 주석의 값과 시간 단위, PPG 유효 구간 및 시간 기준을 확인합니다. 누락 주석을 비AF로 만들지 않습니다.
2. 서로 대응되는 시간의 PPG와 ECG 주석만 매칭하고 AF 전환 경계·주석 불확실 구간의 처리 규칙을 사전에 정합니다.
3. v3 참가자별 분할을 고정한 뒤 30초 구간을 생성합니다. 한 사람의 구간이 학습과 시험에 동시에 들어가면 안 됩니다.
4. AF의 불규칙성을 잡음으로 제거하지 않도록 품질 기준의 클래스별 제외율을 기록합니다.
5. MIMIC과 손목 자료를 무조건 합치지 않고 데이터셋별 AF/비AF 구성과 성능을 보고합니다. 일부 신규 참가자는 최종 시험용으로 남깁니다.

이번 업데이트는 데이터 확보·검증 단계입니다. 새 자료를 사용한 재학습이나 성능 향상을 완료했다는 의미가 아닙니다.
