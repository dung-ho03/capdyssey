# 손목 PPG 품질 분류 전용 경로

이 경로의 목적은 **10초 원시 PPG → good / bad / invalid_input**입니다.
AF 정답·AF 모델을 사용하지 않습니다. 기존 모델을 수정하거나 성능 향상을 주장하지 않습니다.
기존 `models/quality_af_v2_final/`의 RF는 그대로 로드하고 manifest 기준
`0.2739494799615767`을 사용합니다. joblib 안의 다른 학습 임계값을 대신 쓰지 않습니다.

## 1. 실행 환경과 모델

저장소를 내려받아 저장소 최상위에서 실행하세요. 파일명에 `(1)`을 붙이지 않습니다.
Python 3.10 이상, 가상환경 사용을 권장합니다.

```sh
python -m pip install -r requirements-quality.txt
```

CPU RF는 torch·CUDA·AF 모델이 필요 없습니다. 아래 호출은 기능 연결 예시입니다.
실제 데이터의 샘플 번호/시간 검사는 다음 절의 CSV 평가기로 수행합니다.

```python
import sys
sys.path.insert(0, 'scripts')
from quality_only import QualityOnly
model = QualityOnly('models/quality_af_v2_final')
# raw_ppg: 필터링/정규화하지 않은 단일 채널의 정확한 10초 값
# result = model.predict(raw_ppg, fs=100)
```

- `good` / `bad`: `predicted_quality=1/0`, `quality_score`는 모델 점수입니다.
- `invalid_input`: 누락/시간 오류 또는 비유한·평탄 입력. 품질 정답 0으로 바꾸지 않습니다.
- API의 길이·주파수 오류는 예외입니다. CSV 경로는 불완전 구간을 `invalid_input`으로 보고합니다.
- 체크섬은 모델 무결성 검사입니다. 신뢰하는 프로젝트 모델만 로드하세요.
- 기존 BUT RF 10개 특징과 v2 RF 19개 특징을 지원합니다. 새 학습 결과도 같은 인터페이스입니다.

## 2. 장치 데이터 수집

기존 `log_ppg.py`의 IR 한 열 파일에는 시간·누락을 확인할 정보가 없습니다.
그 파일에 추정한 샘플 번호/시간을 덧붙여 검증된 데이터처럼 사용하지 않습니다.
새 수집기는 펌웨어가 **줄바꿈으로 끝나는 다음 CSV**를 전송해야 합니다.

```text
sample_index,timestamp_us,ppg_red,ppg_ir,ppg_green
0,0,120000,130000,110000
1,10000,120010,130005,110009
```

헤더/상태 메시지는 별도 로그에 남고, 숫자 데이터만 파싱합니다. 선택적으로 마지막에
`ax,ay,az,gx,gy,gz` 여섯 값을 추가할 수 있습니다. 단위와 IMU 취득 시점/정렬 방법은
수집 프로토콜에 기록해야 합니다. 이 수집기는 IMU를 자동 동기화하거나 품질 모델에 넣지 않습니다.

펌웨어 조건:

- `sample_index`: PPG 샘플 취득마다 증가하는 번호. PC에서 재부여하지 않습니다.
- `timestamp_us`: USB 출력 시간이 아닌 **샘플 취득 시각**. FIFO 일괄 읽기 시 각 샘플의 시각을 복원해야 합니다.
- 카운터/시계 wrap은 펌웨어에서 확장하거나 새 세션으로 나눕니다. 현재 감사기는 reset/wrap을 거절합니다.
- 100Hz는 예시입니다. 센서 평균화 이후 실제 출력률을 확인해서 `--fs`에 지정합니다.
- 아직 보드 펌웨어가 이 형식인지 확인하지 않았습니다. 단일 IR 출력 펌웨어에는 바로 연결되지 않습니다.

```sh
python scripts/collect_quality_csv.py --port COM3 --fs 100 --seconds 60 --subject S01 --device D01 --session S01_D01_001 --activity rest --channel ppg_red --sensor-settings "red current=..., pulse width=..., averaging=..., firmware=..." --out recordings/session01
```

`samples.csv`, `sessions.csv`, `collection.json`, `serial_messages.txt`가 저장됩니다.
설정의 `...`는 실제 값을 적으세요. 측정 파일에 활동/참가자가 섞이면 파일과 세션을 나누세요.
중단되거나 오류 난 수집물도 자동 보정하지 말고 원본을 보존합니다.

## 3. CSV 감사와 라벨 작성

세션 목록 `sessions.csv`:

```csv
filename,subject_id,device_id,session_id,activity,channel,fs
samples.csv,S01,D01,S01_D01_001,rest,ppg_red,100
```

원시 CSV에는 `sample_index,timestamp_s`와 선택한 채널 열이 있어야 합니다.
시각 단위는 초, 장치 취득 시계여야 하며 `host_elapsed_s`를 대신 쓰지 않습니다.
센서 설정·단위·극성·펌웨어 버전은 목록의 추가 열이나 별도 원본 메타데이터로 보존하세요.

먼저 라벨 없이 검사:

```sh
python scripts/evaluate_quality_csv.py --manifest recordings/session01/sessions.csv --data-root recordings/session01 --model models/quality_af_v2_final --out results/audit_01
```

- 원본 행 순서대로 10초 분량씩 나눕니다. `window_index`는 0부터 시작하는 **행 기준 구간 번호**입니다.
- 구간 경계의 누락도 검사합니다. 마지막 불완전 구간도 버리지 않고 기록합니다.
- 샘플 번호가 1씩 증가하는지, 시간이 증가하는지, 간격이 설정 fs에 맞는지 확인합니다.
- 기본 개별 시간 간격 허용치는 ±20%, 구간 전체 누적 오차는 1샘플입니다.
  `--jitter-fraction` 변경 시 결과에 기록됩니다. 오류를 통과시키기 위해 사후 조정하지 마세요.
- 정렬·패딩·보간으로 문제를 숨기지 않습니다. 시계 정보가 없으면 `invalid_input`입니다.
- `predictions.csv`, 빈 `label_template.csv`, `report.json`을 저장합니다.

`label_template.csv`에 독립 검토한 `quality=0/1`을 입력해 `labels.csv`로 저장합니다.
판단 불가 구간은 빈칸으로 두세요. 모델 예측을 가리고 원시 파형으로 판독하며,
가능하면 두 평가자가 독립 검토하고 불일치를 조정합니다.

```csv
filename,window_index,quality
samples.csv,0,1
samples.csv,1,0
samples.csv,2,
```

라벨은 해당 원본 파일/행 기준 구간에 연결됩니다. 라벨 후 CSV 행 삭제/재정렬을 하지 마세요.
모든 신호를 움직임 여부로 자동 라벨링하지 않습니다. BUT의 정답은 심박수 추정 가능성이므로
손목의 육안 형태 라벨과 완전히 동일하다고 가정하지 않습니다. 판정 기준과 차이를 보고서에 명시하세요.

## 4. 독립 품질 평가

```sh
python scripts/evaluate_quality_csv.py --manifest recordings/session01/sessions.csv --data-root recordings/session01 --model models/quality_af_v2_final --labels labels.csv --out results/external_01
```

혼동행렬 순서는 `[bad, good]`입니다. 클래스별 precision/recall/F1, balanced accuracy,
AUROC, good AUPRC, bad_pass_rate(불량 중 양호로 통과한 비율), 사람/활동/장치별 결과를 저장합니다.
라벨이 없으면 정확도/F1을 계산하지 않습니다. 빈 라벨·입력 거절 수와 클래스별 입력 거절률을
별도 보고합니다. 입력 거절을 정답 bad로 간주하지 않으며 유효 구간의 성능만 보고하지 마세요.
단일 클래스 그룹에는 balanced accuracy/AUROC가 없습니다.

파일 오류·매칭 안 된 라벨은 `evaluation_complete=false`, CLI 종료 코드 2입니다.
출력 폴더는 새 이름을 사용하며 기존 평가를 덮어쓰지 않습니다. 입력 파일/라벨/모델 해시를 기록합니다.
프로그램은 실제 라벨이 독립적인지, 같은 사람이 다른 익명 ID로 들어왔는지를 확인할 수 없습니다.
개선용 참가자와 최종 평가용 참가자를 수집 전에 분리하고 최종 모델/채널/임계값을 먼저 동결하세요.
최종 평가 결과를 본 뒤 수정하면 새로운 독립 평가군이 필요합니다.

## 5. 재현 가능한 새 모델 비교

과거 v2 RF의 미공개 실험 코드를 복원했다고 주장하지 않습니다. 다음은 **새 학습 프로토콜**입니다.
이미 확보한 BUT v2.0.0 폴더를 지정하세요. 없으면 저장소의 `scripts/download_data.py but --workers 8`을
실행해 준비할 수 있습니다. 전체 다운로드/학습은 이 수정 작업에서 수행하지 않았습니다.

```sh
python scripts/train_quality_models.py --raw-dir data/raw/but --out results/quality_common_01
```

- 기존 10개 특징 RF와 19개 특징 확장 RF를 동일 참가자 분할로 학습합니다.
- 전처리 imputer는 학습군에만 적합하고 임계값은 검증군 balanced accuracy로 결정합니다.
- 시험군은 점수 보고에만 사용합니다. 시험 결과로 모델/특징/임계값을 선택하지 않습니다.
- 분할·특징·원본 fingerprint·코드 해시·환경·학습 모델·시험 예측값을 저장합니다.
- 결과 폴더 내 `rf_baseline` / `rf_extended`를 `--model`에 넣어 외부 평가할 수 있습니다.
- 진폭 특징은 호환성을 위해 보존했습니다. 무작정 스케일을 바꾸지 않습니다.

CNN까지 같은 분할·임계값 목적함수로 비교하려면 torch를 추가 설치하고:

```sh
python -m pip install "torch>=2.2,<3"
python scripts/train_quality_models.py --raw-dir data/raw/but --out results/quality_common_02 --include-cnn --device cuda
```

GPU가 없는 PC는 `--device cpu`를 사용합니다. RF 경로에는 torch가 필요 없습니다.
CNN은 기존 구조를 사용하되 새 공통 분할, 공통 RF 데이터 리더, 검증 AUPRC early stopping,
검증 balanced accuracy 임계값을 사용합니다. 과거 CNN의 Platt 보정·증강 절차를 그대로
재현하는 실험은 아닙니다. sigmoid는 보정된 확률이 아닙니다. 헤더 복원 48개를 포함한
모든 입력을 세 모델이 공유합니다. 비교 결과는 과거 노트북 수치와 구분하세요.
새 `cnn` 폴더도 동일 CSV 평가기에서 사용할 수 있으며 이 경우에만 torch를 불러옵니다.

성능이 상승했다고 보고하려면 새 실험과 손목 외부 평가 결과가 필요합니다.

## 6. 확인한 범위

`tests/test_quality_standalone.py`는 실제 저장 RF의 CPU 로드, 임계값 선택, 모델 해시,
취득 오류, 라벨 분모, 합성 RF/CNN 학습·저장·재로드를 검사합니다.
이 테스트는 정확도 검증이나 실제 센서 연동 시험이 아닙니다.

```sh
python -m pip install pytest
python -m pytest tests/test_quality_standalone.py tests/test_quality_af_v2.py -q
```
