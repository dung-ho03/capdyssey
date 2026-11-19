# IMU 움직임 제외 기능 추가본

이 버전은 기존 학습 모델 앞에 가속도 기반 제외 규칙을 추가합니다. 모델을 재학습한 버전이 아닙니다.

## 처리 흐름

1. 같은 10초 구간의 IMU 데이터 확인.
2. 움직임이 크면 `status='excluded', reason='motion_detected'` 반환. 품질 RF와 AF 모델을 모두 건너뜁니다.
3. 움직임이 작으면 기존 RF 품질 검사와 AF 분류를 그대로 실행합니다.
4. IMU가 없거나 유효하지 않으면 `status='remeasure'`로 반환합니다.

제외된 구간의 `label`, `quality_score`, `af_score`는 모두 `None`입니다. 이를 0 또는 비AF로 바꿔 저장하지 마세요. 화면에는 “움직임 감지 — 잠시 멈춰주세요”를 표시하면 됩니다. 다음 10초 구간을 새로 입력하면 다시 검사합니다. 이 패키지는 자동 수집 루프나 장치 펌웨어를 포함하지 않습니다.

## Colab / Python 사용

기존 모델을 실행하던 Colab GPU 환경에서 ZIP을 새 폴더에 풉니다. 기존 `QualityAF` 대신 `MotionQualityAF`를 사용해야 IMU 검사가 적용됩니다. 이전 PPG 전용 API는 과거 평가 재현을 위해 보존했습니다.

```python
import sys
from pathlib import Path
folder = Path('/content/Capdyssey_quality_af_v2_IMU')
sys.path.insert(0, str(folder / 'scripts'))
from quality_af_imu import MotionQualityAF
from imu_motion_gate import MotionConfig

model = MotionQualityAF(folder, MotionConfig(
    rms_threshold_g=0.10, peak_threshold_g=0.50))

# 실제 수집 데이터로 아래 변수들을 준비합니다.
# raw_ppg: 정확한 10초, 한 채널, shape=(10 * ppg_fs,)
# accel_xyz: 중력을 포함한 3축 가속도, shape=(N, 3)
# imu_times_s: 각 가속도 샘플의 장치 타임스탬프(초), shape=(N,)
# ppg_start_s: PPG 첫 샘플의 장치 타임스탬프(초)
# ppg_fs: 실제 정수 샘플링 주파수
result = model.predict(
    raw_ppg, ppg_fs,
    accel=accel_xyz, imu_times_s=imu_times_s,
    ppg_start_s=ppg_start_s, accel_unit='g',
)

if result['status'] == 'excluded':
    print('움직임 감지 — 잠시 멈춰주세요')
elif result['status'] == 'remeasure':
    print('측정 데이터 확인 / 재측정:', result['reason'])
else:
    print(result['label'], result['af_score'])
```

## IMU 입력 조건

- PPG와 IMU는 동일한 장치 시계를 사용해야 합니다. PC/BLE 수신 시간을 센서 측정 시간 대신 쓰지 마세요.
- 가속도 단위는 `g` 또는 `m/s2`를 명시합니다. 중력을 제거한 선형 가속도가 아니라 센서 원본 가속도를 사용합니다.
- IMU는 25 Hz 이상 수집을 권장합니다. 10초 구간의 앞뒤 0.05초 이내까지 데이터가 있어야 하며, 0.10초를 넘는 샘플 공백은 허용하지 않습니다.
- 시간은 증가해야 하고 NaN, 무한대, 전부 0인 가속도는 허용하지 않습니다.
- XIAO에서 IMU를 읽어 PPG와 함께 저장/전송하는 펌웨어는 별도로 필요합니다. 센서 연결만으로 이 Python 코드에 값이 자동 전달되지는 않습니다.

## 움직임 기준과 검증 범위

각 축의 10초 평균을 뺀 뒤 3축 변화량의 크기를 계산합니다. RMS가 0.10 g 이상이거나 순간 최대값이 0.50 g 이상이면 그 구간을 제외합니다. 이 값은 **동작 확인용 임시 기준**이며 팀 센서/손목 착용 데이터로 검증되지 않았습니다. 걷기 등 활동 종류를 분류하는 모델이 아닙니다. 회전으로 중력 방향이 바뀌어도 변화량이 커질 수 있습니다.

실제 장치의 정지, 손목 흔들기, 걷기 데이터를 기록하고 `motion.rms_g`, `motion.peak_g`를 비교해 기준을 조정하세요. 정지 상태에서도 노이즈가 있고, 작은 움직임이 PPG를 망가뜨릴 수 있으므로 기존 PPG 품질 검사를 유지합니다. 멈춘 IMU 데이터가 반복 전송되는 상황, 센서 포화, 서로 잘못 맞춘 시계는 수집 단계에서도 검사해야 합니다.

기존 AF/품질 모델 가중치와 임계값은 변경하지 않았습니다. 기존 성능표에는 IMU 제외 효과가 반영되어 있지 않습니다. 새 평가에서는 전체 구간 수, 움직임 제외 수, IMU 오류 수, 품질 미달 수, 실제 판정 수를 따로 기록하고 판정된 구간의 AF 성능을 평가하세요.

합성 IMU 데이터와 대체 PPG 예측기를 이용한 테스트는 다음과 같습니다. 이 검사는 분기 로직을 검증하며 실제 장치의 움직임 정확도나 기존 GPU 모델의 정확도를 검증하지 않습니다.

```bash
python -m unittest discover -s tests -v
```
