# 품질 RF + AF v2 — IMU 움직임 제외 추가본

실행 클래스: `scripts/quality_af_imu.py`의 `MotionQualityAF`.

큰 움직임 → 판정 제외 / 움직임 적음 → 기존 품질 검사 → AF·비AF 분류.
IMU 입력 누락·오류는 재측정이며, 비AF로 처리하지 않습니다.

설치 및 입력 예제는 **IMU_사용법.md**를 확인하세요. 기존 모델 실행에는 Colab GPU가 필요합니다.
보드 IMU 수집 펌웨어는 포함되지 않습니다. 움직임 기준은 실제 장치 검증 전 임시 값입니다.
기존 모델 가중치·임계값은 동일하며 과거 평가는 IMU 성능을 나타내지 않습니다.
원래 설명은 README_PPG_ORIGINAL.md에 보존했습니다.
