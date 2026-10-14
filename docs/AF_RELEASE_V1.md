# AF/비AF 연구 모델 v1

MIMIC+6종 PPG의 고정한 연구용 산출물이다. 품질 AI는 포함하지 않는다.
모델과 전처리, 임계값 0.286을 함께 고정한다. 성능 목표 전체를 달성했다는 뜻의
최종본은 아니다. 기존 시험군에서 MIMIC 민감도98.52%/특이도69.97%,
6종 민감도90.74%/특이도80.80%였다. 기존 시험군을 재사용한 개발 평가다.
MIMIC 비AF 오탐, 손목·새 센서 일반화, 품질 판정은 미해결이다.

## 입력과 출력

- 단일 채널 PPG의 연속10초 원시 수치, 실제 샘플링 주파수 전달.
- 100Hz이면1000점, 125Hz이면1250점. 내부에서50Hz/500점으로 전처리한다.
- 재샘플링 전에 평균 제거 후 표준화하고 0.5–5Hz 필터·맥박/주파수 특징 사용.
- 결측, 평탄, 길이 오류는 예외 처리한다. 이상 신호 전체를 판별하는 품질 AI는 아니다.
- `prediction`: AF 또는 non-AF. 비AF는 정상/건강을 뜻하지 않는다.
- `af_score`: 모델 점수이며 질병 확률로 해석하지 않는다.
- GPU 필요. 지원 가능한 입력 주파수 범위는 전처리의 수치 조건일 뿐 새 기기의 성능 보장이 아니다.

## Colab에서 사용

Drive `Capdyssey/af_integrated/releases/af-two-source-v1.zip`을 압축 해제하거나
같은 위치의 폴더를 사용한다. 원본 데이터 없이 모델 패키지만으로 추론한다.

```python
import sys
from pathlib import Path
release=Path('/content/drive/MyDrive/Capdyssey/af_integrated/releases/af-two-source-v1')
sys.path.insert(0,str(release/'runtime'))
from af_model_v1 import AFModel
model=AFModel(release)
result=model.predict(raw_ppg,sampling_hz=100)
print(result)
```

명령행: `python runtime/af_model_v1.py --release <폴더> --input <단일채널.npy> --sampling-hz 100`

패키지: `model.pt`, `release.json`(해시/설정), `runtime/`, 실행 환경 버전,
고정 선택 기록, 성능 보고서, 재현 검사 결과. 파일 검사는 우발적인 변경 탐지용이며
서명된 배포 체계가 아니다. 새 센서 연결과 품질 AI 통합은 이번 릴리스 범위 밖이다.
학습 가중치와 연구 데이터는 공개 Git에 올리지 않고 Drive에 보관한다.
