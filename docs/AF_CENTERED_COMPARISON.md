# 평균 제거 순서 수정: 동일 조건 재학습

오분류 점검에서 원본의 큰 DC 기준값이 리샘플링 경계 스파이크를 만드는 것을
재현했다. 원본 평균을 리샘플링 전에 제거하고 마지막 z-score는 유지한다.
기존 전처리/모델을 덮어쓰지 않고 `provisional_centered_v2`로 분리한다.

## 고정하는 조건

- 기존 164,791개 구간과 잠정 정답을 그대로 유지한다. 사람/구간을 추가로 제외하지 않는다.
- 원본에서 기존 전처리 입력을 재현하여 저장된 입력과 최대 오차 1e-5 이하인지 확인한다.
- 이전 사람별 train/val/test 분할과 정확히 동일한지 학습 전에 검사한다.
- 같은 CNN, seed 42, 소스/클래스/사람별 표본 가중치, 학습률 .001, 최대 20회,
  검증 BCE로 모델 선택, patience 5, 판정 임계값 .5를 유지한다.
- 판정 조건을 시험 결과에 맞춰 변경하지 않는다. 이미 점검한 시험군이므로 결과는
  개발 비교이며 새로운 외부 검증이라고 부르지 않는다.

## Colab 실행

[실행 노트북](https://colab.research.google.com/github/dung-ho03/capdyssey/blob/main/notebooks/06_af_centered_comparison_colab.ipynb)

원본은 Drive에서 읽고 계산은 Colab에서 한다. 큰 DC 오프셋 불변성과 무효 파형
거절을 확인하는 합성 테스트 2개를 Colab에서 통과했다. 재전처리는 파일별로
완료 내역과 해시를 저장해 중단 후 검증/재사용할 수 있다.
학습 중단 시 optimizer 이어학습은 구현하지 않았으므로 같은 출력 폴더를
덮어쓰지 않고 별도 실행 경로를 사용해야 한다.

Drive `Capdyssey/af_integrated/` 아래:

- `processed/provisional_centered_v2/`: 새 입력과 재현 검사
- `models/provisional_centered_v2/`: 새 모델·분할·평가·comparison.json
- `provisional_centered_v2.log`: 현재 실행 로그

수정 후 실측 성능은 실행 완료 후 기록한다. 코드 수정만으로 향상을 주장하지 않는다.
