# MIMIC + 6종 PPG 주 모델

손목 자료의 낮은 구별 능력과 미해결 품질·정렬 문제 때문에, 2026-10-14부터
주 모델의 학습/선택/시험은 MIMIC과 6종으로 제한한다. 손목 원본과 기존 실험은
보존한다. 원본 데이터 전체가 잘못되었다는 결론이나 손목 일반화 성공 주장은 하지 않는다.

## 고정 절차

1. 검사 완료한 `provisional_centered_v2` 중 두 자료만 사용한다. 기존 피험자 분할 유지.
2. 10초/50Hz/500점, 평균 제거 후 리샘플링, 표준화, 기존 맥박·주파수 특징 유지.
3. 손목을 제외한 학습군으로 특징 평균/표준편차와 source/class/person 가중치를 계산한다.
4. Colab GPU에서 seed 42/43, 최대20epoch, patience5, AdamW 0.0005로 학습한다.
5. 검증군에서 두 자료 각각 특이도80% 이상인 공통 임계값 중 평균 민감도 최대값 선택.
6. 검증 평균 민감도가 높은 seed 선택(동률이면42). 체크포인트 해시와 임계값을 먼저 저장한다.
7. 고정한 모델로 기존 시험군 평가. 시험 결과를 보고 임계값이나 seed를 바꾸지 않는다.

이는 기존 검증/시험 분할을 재사용한 개발 평가다. 시험군은 과거 다른 실험에서 이미
확인했으므로 완전히 독립된 외부 검증 또는 임상 검증이라고 부르지 않는다.
손목 포함 이전 실험과 모델·분할·최적화 설정은 같지만 학습 자료 수가 줄어 epoch당
업데이트 수와 임계값 제약 대상이 달라진다. 순수하게 동일한 계산량의 인과 비교는 아니다.

## 실행과 저장

`python scripts/af_pulse_experiment.py --base <Drive 경로> --two-sources`

완료 후 `python scripts/finalize_two_sources.py --base <Drive 경로>`.
모델과 결과는 Drive `models/main_two_sources_v1`에 저장한다.
`candidate/model.pt`, `frozen_selection.json`, `test_report.json`이 연구 후보 산출물이다.
입력은 ECG 없이 PPG만 사용한다. 전처리된 입력 추론은 `finalize_two_sources.Predictor`.
원본 PPG는 `prepare_af_centered.centered_window(raw, fs)`로 처리한 후 전달한다.
의료 진단용 배포 모델이 아니며, 낮은 품질 신호에 대한 검증된 판정 보류 기능은 아직 없다.
