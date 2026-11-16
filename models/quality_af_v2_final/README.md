# 품질 RF + AF 통합 v2 (최종본)

학습된 모델이 포함된 연구용 실행본입니다. 개인 Google Drive나 원본 학습 데이터 없이 모델을 불러올 수 있습니다.

- `quality_rf.joblib`: 학습된 품질 RF (약 14.3MB)
- `af_model.pt`: 학습된 AF 모델
- `manifest.json`: 적용 임계값, 환경, 파일 체크섬
- `scripts/`: 모델과 함께 보관한 실행 코드
- `reload_check.json`: Colab 16개 구간 저장 점수 재현 확인
- `development_test.csv`: 반복 사용한 시험군 개발 평가

품질 기준은 manifest의 0.2739494799615767입니다. RF 학습 파일 내부의 다른 임계값으로 대체하지 마세요.
GPU 환경에서 이 폴더의 scripts를 경로에 추가하고 `QualityAF(이_폴더)`를 생성한 뒤 `predict(raw_ppg, 실제_샘플링주파수)`를 호출합니다. 입력은 단일 원본 채널의 정확한 10초입니다.

품질 미달은 비AF가 아니라 재측정입니다. 최종본은 현재 팀 전달용 버전을 뜻하며 임상·새 센서 검증 완료를 의미하지 않습니다. 상세 성능과 한계는 저장소 docs/QUALITY_AF_V2.md를 확인하세요.
