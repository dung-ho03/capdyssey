# 참고 논문

논문 PDF를 재배포하지 않고 원문 링크와 팀 연구에서 읽을 목적을 정리합니다.

| 읽는 순서 | 논문 | 우리 연구와의 연결 |
|---|---|---|
| 1 | [Liu et al., 2022 — Multiclass Arrhythmia Detection and Classification](https://doi.org/10.1161/JAHA.121.023555) | 6종 리듬·ECG 기반 정답·다른 부정맥에서의 오탐. 공개 부분은 검증·시험 자료 |
| 2 | [Bashar et al., 2019 — Atrial Fibrillation Detection from Wrist PPG Signals Using Smartwatches](https://doi.org/10.1038/s41598-019-49092-2) | 손목 신호와 움직임 잡음 처리. Simband 원자료는 별도 접근 절차 |
| 3 | [Torres-Soto & Ashley, 2020 — DeepBeat](https://doi.org/10.1038/s41746-020-00320-4) | 신호 품질과 AF 탐지를 함께 다루는 모델 설계 |
| 4 | [Reiss et al., 2019 — Deep PPG](https://doi.org/10.3390/s19143079) | DaLiA와 운동 중 심박수 추정. 심박수 정답과 AF 정답 구분 |
| 5 | [Han et al., 2020 — PAC/PVC 탐지](https://doi.org/10.3390/s20195683) | 다른 부정맥과 AF를 구분할 필요성 |
| 6 | [Jarchi & Casson, 2017 — Wrist 운동 데이터](https://doi.org/10.3390/data2010001) | 센서 구성·운동 프로토콜·동기화 |
| 7 | [RhythmiNet — arXiv 사전공개본](https://arxiv.org/abs/2511.00949) | PPG·가속도 융합 비교 설계. 논문 접근과 데이터 접근은 별개 |

품질 판별 추가 참고: [BUT PPG 공식 설명과 원 논문 링크](https://physionet.org/content/butppg/2.0.0/). 전문가 품질 정답은 심박수 추정 가능성 기준이며 AF 분석 적합성과 동일하지 않습니다.

보고서에는 사람별 분할, 오탐·미탐, 판단 보류 비율, 센서 위치 차이와 데이터 접근 제약을 명시합니다. 실제 모델을 학습·시험하기 전에는 논문의 성능을 팀 모델 성능으로 인용하지 않습니다.
