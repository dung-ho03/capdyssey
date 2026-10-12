# 손목 v3 정답 연결 전 확인 사항

전처리된 10초 파형을 AF/비AF 학습에 사용하려면 ECG 주석 숫자의 의미와 시간 경계 규칙을 확인해야 한다. 이것은 데이터의 임상적 신뢰성 부족이 아니라 공개 파일의 코드 해석에 관한 문제다.

대상: https://zenodo.org/records/23187681 (v3). 버전 간 참가자 ID가 달라질 수 있으므로 이전 버전의 코드나 ID를 확인 없이 적용하지 않는다.

## 현재 확인한 근거

- 공식 설명은 의료진 검토를 거친 박동별 AF 주석과 ECG/PPG 동기화를 명시한다.
- 초기 표본 검사 이후 39명 ECG·PPG 전체 78개 파일의 문자 데이터와 추가 속성을 Colab에서 검사했다. 오류 0개. 숫자별 리듬 설명은 발견되지 않았다. 상세 범위는 [공개 자료 재검토 기록](WRIST_SOURCE_AUDIT.md)을 참고한다.
- 숫자 코드의 빈도나 RR 불규칙성만으로 정답의 의미를 역추정하지 않는다.
- 39명 전체 구조 검사는 `scripts/audit_wrist_annotations.py`로 Colab에서 실행한다. 보고서는 Drive 전처리 번들의 `annotation_audit_39.json`에 저장한다.

## 연구진에게 확인할 질문 (미발송)

공개 자료에서 연구진 문의를 필수 사용 조건으로 명시한 문구는 확인하지 못했다. 아래 질문은 코드 해석을 해결하기 위한 방법이며, 데이터 사용 허가가 반드시 필요하다는 의미가 아니다. 문의는 발송하지 않았다.

We are using version 3 of your dataset, DOI 10.5281/zenodo.23187681, for a noncommercial university capstone on PPG-based AF detection.

Could you please confirm:

1. The exact rhythm meanings of each numeric value in `AF_annotation`, including 0, 0.25, 0.5 and 1, and any other possible values.
2. Whether annotation element i labels the interval from `QRSindex[i]` to `QRSindex[i+1]`, the preceding/following beat, or another interval convention.
3. Whether `QRSindex` uses one-based ECG sample indices at 500 Hz, and whether additional clock correction is required beyond `recording_startday`/`recording_starttime` for pairing 10-second PPG windows with ECG annotations.
4. How to interpret missing PPG start days in subject 003 records 0–2 and the flat GREEN recordings in v3 subjects 025 and 035.

If available, a v3 annotation codebook or official example loader would be very helpful. Thank you.

## 코드북 확인 후 적용할 방식

확인된 ECG 시간 구간에만 정답을 연결한다. 경계 전환·미확정·시간 정보 없는 구간은 별도 표시하고 감독학습에서 제외한다. 같은 참가자가 학습·검증·시험에 섞이지 않게 분할한다. 원본 숫자 코드와 변환 근거를 보존하고, 학습에 실제 포함된 인원 및 AF/비AF 구간 수를 데이터셋별로 보고한다. 손목 정답을 확인하기 전에는 세 자료 통합 학습 완료로 표시하지 않는다.

## Colab 전체 검사 결과 (2026-10-12)
39명 검사 완료. 확인된 코드는 0.0, 0.25, 0.5, 1.0. 주석/RR/QRS 배열 길이 불일치 0명, QRS 시각 순서 오류 0명. diff(QRSindex)/500과 rr의 최대 차이는 약 5.68e-14초로 수치 오차 수준이다. 이 구조 검증이 주석 코드의 리듬 의미나 앞/뒤 박동 귀속 규칙을 증명하지는 않는다. 원본 및 전처리 재실행 없이 검사했으며 AF/비AF 라벨은 변경하지 않았다.
