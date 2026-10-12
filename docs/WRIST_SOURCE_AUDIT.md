# 손목 39명 자료의 정답 코드 공개 근거 재검토

검토일: 2026-10-12. 대상: [Zenodo v3, 23187681](https://zenodo.org/records/23187681).

## 결론과 이번 검토 이유

ECG를 바탕으로 의료진이 검토한 주석이라는 연구 근거는 확인했다. 그러나 실제 파일의 `AF_annotation` 값 `0, 0.25, 0.5, 1` 각각을 리듬에 연결하는 공식 대응표는 아래에서 확인한 공개 자료에서 찾지 못했다. 이는 데이터가 신뢰할 수 없거나 사용 불가능하다는 결론이 아니다. 공개 자료에서 문의를 필수 사용 조건으로 명시한 문구도 확인하지 못했다.

파일의 숫자 의미를 추정해 잘못된 정답으로 학습하는 것을 막고, 문의 전에 이미 제공된 설명을 충분히 찾기 위해 검토 범위를 전체 원본·부속 문서·관련 연구로 확대했다. 이번 변경은 근거 기록이며 모델 학습이나 성능 개선 결과가 아니다.

## 직접 확인한 범위

| 자료 | 확인 방법 | 결과와 한계 |
|---|---|---|
| v3 공식 설명과 파일 목록 | Zenodo API를 Colab에서 조회 | MAT 78개, LICENSE.txt, subject_info.xlsx. 별도 코드북·예제 코드·논문 부록 파일은 공개 목록에서 확인되지 않음 |
| subject_info.xlsx | 전체 ZIP/XML, 시트 상태, 문자열, 주석/임베디드 파일 유무 검사 | Sheet1 하나. 숨김 시트·메모·내장 문서 없음. 인구학/임상 정보이며 리듬 코드 대응표 없음 |
| 39명 ECG·PPG 원본 78개 | Colab h5py로 모든 객체 이름, 문자 데이터, 추가 속성, MAT 헤더 검사 | 78/78 완료, 오류 0. 문자 설명은 신호 이름·단위·날짜/시간 등. 비표준 추가 속성 없음. 리듬 코드 설명을 찾지 못함 |
| v2 11242869, v1 5815074 | 공개 메타데이터와 설명 조회 | 각각 45명/8명 설명. 이전 버전 파일은 공개 API에서 제공되지 않아 내부 첨부물까지 검사했다고 주장하지 않음 |
| 연구자의 2025 학위논문 | 114쪽 전체 텍스트 검색, 관련 표/그림 직접 확인, 내장 첨부물 검사 | 3.2.3절(인쇄 48쪽/PDF 49쪽)에 의료진 주석 검토 과정 명시. 코드 대응표는 찾지 못함. PDF 내장 첨부물 없음 |
| TriggersAF 2025 학회 초록 | Methods와 공개 그림/설명 확인 | AF, 심방빈맥 또는 조동, 잦은 PAC, 부정맥 없음의 네 범주 명시. 숫자 값은 명시하지 않음 |
| QUMPHY D4 보고서·공개 소프트웨어 | TriggersAF 절과 저장소 코드 검색 | 외부 평가 자료로 활용 제안. TriggersAF 숫자 대응표/로더는 확인되지 않음. 인접 절의 0/1/2/3/5 코드는 다른 Simband 자료에 관한 것 |
| PulseLM 논문·공개 저장소 | 부록 A.1/A.2, README, 공개 파일 구조 확인 | AFPPGECG 재사용 사례이나 부록은 8명 버전을 설명함. 현재 v3 네 숫자의 대응을 증명하지 않음 |

MAT 검사는 파형 전체를 다시 전처리하거나 의료적으로 재판독한 작업이 아니다. 정답 코드 설명이 들어 있는지 확인한 메타데이터 검사다. 기본 MATLAB 자료형 속성은 별도 코드북으로 간주하지 않았다.

## 핵심 출처

- [현재 v3](https://zenodo.org/records/23187681), [v2](https://zenodo.org/records/11242869), [v1](https://zenodo.org/records/5815074). v3는 주석을 재검토하고 ID를 재부여했다고 명시한다. v1은 장치 간 비동기화, v2/v3는 동기화를 설명하므로 버전 간 해석을 그대로 이식하지 않는다.
- Vilma Pluščiauskaitė, 2025, [A Method for Identifying Personal Triggers of Paroxysmal Atrial Fibrillation](https://biomedicine.ktu.edu/wp-content/uploads/sites/237/2025/07/2025_Disertacija_Plusciauskaite-V.pdf), 3.2.3절. 자동 ECG 검출 후 수동으로 누락·오검출·시간 경계를 검토하며 PAC·심방빈맥·조동도 주석했다고 설명한다. 그림의 AF/non-AF 표시는 공개 MAT 숫자표가 아니다.
- [TriggersAF, Europace 2025 초록](https://academic.oup.com/europace/article/27/Supplement_1/euaf085.262/8142105), Methods. 연구 전체 모집단과 공개 39명 부분집합을 동일시하지 않는다.
- [QUMPHY D4](https://arxiv.org/html/2604.01398v1), 3.2–3.4절; [공개 소프트웨어](https://gitlab.com/qumphy/qumphy-software), 검토 커밋 `cc546cd7ea630c2f237d4c70149c97fc9eab97f2`.
- [PulseLM](https://arxiv.org/html/2603.03331v1), 부록 A.1/A.2; [공개 저장소](https://github.com/manhph2211/PulseLM).

추가로 Daivaras Sokas의 2023 학위논문(112쪽, elaba:181371244)도 검색했으나 데이터셋을 출판 목록에 인용한 정도여서 코드 해석 근거로 채택하지 않았다. 검색한 웹·코드에 없다는 결과가 모든 비공개 자료에도 없다는 증명은 아니다.

## Colab 증거와 저장 위치

[실행 노트북](https://colab.research.google.com/drive/1g_-lbbu6mR39TsM4p3J4zAHphUBc8bOB#scrollTo=l5yNwDK_me0J)에서 확인:

```text
SOURCE_DOCUMENT_AUDIT_COMPLETE
MAT_METADATA_AUDIT_COMPLETE 78 files; errors 0
DRIVE_REPORT_SAVED source_document_audit.json 32691 bytes
DRIVE_REPORT_SAVED mat_metadata_audit_78.json 245446 bytes
```

보고서 폴더: Google Drive `MyDrive/Capdyssey/af_integrated/processed/integrated_10s_resume_v1/`.
저장된 JSON을 다시 읽어 78개 및 오류 0개를 확인했다. 원본 파형과 참가자별 임상 정보를 Git에 추가하지 않았다. 로컬에서는 공개 논문/코드 읽기만 했으며 실제 데이터 검사는 Colab에서 수행했다.

실행 당시 노트북 자체에는 원격 변경으로 인한 자동 저장 충돌이 표시되었다. 원격 편집을 덮어쓰지 않았으며, 노트북 저장 성공을 주장하지 않는다. 위 두 JSON 보고서는 노트북 저장과 별개로 Drive에 기록되고 재열기 검증됐다.

## 다음 학습에 적용할 결정

- 확인된 숫자-리듬 대응과 시간 귀속 규칙을 얻기 전에는 손목 `y=-1`을 유지한다. 빈도나 RR 불규칙성만으로 정답을 추정하지 않는다.
- 원본 다운로드와 10초 파형 준비 결과는 재사용한다. 대응표가 없다는 이유로 다운로드부터 반복할 필요는 없다.
- 연구진의 코드 설명 또는 공식 예제는 남은 해석 문제를 해결할 수 있다. 문의는 아직 발송하지 않았다.
- 손목 정답 연결이 끝나기 전에는 세 자료 통합 학습 완료 또는 성능 결과를 보고하지 않는다.
