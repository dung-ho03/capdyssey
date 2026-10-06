# 데이터 다운로드

Python 3.10 이상에서 저장소 최상위 폴더를 기준으로 실행합니다.

```sh
python -m pip install -r requirements.txt
python scripts/download_data.py --list
python scripts/download_data.py mimic_af --plan
python scripts/download_data.py mimic_af
python scripts/download_data.py wrist
python scripts/download_data.py ptt but
python scripts/download_data.py dalia
```

`--plan`은 이름·이용 조건·다운로드 범위를 출력하며 다운로드하지 않습니다. `--workers 4`로 동시 다운로드 수를 조절할 수 있습니다. 기본값은 8입니다.

원본은 `data/raw/<자료명>/`에 저장됩니다. MIMIC은 두 원본 MAT, DaLiA는 원본 ZIP, PhysioNet 세 자료는 WFDB 파형·주석·배포 문서입니다. PTT는 같은 내용을 중복하는 CSV 버전을 제외합니다. 재실행하면 SHA256이 맞는 파일은 건너뛰고 손상된 파일은 다시 받습니다. 오류가 발생하면 성공으로 표시하지 않습니다.

DaLiA는 약 2.7GB 이상의 여유 공간이 필요하며 압축 해제에는 추가 공간이 필요합니다. ZIP은 자동으로 풀지 않습니다. BUT에는 작은 파일 약 2만 7천 개가 있어 다운로드에 시간이 걸립니다. 네트워크 정책 때문에 실패하면 해당 공식 페이지에서 직접 받을 수 있습니다.

받기 전에 [자료별 이용 조건](../docs/DATA_LICENSES.md)을 읽고 보고서에 출처를 인용합니다. 원본이나 가공 데이터를 공개 공유하려면 각각의 배포 조건을 적용해야 합니다.

## 파형 읽기

분석에는 필요에 따라 `numpy scipy wfdb`를 별도로 설치합니다. 다운로드 도구는 학습·전처리를 수행하지 않습니다.

- MIMIC: 원본 MAT의 구조와 라이선스를 확인합니다. 기존 팀 가공본(30초 창 1,342개)은 이번 다운로드에 포함되지 않습니다.
- PTT: `wfdb.rdrecord('data/raw/ptt/s1_walk')`, R파는 `wfdb.rdann('data/raw/ptt/s1_walk', 'atr')`로 읽습니다.
- BUT: `quality-hr-ann.csv`의 ID를 같은 폴더의 PPG 파형과 연결합니다. 1=심박수 추정에 좋은 품질, 0=나쁜 품질이며 AF 정답이 아닙니다.
- Wrist: 공식 설명에 따라 센서 시간축 정렬을 확인합니다.
- DaLiA: ZIP 안의 readme를 읽고 원본 신호와 ECG 기반 심박수 정답의 시간창을 맞춥니다.

처음부터 같은 참가자의 데이터가 학습·시험에 섞이지 않도록 분할합니다. 다운로드 검증은 파일 무결성 검사이며 모델 성능이나 임상 신뢰도 검증이 아닙니다.
