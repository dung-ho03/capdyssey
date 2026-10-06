# capdyssey
Capstone Design in hallym university

PPG에서 심방세동 의심 구간을 탐지하고, 움직임 정보를 활용한 신호 품질 처리를 연구합니다.

## 팀 연구 자료

- [데이터 6종의 용도·정답·한계](docs/DATASETS.md)
- [다운로드 및 실행 방법](data/README.md)
- [이용 조건과 인용](docs/DATA_LICENSES.md)
- [참고 논문과 읽는 순서](docs/PAPERS.md)
- [2026-10-05 로컬 파일 검증 결과](docs/verification-2026-10-05.json)

공식 출처에서 받은 자료를 재현할 수 있도록 다운로드 도구와 파일 검증 정보를 제공합니다. 원본 신호·가공 데이터·논문 PDF는 이 저장소에 포함하지 않습니다. 6종 리듬 PPG는 연구 사용 안내가 있으나 라이선스 범위가 미확정이어서 자동 다운로드 대상에서 제외했습니다.

```sh
python -m pip install -r requirements.txt
python scripts/download_data.py --list
python scripts/download_data.py mimic_af
```

현재 단계는 데이터 확보·검증입니다. 모델 학습 성능은 아직 보고하지 않았습니다. 데이터는 사람별로 분할하고, 비심방세동을 모두 정상이라고 표시하지 않습니다. 서로 다른 데이터의 PPG와 움직임을 임의로 짝지어 실제 동시 측정 자료처럼 사용하지 않습니다.
