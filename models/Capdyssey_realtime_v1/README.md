# Capdyssey 실시간 측정 패키지 v1

기존 RF 품질 모델과 AF 모델을 그대로 사용하며, **USB 수신 → 실시간 파형 → 메모리에 모은 10초 데이터 전송 → IMU 검사 → 품질 검사 → AF 판정**을 연결하는 연구용 코드입니다. 파일 저장은 기록용이며, 저장 파일을 읽어야 분석되는 구조가 아닙니다.

## 포함 파일

| 파일/폴더 | 실행 위치와 역할 |
|---|---|
| `desktop_app.py` | Windows: 파형 표시, USB 수신, CSV 저장, 분석 결과 표시 |
| `setup_windows.bat` / `run_desktop.bat` | Windows: 최초 설치 / 프로그램 실행 |
| `Colab_realtime_server.ipynb` | Colab GPU: 원본 모델 로딩과 인증된 분석 연결 |
| `firmware/capdyssey_usb/capdyssey_usb.ino` | XIAO: MAX30101 RED + IMU를 USB로 전송하는 참조 코드 |
| `realtime/` | 데이터 검사, 10초 구간 생성, 서버·클라이언트 연결 |
| `model_bundle/` | 원래 모델 가중치와 코드, IMU 제외 기능 |
| `tests/` | 구간 누락, 움직임 제외, 통신 및 화면 테스트 |
| `검증결과.md` | 실행한 검사와 아직 확인하지 못한 항목 |

## 1. 먼저 예시 화면 실행

1. ZIP을 새 폴더에 압축 해제합니다. 압축파일 안에서 바로 실행하지 않습니다.
2. Windows용 Python 3.11 이상을 설치합니다. Tkinter/IDLE과 Python launcher를 포함합니다.
3. `setup_windows.bat`를 실행하고 설치 완료를 기다립니다.
4. `run_desktop.bat`를 실행합니다.
5. **예시 모드**를 체크한 상태에서 **측정 시작**을 누릅니다.

예시 모드는 생성한 파형만 보여주며, 실제 AF·비AF 결과를 만들거나 Colab으로 보내지 않습니다. 결과를 확인하려면 실제 센서와 분석 서버가 필요합니다.

## 2. 보드 코드 준비 및 실제 센서 연결

Arduino IDE에서 Seeed 공식 XIAO nRF52840 Sense 설치 안내에 맞는 보드를 선택하고 다음 라이브러리를 설치합니다.

- SparkFun MAX3010x Pulse and Proximity Sensor Library (`MAX30105.h`)
- Seeed Arduino LSM6DS3 (`LSM6DS3.h`, `LSM6DS3` 클래스)

`firmware/capdyssey_usb/capdyssey_usb.ino`를 열어 컴파일 후 보드에 업로드합니다. 이 코드는 **실제 보드에서 아직 컴파일·업로드 검증되지 않은 참조 구현**입니다. 같은 이름의 다른 IMU 라이브러리는 API가 다를 수 있습니다. 센서는 Expansion Base의 I2C Grove 포트에 Grove–Qwiic 케이블로 연결한 구성을 전제로 합니다.

USB 데이터 케이블로 PC에 연결한 후 Arduino Serial Monitor를 닫습니다. PC 화면에서 예시 모드를 해제하고 포트를 선택합니다. 먼저 분석 주소를 비운 채로 측정을 시작해 실제 파형과 저장 파일을 확인하세요.

현재 구현은 **USB 방식**입니다. BLE는 포함되지 않았습니다. RED 원본 채널 100 Hz, 평균화 1을 고정으로 사용합니다. 채널이나 샘플링 속도를 임의로 바꾸면 입력 규약도 함께 변경해야 합니다. 손목에서 RED 신호가 잘 측정되는지와 원래 모델의 센서 전이 성능은 별도 확인이 필요합니다.

## 3. Colab 분석 연결

1. `Colab_realtime_server.ipynb`를 Colab에 업로드합니다.
2. 런타임을 GPU로 설정합니다.
3. 첫 셀부터 순서대로 실행하고, 요청할 때 **이 전체 패키지 ZIP**을 업로드합니다.
4. 라이브러리 설치 후 재시작 안내가 나오면 런타임을 재시작한 다음 서버 셀을 실행합니다.
5. 서버 셀이 출력하는 HTTPS 주소와 임시 비밀번호를 PC 화면에 입력합니다.
6. 실제 센서를 연결하고 측정을 시작합니다. 첫 결과는 10초 수집과 네트워크·분석 시간 이후 나옵니다.

전송에는 Gradio의 인증된 공유 연결을 사용합니다. 측정 데이터가 PC에서 Colab으로 인터넷을 통해 전달됩니다. 공유 주소와 비밀번호는 저장소/발표 자료에 공개하지 마세요. 측정 세션 파일에는 비밀번호를 저장하지 않습니다.

**Colab 운영 제한:** 무료 관리형 런타임은 노트북 UI를 우회해 외부 화면을 주로 사용하는 작업을 제한할 수 있습니다. 이 코드를 무료 Colab의 상시 서버처럼 사용하지 마세요. 유료 플랜의 양수 컴퓨팅 잔액이 있는 대화형 실험 환경이나 별도 GPU 실행 환경을 사용하고, 당시 Colab 정책을 확인하세요. 유료 Colab도 지속 연결을 보장하지 않습니다. 연결 종료 시 주소를 다시 받아 입력하고 측정을 재시작합니다. 제한을 우회하는 자동 재연결·세션 유지 코드는 없습니다.

별도 CUDA GPU 컴퓨터에서는 같은 의존성과 CUDA용 PyTorch를 준비한 뒤 `python serve_analysis.py`로 로컬 분석 서버를 열 수 있습니다. 같은 컴퓨터의 UI 주소는 `http://127.0.0.1:7860`입니다. 기존 AF 코드는 GPU 필수이며 이 패키지는 CPU 지원으로 바꾸지 않았습니다.

## 결과 해석과 지연

- `excluded / motion_detected`: 큰 움직임. AF 판정 제외, AF 점수·라벨 없음.
- `remeasure`: 품질 부족 또는 IMU 누락·오류. 비AF가 아님.
- `classified / AF` 또는 `non_AF`: 움직임과 품질 검사를 통과한 구간의 모델 출력.
- 센서 누락·시계 이상: 부분 구간 폐기 후 새로운 10초 수집.
- 분석 연결 실패: 파형 수신과 원본 저장은 계속되며 현재 판정은 표시하지 않음.

첫 버전은 겹치지 않는 10초 구간마다 요청합니다. 파형은 50 ms마다 다시 그립니다. 이는 샘플마다 AF 판정한다는 의미가 아닙니다. 요청은 1개 실행 + 최신 대기 1개로 제한하며 밀린 구간을 무한히 쌓지 않습니다. 오래된 응답은 기록하되 현재 결과로 표시하지 않습니다. 분석 제외/지연으로 판정되지 않은 구간도 비AF로 집계하지 마세요.

IMU 기준 RMS 0.10 g / peak 0.50 g는 **검증 전 임시 값**입니다. 실제 정지·움직임 데이터로 조정해야 합니다. 설정은 `model_bundle/scripts/imu_motion_gate.py`의 `MotionConfig`에 정의되어 있고 서버 모델 생성 시 별도 `MotionConfig`로 지정할 수 있습니다. 코드 파일을 직접 바꾸면 manifest 체크섬도 의도적으로 갱신해야 하므로 설정 객체 전달을 권장합니다.

## 저장 파일

기본 위치는 사용자 홈의 `Capdyssey_recordings`이며 UI에서 변경할 수 있습니다. 각 측정마다 별도 폴더를 만듭니다.

- `raw.csv`: 순번, 보드 시간(초), 원본 RED PPG, 가속도 x/y/z(g).
- `events.jsonl`: 연결·구간 폐기·누락·분석 결과·지연·화면 표시 여부. 예시 데이터는 `demo=true`로 표시.

UI 파형은 자동 세로 축척이며 원본 저장값과 모델 입력에는 그 축척을 적용하지 않습니다.

## 입력 타이밍과 현재 한계

보드가 샘플을 읽은 시점을 `micros()` 기반 시간으로 기록합니다. PC 도착 시간을 센서 시간으로 쓰지 않습니다. 다만 MAX30101 변환 시작 순간을 하드웨어로 찍은 정밀 동기화는 아니며, IMU 축 읽기도 약간의 시간차가 있습니다. FIFO가 밀리면 시간을 만들어 채우지 않고 `#GAP`을 보냅니다. 측정 속도·USB 부하·IMU 통신 실패는 실제 장치에서 확인해야 합니다.

원래 품질 RF/AF 가중치와 임계값은 변경하지 않았고 원본 체크섬을 보존합니다. 원래 성능표는 새 센서·IMU·실시간 시스템의 성능 보증이 아닙니다. `model_bundle/README_PPG_ORIGINAL.md`와 manifest의 기존 한계도 적용됩니다.

## 공식 참고 자료

- [Seeed XIAO 안내](https://wiki.seeedstudio.com/XIAO_BLE/)
- [SparkFun 센서 라이브러리](https://github.com/sparkfun/SparkFun_MAX3010x_Sensor_Library)
- [Seeed IMU 라이브러리](https://github.com/Seeed-Studio/Seeed_Arduino_LSM6DS3)
- [Gradio 공유 및 인증](https://www.gradio.app/guides/sharing-your-app)
- [Gradio Python client](https://www.gradio.app/guides/getting-started-with-the-python-client)
- [Colab 사용 제한](https://research.google.com/colaboratory/faq.html)
