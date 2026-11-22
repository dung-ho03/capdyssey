"""XIAO가 시리얼로 보내는 PPG 값을 실시간으로 화면에 찍고 CSV로 저장합니다.

사용법 (저장소 최상위 폴더에서):
    pip install pyserial
    python scripts/log_ppg.py COM3 60 ppg.csv          # Windows 예시
    python scripts/log_ppg.py /dev/cu.usbmodem1101 60 ppg.csv   # Mac 예시

인자: 포트이름  측정초(기본 60)  저장파일(기본 ppg.csv)
Arduino 시리얼 모니터/플로터가 켜져 있으면 포트가 겹쳐서 열리지 않으니 먼저 닫으세요.
"""
import sys
import time

import serial

EXPECTED_FS = 100  # 펌웨어의 SAMPLE_RATE / SAMPLE_AVERAGE 와 같아야 합니다

port = sys.argv[1]
seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 60
out = sys.argv[3] if len(sys.argv) > 3 else "ppg.csv"

count = 0
with serial.Serial(port, 115200, timeout=1) as s, open(out, "w", encoding="utf-8") as f:
    s.reset_input_buffer()
    f.write("ir\n")
    start = time.time()
    while time.time() - start < seconds:
        line = s.readline().decode(errors="ignore").strip()
        if not line:
            continue
        if not line.lstrip("-").isdigit():
            print("[센서 메시지]", line)  # ready / not found 등
            continue
        f.write(line + "\n")
        count += 1
        if count % 10 == 0:  # 초당 10번만 화면에 찍기
            print(f"{time.time() - start:6.1f}s  ir={line}")
    elapsed = time.time() - start

print(f"\n저장: {out} / 샘플 {count}개 / 실제 {count / elapsed:.1f} Hz (기대 {EXPECTED_FS} Hz)")
if abs(count / elapsed - EXPECTED_FS) > 0.05 * EXPECTED_FS:
    print("경고: 속도가 기대값과 다릅니다. 샘플이 빠졌거나 SAMPLE_AVERAGE 설정이 다를 수 있어요.")
