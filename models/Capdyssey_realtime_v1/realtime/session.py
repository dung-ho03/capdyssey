"""Independent USB acquisition, recording and network workers."""
from collections import deque
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import queue
import threading
import time
import uuid
from realtime.protocol import HEADER, Sample, WindowBuilder, parse_sample, replace_pending


class Session:
    def __init__(self, directory, *, demo=False, port='', url='', password=''):
        self.id = uuid.uuid4().hex
        self.directory = Path(directory) / (datetime.now().strftime('%Y%m%d_%H%M%S_') + self.id[:6])
        self.directory.mkdir(parents=True, exist_ok=False)
        self.demo, self.port, self.url, self.password = demo, port, url, password
        self.stop_event = threading.Event()
        self.events = queue.Queue(maxsize=100)
        self.pending = queue.Queue(maxsize=1)
        self.plot = deque(maxlen=1000)
        self.lock = threading.Lock()
        self.log_lock = threading.Lock()
        self.last_sample_wall = time.monotonic()
        self.latest_id = None
        self.reader = None
        self.network = None

    def emit(self, kind, value):
        try:
            self.events.put_nowait((kind, value))
        except queue.Full:
            try: self.events.get_nowait()
            except queue.Empty: pass
            self.events.put_nowait((kind, value))

    def log(self, event, **values):
        with self.log_lock:
            with (self.directory / 'events.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(dict(event=event, utc=datetime.now(timezone.utc).isoformat(),
                    demo=self.demo, **values), ensure_ascii=False, allow_nan=False) + '\n')

    def start(self):
        self.log('start', protocol=1, channel='red', fs=100)
        if not self.demo and self.url:
            self.network = threading.Thread(target=self.analyze, daemon=True)
            self.network.start()
        self.reader = threading.Thread(target=self.acquire, daemon=True)
        self.reader.start()

    def stop(self):
        self.stop_event.set()
        self.latest_id = None

    def invalidate(self, message, builder):
        builder.reset()
        self.latest_id = None
        try: self.pending.get_nowait()
        except queue.Empty: pass
        self.emit('invalid', message)

    def accept(self, s, writer, builder):
        writer.writerow([s.seq, s.time_s, s.ppg, s.ax, s.ay, s.az])
        with self.lock:
            self.plot.append(s)
            self.last_sample_wall = time.monotonic()
        try:
            p = builder.add(s)
        except ValueError as e:
            self.invalidate(str(e), builder)
            self.log('discard_partial', reason=str(e))
            return
        if p is None:
            return
        self.latest_id = p['window_id']
        self.emit('window', {'id': p['window_id'], 'start': p['times_s'][0]})
        if self.demo:
            self.emit('demo', '예시 파형 — 실제 AF 판정 없음')
        elif self.url:
            dropped = replace_pending(self.pending, (p, time.monotonic()))
            if dropped:
                self.log('analysis_skipped', window_id=dropped[0]['window_id'], reason='newer_window')
        else:
            self.emit('offline', '수신·저장 중 / 분석 서버 미연결')

    def acquire(self):
        builder = WindowBuilder(self.id)
        try:
            with (self.directory / 'raw.csv').open('w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['seq', 'device_time_s', 'ppg_red', 'ax_g', 'ay_g', 'az_g'])
                if self.demo:
                    begin = time.monotonic()
                    i = 0
                    while not self.stop_event.is_set():
                        t = i / 100
                        a = .3 * math.sin(12*t) if int(t)//10 % 2 else .002 * math.sin(t)
                        self.accept(Sample(i, t, 50000+2000*math.sin(2*math.pi*1.2*t), a, 0, 1), writer, builder)
                        i += 1
                        self.stop_event.wait(max(0, begin + i/100 - time.monotonic()))
                        if i % 100 == 0: f.flush()
                else:
                    self.read_usb(writer, builder, f)
        except Exception as e:
            self.stop_event.set()
            self.emit('fatal', f'수신·저장 중단 ({type(e).__name__}). 포트와 저장 경로를 확인하세요.')
            try: self.log('acquisition_error', error_type=type(e).__name__)
            except OSError: pass
        finally:
            self.latest_id = None
            self.emit('ended', '측정 종료 — 현재 판정 없음')

    def read_usb(self, writer, builder, file):
        import serial
        with serial.Serial(self.port, 115200, timeout=.1, write_timeout=1) as device:
            device.reset_input_buffer()
            device.write(b'START\n')
            ready, buf = False, b''
            last_receive = time.monotonic()
            while not self.stop_event.is_set():
                data = device.read_until(b'\n', 512)
                if not data:
                    if time.monotonic() - last_receive > 3:
                        raise TimeoutError('USB data stopped')
                    continue
                buf += data
                if len(buf) > 1024:
                    raise ValueError('USB line too long')
                if not buf.endswith(b'\n'):
                    continue
                last_receive = time.monotonic()
                line, buf = buf.decode('ascii', errors='strict').strip(), b''
                if line == HEADER:
                    ready = True
                    self.invalidate('장치 연결됨 — 새 10초 구간 수집 중', builder)
                    self.log('device_start')
                    continue
                if line.startswith('#ERROR'):
                    raise RuntimeError('Device sensor error')
                if line.startswith('#GAP'):
                    self.invalidate('센서 데이터 공백 — 해당 구간 폐기', builder)
                    self.log('sensor_gap')
                    continue
                if line.startswith('#'):
                    continue
                if not ready:
                    raise ValueError('Firmware handshake required')
                try:
                    s = parse_sample(line)
                except (ValueError, OverflowError):
                    self.invalidate('잘못된 센서 데이터 — 해당 구간 폐기', builder)
                    self.log('invalid_sample')
                    continue
                self.accept(s, writer, builder)
                if s.seq % 100 == 0: file.flush()
            device.write(b'STOP\n')

    def analyze(self):
        from realtime.client import AnalysisClient
        client = None
        while not self.stop_event.is_set():
            try: payload, received = self.pending.get(timeout=.2)
            except queue.Empty: continue
            if time.monotonic() - received > 15:
                self.emit('network_error', '분석 대기 시간 초과 — 다음 구간 대기')
                continue
            try:
                if client is None:
                    client = AnalysisClient(self.url, self.password)
                result = client.predict(payload, self.stop_event)
                latency = time.monotonic() - received
                current = not self.stop_event.is_set() and payload['window_id'] == self.latest_id
                self.log('result', result=result, latency_s=latency, displayed=current)
                if current:
                    self.emit('result', dict(result, latency_s=latency))
            except Exception as e:
                if client is not None: client.close()
                client = None
                if self.stop_event.is_set(): break
                self.emit('network_error', '분석 연결 실패·시간 초과 — 파형 수신과 저장은 계속됩니다')
                try: self.log('analysis_error', window_id=payload['window_id'], error_type=type(e).__name__)
                except OSError:
                    self.stop_event.set()
                    self.emit('fatal', '결과 저장 실패 — 저장 공간을 확인하세요')
        if client is not None: client.close()
