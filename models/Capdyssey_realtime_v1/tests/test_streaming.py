import copy
import io
import csv
import queue
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'model_bundle' / 'scripts'))
from realtime.protocol import Sample, WindowBuilder, parse_sample, validate_payload, replace_pending
from realtime.analysis import AnalysisService, build_app
from realtime.client import AnalysisClient
from realtime.session import Session
from quality_af_imu import MotionQualityAF


def payload(moving=False):
    import math
    b = WindowBuilder('test')
    p = None
    for i in range(1000):
        p = b.add(Sample(i, 100+i/100, 50000+1000*math.sin(i/10),
                        .3*math.sin(i/5) if moving else 0, 0, 1))
    return p


class FakePPG:
    def __init__(self): self.calls = 0
    def predict(self, x, fs):
        self.calls += 1
        return dict(status='classified', label='AF', af_score=.8, quality_score=.9, reason=None)


class ProtocolTests(unittest.TestCase):
    def test_serial_parsing(self):
        s = parse_sample('D,1,10000,50000,0,0,1\n')
        self.assertEqual(s.time_s, .01)
        for text in ['D,1,0,nan,0,0,1', 'D,-1,0,5,0,0,1', 'D,1,0,999999,0,0,1', 'broken']:
            with self.assertRaises(ValueError): parse_sample(text)

    def test_exact_window(self):
        p = payload()
        self.assertEqual(p['window_id'], 'test:1')
        self.assertEqual(len(p['raw_ppg']), 1000)
        validate_payload(p)

    def test_gap_resets_window(self):
        b = WindowBuilder('x')
        for i in range(500): b.add(Sample(i,i/100,1,0,0,1))
        with self.assertRaises(ValueError): b.add(Sample(501,5.01,1,0,0,1))
        self.assertEqual(len(b.samples), 0)
        self.assertIsNone(b.add(Sample(502,5.02,1,0,0,1)))

    def test_invalid_payloads(self):
        p = payload()
        for key, value in [('fs', 50), ('channel', 'green'), ('accel_unit', 'm/s2'),
                           ('window_id', ''), ('raw_ppg', [1]), ('accel', [[0,1]]*1000)]:
            bad = dict(p, **{key:value})
            with self.assertRaises(ValueError): validate_payload(bad)
        for key, index, value in [('times_s', 30, 0), ('raw_ppg', 5, float('nan')),
                                  ('seq', 50, 100), ('times_s', 20, 'bad')]:
            bad = copy.deepcopy(p)
            bad[key][index] = value
            with self.assertRaises(ValueError): validate_payload(bad)
        bad = copy.deepcopy(p)
        bad['times_s'] = [100+i*.0101 for i in range(1000)]
        with self.assertRaises(ValueError): validate_payload(bad)

    def test_queue_keeps_latest(self):
        q = queue.Queue(maxsize=1)
        self.assertIsNone(replace_pending(q, 'first'))
        self.assertEqual(replace_pending(q, 'last'), 'first')
        self.assertEqual(q.get_nowait(), 'last')

    def test_motion_and_still_service(self):
        model = MotionQualityAF('unused')
        fake = FakePPG()
        model._ppg_model = fake
        service = AnalysisService(None, model=model)
        r = service.predict(payload(True))
        self.assertEqual(r['status'], 'excluded')
        self.assertIsNone(r['label'])
        self.assertEqual(fake.calls, 0)
        r = service.predict(payload())
        self.assertEqual(r['label'], 'AF')
        self.assertEqual(fake.calls, 1)

    def test_disk_recording_and_partial_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = Session(tmp, demo=True)
            buffer = io.StringIO()
            writer = csv.writer(buffer)
            builder = WindowBuilder(s.id)
            s.accept(Sample(0,0,50000,0,0,1), writer, builder)
            self.assertIn('50000', buffer.getvalue())
            s.latest_id = 'old'
            s.pending.put(('old', 0))
            s.invalidate('gap', builder)
            self.assertIsNone(s.latest_id)
            self.assertTrue(s.pending.empty())

    def test_usb_reader_with_partial_lines(self):
        from collections import deque
        from realtime.protocol import HEADER
        with tempfile.TemporaryDirectory() as tmp:
            s = Session(tmp)
            lines = deque([(HEADER + '\n').encode()])
            for i in range(1000):
                packet = f'D,{i},{i*10000},50000,0,0,1\n'.encode()
                lines.extend([packet[:8], packet[8:]])
            class Device:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def reset_input_buffer(self): pass
                def write(self, x): return len(x)
                def read_until(self, *args):
                    if lines: return lines.popleft()
                    s.stop_event.set()
                    return b''
            with patch('serial.Serial', return_value=Device()):
                f = io.StringIO()
                s.read_usb(csv.writer(f), WindowBuilder(s.id), f)
            self.assertEqual(len(f.getvalue().splitlines()), 1000)
            self.assertIsNotNone(s.latest_id)

    def test_late_result_not_displayed(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = Session(tmp, url='https://example.test', password='test')
            p = payload()
            s.latest_id = 'newer'
            s.pending.put((p, __import__('time').monotonic()))
            class FakeClient:
                def __init__(self, *args): pass
                def close(self): pass
                def predict(self, request, stop):
                    # A stopped session is also forbidden from showing an old result.
                    stop.set()
                    return dict(window_id=request['window_id'], status='classified', label='AF')
            with patch('realtime.client.AnalysisClient', FakeClient): s.analyze()
            self.assertTrue(s.events.empty())
            self.assertIn('"displayed": false', (s.directory/'events.jsonl').read_text())


class TransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import socket
        sock = socket.socket()
        sock.bind(('127.0.0.1', 0))
        cls.port = sock.getsockname()[1]
        sock.close()
        model = MotionQualityAF('unused')
        model._ppg_model = FakePPG()
        cls.app = build_app(AnalysisService(None, model=model))
        cls.app.queue(max_size=4).launch(server_name='127.0.0.1', server_port=cls.port,
            share=False, auth=('capdyssey', 'test-only-password'), prevent_thread_lock=True,
            quiet=True)
        cls.url = f'http://127.0.0.1:{cls.port}'

    @classmethod
    def tearDownClass(cls): cls.app.close()

    def test_authenticated_request(self):
        c = AnalysisClient(self.url, 'test-only-password')
        r = c.predict(payload(), threading.Event())
        self.assertEqual(r['label'], 'AF')
        self.assertEqual(r['window_id'], 'test:1')
        r = c.predict(payload(True), threading.Event())
        self.assertEqual(r['status'], 'excluded')
        c.close()

    def test_wrong_password_rejected(self):
        with self.assertRaises(Exception): AnalysisClient(self.url, 'incorrect')

    def test_timeout_cancels_job(self):
        from unittest.mock import Mock
        c = AnalysisClient.__new__(AnalysisClient)
        c.client = Mock()
        c.timeout = 0
        job = c.client.submit.return_value
        job.done.return_value = False
        with self.assertRaises(TimeoutError): c.predict(payload(), threading.Event())
        job.cancel.assert_called_once()

    def test_response_mismatch_rejected(self):
        from unittest.mock import Mock
        c = AnalysisClient.__new__(AnalysisClient)
        c.client = Mock()
        c.timeout = 1
        job = c.client.submit.return_value
        job.done.return_value = True
        job.result.return_value = dict(window_id='other')
        with self.assertRaises(ValueError): c.predict(payload(), threading.Event())


if __name__ == '__main__': unittest.main()
