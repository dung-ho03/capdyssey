"""USB wire format and strict non-overlapping 10-second windows."""
from collections import deque
from dataclasses import dataclass
import math

HEADER = '#CAPDYSSEY,1,100,red,g'
FS = 100
SAMPLES = 1000


@dataclass(frozen=True)
class Sample:
    seq: int
    time_s: float
    ppg: float
    ax: float
    ay: float
    az: float


def parse_sample(line):
    parts = line.strip().split(',')
    if len(parts) != 7 or parts[0] != 'D':
        raise ValueError('Invalid USB record')
    seq, us = int(parts[1]), int(parts[2])
    values = list(map(float, parts[3:]))
    if seq < 0 or us < 0 or not all(math.isfinite(v) for v in values):
        raise ValueError('Invalid sample values')
    if not 0 <= values[0] <= 262143 or max(abs(v) for v in values[1:]) > 16:
        raise ValueError('ADC or acceleration outside protocol range')
    return Sample(seq, us / 1_000_000, *values)


def validate_payload(p):
    if not isinstance(p, dict) or p.get('protocol') != 1:
        raise ValueError('Protocol version mismatch')
    if p.get('fs') != FS or p.get('channel') != 'red' or p.get('accel_unit') != 'g':
        raise ValueError('Expected 100 Hz raw red PPG and acceleration in g')
    if not isinstance(p.get('window_id'), str) or not 1 <= len(p['window_id']) <= 100:
        raise ValueError('Invalid window id')
    for key in ('raw_ppg', 'times_s', 'accel', 'seq'):
        if not isinstance(p.get(key), list) or len(p[key]) != SAMPLES:
            raise ValueError('Exactly 1000 synchronized samples required')
    try:
        seq = p['seq']
        times = p['times_s']
        for i in range(SAMPLES):
            if type(seq[i]) is not int or seq[i] < 0:
                raise ValueError('Invalid sequence')
            if len(p['accel'][i]) != 3:
                raise ValueError('Three acceleration axes required')
            vals = [times[i], p['raw_ppg'][i], *p['accel'][i]]
            if not all(type(v) in (int, float) and math.isfinite(v) for v in vals):
                raise ValueError('Nonfinite or nonnumeric sample')
            if times[i] < 0 or not 0 <= p['raw_ppg'][i] <= 262143:
                raise ValueError('Invalid timestamp or ADC value')
            if max(abs(v) for v in p['accel'][i]) > 16:
                raise ValueError('Invalid acceleration range')
            if i and (seq[i] != seq[i-1] + 1 or not .004 <= times[i]-times[i-1] <= .016):
                raise ValueError('Missing/reordered sample or invalid acquisition timing')
        if abs((times[-1] - times[0]) - 9.99) > .03:
            raise ValueError('Measured sample rate differs from 100 Hz')
    except (TypeError, KeyError, IndexError) as e:
        raise ValueError('Malformed arrays') from e
    return p


class WindowBuilder:
    def __init__(self, session):
        self.session, self.count = session, 0
        self.samples = []
        self.last = None

    def reset(self):
        self.samples.clear()
        self.last = None

    def add(self, s):
        if self.last is not None:
            if s.seq != self.last.seq + 1 or not .004 <= s.time_s - self.last.time_s <= .016:
                self.reset()
                raise ValueError('USB sample gap: discarded partial window')
        self.last = s
        self.samples.append(s)
        if len(self.samples) < SAMPLES:
            return None
        rows, self.samples = self.samples, []
        self.count += 1
        p = dict(protocol=1, window_id=f'{self.session}:{self.count}', fs=FS,
                 channel='red', accel_unit='g', seq=[s.seq for s in rows],
                 times_s=[s.time_s for s in rows], raw_ppg=[s.ppg for s in rows],
                 accel=[[s.ax, s.ay, s.az] for s in rows])
        return validate_payload(p)


def replace_pending(queue, item):
    """One pending window; keep the newest rather than accumulate latency."""
    from queue import Empty, Full
    try:
        queue.put_nowait(item)
        return None
    except Full:
        try:
            dropped = queue.get_nowait()
        except Empty:
            dropped = None
        queue.put_nowait(item)
        return dropped
