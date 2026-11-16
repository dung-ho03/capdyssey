"""Save timestamped firmware CSV packets; never invent missing device timestamps."""
import argparse
import csv
import json
import math
import time
from pathlib import Path

FIELDS = ['sample_index', 'timestamp_us', 'ppg_red', 'ppg_ir', 'ppg_green']
IMU = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']


def parse_packet(line):
    parts = next(csv.reader([line]))
    if len(parts) not in (5, 11):
        raise ValueError('Expected 5 or 11 CSV values (sample_index,timestamp_us,red,ir,green[,IMU])')
    sample, micros = int(parts[0]), int(parts[1])
    values = [float(value) for value in parts[2:]]
    if sample < 0 or micros < 0 or not all(math.isfinite(v) for v in values):
        raise ValueError('Nonfinite or negative index/timestamp')
    row = dict(zip(FIELDS + IMU, [sample, micros, *values]))
    row['timestamp_s'] = micros / 1_000_000
    return row


def run(args):
    import serial
    from quality_only import sampling_rate
    sampling_rate(args.fs)
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError('seconds must be positive')
    args.out.mkdir(parents=True, exist_ok=False)
    records, errors = 0, 0
    start = time.monotonic()
    with serial.Serial(args.port, args.baud, timeout=1) as port, \
            (args.out / 'samples.csv').open('w', newline='', encoding='utf-8') as file, \
            (args.out / 'serial_messages.txt').open('w', encoding='utf-8') as log:
        writer = csv.DictWriter(file, fieldnames=FIELDS + ['timestamp_s', 'host_elapsed_s'] + IMU)
        writer.writeheader()
        while time.monotonic() - start < args.seconds:
            raw = port.readline()
            if not raw:
                continue
            line = raw.decode('utf-8', errors='replace').strip()
            if not raw.endswith(b'\n'):
                log.write('PARTIAL ' + repr(line) + '\n'); errors += 1; continue
            try:
                row = parse_packet(line)
            except (ValueError, csv.Error):
                log.write(line + '\n'); errors += 1; continue
            row['host_elapsed_s'] = time.monotonic() - start
            writer.writerow(row)
            records += 1
    metadata = dict(filename='samples.csv', subject_id=args.subject, device_id=args.device,
                    session_id=args.session, activity=args.activity, channel=args.channel, fs=args.fs,
                    sensor_settings=args.sensor_settings)
    with (args.out / 'sessions.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(metadata))
        writer.writeheader(); writer.writerow(metadata)
    (args.out / 'collection.json').write_text(json.dumps(dict(metadata=metadata,
        valid_packets=records, unparsed_lines=errors, host_elapsed_s=time.monotonic()-start,
        note='Device indices/timestamps preserved. Host elapsed time is NOT acquisition time. Run CSV audit before use.'),
        indent=2, ensure_ascii=False), encoding='utf-8')
    print('Saved', records, 'packets to', args.out)
    if not records:
        raise ValueError('No valid timestamped packets. Single IR values are not accepted; update firmware output.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', required=True)
    p.add_argument('--baud', type=int, default=115200)
    p.add_argument('--seconds', type=float, default=60)
    p.add_argument('--fs', type=int, required=True)
    p.add_argument('--subject', required=True)
    p.add_argument('--device', required=True)
    p.add_argument('--session', required=True)
    p.add_argument('--activity', required=True)
    p.add_argument('--channel', choices=['ppg_red', 'ppg_ir', 'ppg_green'], required=True)
    p.add_argument('--sensor-settings', required=True, help='LED current, pulse width, averaging, firmware version, etc.')
    p.add_argument('--out', type=Path, required=True)
    run(p.parse_args())
