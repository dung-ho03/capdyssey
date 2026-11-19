"""Validated server adapter. The original model remains unchanged."""
from pathlib import Path
import sys
import threading
from realtime.protocol import validate_payload


class AnalysisService:
    def __init__(self, bundle, *, model=None):
        self.lock = threading.Lock()
        if model is None:
            bundle = Path(bundle).resolve()
            sys.path.insert(0, str(bundle / 'scripts'))
            from quality_af_imu import MotionQualityAF
            from quality_af_v2 import QualityAF
            model = MotionQualityAF(bundle)
            # Fail early on missing GPU, dependency or checksum mismatch.
            model._ppg_model = QualityAF(bundle)
        self.model = model

    def predict(self, payload):
        p = validate_payload(payload)
        # Serialized access prevents GPU model races across callers.
        with self.lock:
            r = self.model.predict(p['raw_ppg'], p['fs'], accel=p['accel'],
                imu_times_s=p['times_s'], ppg_start_s=p['times_s'][0], accel_unit=p['accel_unit'])
        return dict(r, window_id=p['window_id'], start_s=p['times_s'][0],
                    end_s=p['times_s'][0] + 10, protocol=1)


def build_app(service):
    import gradio as gr
    with gr.Blocks(title='Capdyssey research analysis') as app:
        gr.Markdown('Capdyssey: 10-second PPG + IMU. Research prototype; device validation pending.')
        payload = gr.JSON(label='Window')
        output = gr.JSON(label='Result')
        gr.Button('Analyze').click(service.predict, inputs=payload, outputs=output,
                                   api_name='analyze', concurrency_limit=1)
    return app
