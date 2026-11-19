"""Authenticated, bounded-latency requests from the desktop worker."""
import time
from urllib.parse import urlparse


class AnalysisClient:
    def __init__(self, url, password, timeout=20):
        parsed = urlparse(url)
        if parsed.scheme != 'https' and not (parsed.scheme == 'http' and parsed.hostname in ('127.0.0.1', 'localhost')):
            raise ValueError('Use HTTPS (or localhost for testing)')
        if not password:
            raise ValueError('Analysis password required')
        from gradio_client import Client
        self.client = Client(url, auth=('capdyssey', password), verbose=False,
                             httpx_kwargs={'timeout': 10.0})
        self.timeout = timeout

    def close(self):
        self.client.close()

    def predict(self, payload, stop_event):
        job = self.client.submit(payload, api_name='/analyze')
        deadline = time.monotonic() + self.timeout
        while not job.done():
            if stop_event.wait(.1) or time.monotonic() >= deadline:
                job.cancel()
                raise TimeoutError('Analysis stopped or timed out')
        result = job.result()
        if not isinstance(result, dict) or result.get('window_id') != payload['window_id']:
            raise ValueError('Response window mismatch')
        if result.get('status') not in ('classified', 'excluded', 'remeasure'):
            raise ValueError('Unknown response status')
        if result['status'] == 'classified' and result.get('label') not in ('AF', 'non_AF'):
            raise ValueError('Invalid classification label')
        if result['status'] != 'classified' and (result.get('label') is not None or result.get('af_score') is not None):
            raise ValueError('Unclassified window must not have an AF result')
        return result
