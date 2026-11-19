"""Optional local GPU server. Colab notebook uses the same service."""
from getpass import getpass
from pathlib import Path
from realtime.analysis import AnalysisService, build_app

if __name__ == '__main__':
    password = getpass('Analysis password (at least 12 characters): ')
    if len(password) < 12:
        raise ValueError('Use at least 12 characters')
    service = AnalysisService(Path(__file__).resolve().parent / 'model_bundle')
    build_app(service).queue(max_size=4).launch(server_name='127.0.0.1',
        server_port=7860, share=False, auth=('capdyssey', password))
