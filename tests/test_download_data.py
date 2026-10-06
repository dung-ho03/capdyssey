import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('download_data', Path(__file__).resolve().parents[1] / 'scripts/download_data.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class DownloadTests(unittest.TestCase):
    def test_traversal_rejected(self):
        for name in ['../secret', '/absolute', 'C:/file', 'a/../../b', 'a\\b']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                module.safe_path(Path(tempfile.gettempdir()), name)

    def test_verified_existing_file_skips_network(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data'
            path.write_bytes(b'confirmed')
            with patch.object(module.requests, 'Session', side_effect=AssertionError('network called')):
                module.download('https://example.invalid/data', path, module.digest(path))

    def test_manifest_preserves_checksums_and_filters_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            checksum = 'a' * 64
            (folder / 'SHA256SUMS.txt').write_text(f'{checksum} record.dat\n{checksum} csv/record.csv\n')
            with patch.object(module, 'download'):
                result = module.entries({'base_url': 'https://example.invalid/', 'checksums_sha256': checksum, 'root_only': True}, folder)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0][2], checksum)
            self.assertEqual(result[0][1].name, 'record.dat')

if __name__ == '__main__':
    unittest.main()
