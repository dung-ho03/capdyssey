import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from download_longterm_af import fetch, ranged_download

class LongtermDownloadTests(unittest.TestCase):
    def test_range_resume_keeps_existing_prefix(self):
        with tempfile.TemporaryDirectory() as folder:
            partial=Path(folder)/'a.part'; partial.write_bytes(b'ab')
            response=MagicMock(); response.__enter__.return_value=response
            response.status_code=206; response.headers={'Content-Range':'bytes 2-5/6'}
            response.iter_content.return_value=[b'cdef']
            with patch('download_longterm_af.requests.get',return_value=response) as request:
                ranged_download({'path':'a','url':'https://example.invalid','bytes':6},partial)
                self.assertEqual(request.call_args.kwargs['headers']['Range'],'bytes=2-5')
            self.assertEqual(partial.read_bytes(),b'abcdef')

    def test_range_ignored_by_server_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            partial=Path(folder)/'a.part'
            response=MagicMock(); response.__enter__.return_value=response
            response.status_code=200
            with patch('download_longterm_af.requests.get',return_value=response):
                with self.assertRaises(ValueError): ranged_download({'path':'a','url':'https://example.invalid','bytes':6},partial)
            self.assertFalse(partial.exists())

    def test_verified_file_skips_network(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'a.mat').write_bytes(b'abc')
            item={'path':'a.mat','url':'https://example.invalid','bytes':3,'md5':hashlib.md5(b'abc').hexdigest()}
            with patch('download_longterm_af.requests.get') as request:
                result=fetch(item,root)
                request.assert_not_called()
            self.assertEqual(result['sha256'],hashlib.sha256(b'abc').hexdigest())

    def test_corrupt_download_is_not_promoted(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            item={'path':'a.mat','url':'https://example.invalid','bytes':3,'md5':hashlib.md5(b'abc').hexdigest()}
            response=MagicMock(); response.__enter__.return_value=response
            response.iter_content.return_value=[b'bad']
            with patch('download_longterm_af.requests.get',return_value=response):
                with self.assertRaises(ValueError): fetch(item,root)
            self.assertFalse((root/'a.mat').exists())

if __name__=='__main__': unittest.main()
