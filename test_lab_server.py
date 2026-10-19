import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request
import zipfile

import lab_server


class LocalServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.server = lab_server.make_server('unused.py', Path(self.temp.name))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        self.temp.cleanup()

    def call(self, path, method='GET', body=None, headers=None):
        request = urllib.request.Request(self.url+path, data=body, headers=headers or {}, method=method)
        try:
            response = urllib.request.urlopen(request, timeout=3)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, response.headers, response.read()

    def post(self, **extra):
        headers = {'Content-Type':'application/json', 'X-CSRF-Token':self.server.lab.token}
        headers.update(extra)
        return headers

    def test_static_routes_have_security_headers_and_no_external_assets(self):
        for path in ('/', '/lab.css', '/lab.js'):
            status, headers, content = self.call(path)
            self.assertEqual(status, 200)
            self.assertEqual(headers['Cache-Control'], 'no-store')
            self.assertEqual(headers['X-Content-Type-Options'], 'nosniff')
            self.assertIn("script-src 'self'", headers['Content-Security-Policy'])
            self.assertTrue(content)

    def test_state_and_outputs_are_not_successful_before_completion(self):
        status, _, content = self.call('/api/state')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(content)['status'], 'idle')
        self.assertEqual(self.call('/api/report')[0], 409)
        self.assertEqual(self.call('/api/evidence.zip')[0], 409)

    def test_rejects_cross_origin_bad_host_and_missing_session(self):
        self.assertEqual(self.call('/api/run', 'POST', b'{}', {'Content-Type':'application/json'})[0], 403)
        self.assertEqual(self.call('/api/run', 'POST', b'{}', self.post(Origin='https://example.com'))[0], 403)
        self.assertEqual(self.call('/', headers={'Host':'evil.example'})[0], 403)

    def test_rejects_bad_body_types_unknown_fields_and_oversized_payload(self):
        for body in (b'[]', b'{"seed":43}', b'null', b'bad-json'):
            self.assertEqual(self.call('/api/run', 'POST', body, self.post())[0], 400)
        self.assertEqual(self.call('/api/run', 'POST', b' '*1025, self.post())[0], 413)
        self.assertEqual(self.call('/api/run', 'POST', b'{}', self.post(**{'Content-Type':'text/plain'}))[0], 415)

    def test_bounded_worker_allows_only_one_active_run(self):
        entered, release = threading.Event(), threading.Event()
        def fake_run(folder, comparator, progress):
            entered.set()
            release.wait(3)
            return {'version':2}
        with patch.object(lab_server.release_lab, 'run', fake_run):
            try:
                self.assertEqual(self.call('/api/run', 'POST', b'{}', self.post())[0], 202)
                self.assertTrue(entered.wait(1))
                self.assertEqual(self.call('/api/run', 'POST', b'{}', self.post())[0], 409)
            finally:
                release.set()
                self.server.lab.worker.join(3)
        self.assertEqual(self.server.lab.snapshot()['status'], 'complete')

    def test_worker_failure_does_not_publish_partial_evidence(self):
        with patch.object(lab_server.release_lab, 'run', side_effect=ValueError('failure')):
            with patch('traceback.print_exc'):
                self.server.lab.start()
                self.server.lab.worker.join(3)
        self.assertEqual(self.server.lab.snapshot()['status'], 'failed')
        self.assertIsNone(self.server.lab.evidence_zip())

    def test_completed_bundle_download_is_scoped_and_has_real_file_bytes(self):
        folder = Path(self.temp.name)/'done'
        folder.mkdir()
        (folder/'summary.json').write_text('{"complete":true}')
        self.server.lab.current_output = folder
        self.server.lab.state.update(status='complete', result={'version':2})
        status, headers, content = self.call('/api/evidence.zip')
        self.assertEqual(status, 200)
        self.assertIn('attachment;', headers['Content-Disposition'])
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            self.assertEqual(archive.namelist(), ['summary.json'])
            self.assertEqual(archive.read('summary.json'), b'{"complete":true}')

    def test_unknown_and_traversal_paths_are_not_served(self):
        for path in ('/../lab_protocol.json', '/api/file?name=../../secret', '/lab/index.html'):
            self.assertEqual(self.call(path)[0], 404)


if __name__ == '__main__':
    unittest.main()
