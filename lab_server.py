"""Loopback-only interactive viewer for actual Release Lab experiments.

No user-supplied model execution, remote endpoints, external assets or hosting.
This is a local learning tool, not an authenticated production service.
"""
from __future__ import annotations

import argparse
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import secrets
import threading
from urllib.parse import urlparse
import uuid
import zipfile

import release_lab
import diagnostics

ROOT = Path(__file__).resolve().parent
MAX_BODY = 1024


class Lab:
    def __init__(self, comparator, output_root):
        self.comparator = Path(comparator).resolve()
        self.output_root = Path(output_root).resolve()
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.Lock()
        self.worker = None
        self.state = {'status': 'idle', 'events': [], 'result': None, 'diagnostics': None, 'error': None}
        self.current_output = None

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.state))

    def start(self):
        with self.lock:
            if self.state['status'] == 'running':
                return False
            folder = self.output_root/('run-'+uuid.uuid4().hex)
            self.state = {'status': 'running', 'events': [], 'result': None, 'diagnostics': None, 'error': None}
            self.current_output = folder
            self.worker = threading.Thread(target=self._run, args=(folder,), daemon=True)
            self.worker.start()
        return True

    def _progress(self, event):
        with self.lock:
            self.state['events'].append(event)

    def _run(self, folder):
        try:
            result = release_lab.run(folder, self.comparator, self._progress)
            analysis = diagnostics.summarize(result)
            with self.lock:
                self.state.update(status='complete', result=result, diagnostics=analysis)
        except Exception:
            with self.lock:
                self.state.update(status='failed', error='Experiment failed. Inspect the local server log; partial output is not a completed evidence bundle.')
            import traceback
            traceback.print_exc()

    def evidence_zip(self):
        with self.lock:
            if self.state['status'] != 'complete':
                return None
            folder = self.current_output
            # Hold the lock while copying this small bundle so a new run cannot
            # substitute its path between the completion check and archive read.
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(folder.rglob('*')):
                    if path.is_file() and not path.is_symlink():
                        archive.write(path, str(path.relative_to(folder)))
            return buffer.getvalue()


class Handler(BaseHTTPRequestHandler):
    def send(self, status, body, mime='application/json; charset=utf-8', download=None):
        if not isinstance(body, bytes):
            body = json.dumps(body, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        if download:
            self.send_header('Content-Disposition', f'attachment; filename="{download}"')
        self.end_headers()
        self.wfile.write(body)

    def valid_host(self):
        return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}')

    def do_GET(self):
        if not self.valid_host():
            return self.send(403, {'error': 'loopback host required'})
        path = urlparse(self.path).path
        lab = self.server.lab
        if path == '/api/state':
            return self.send(200, lab.snapshot())
        if path == '/api/session':
            return self.send(200, {'token': lab.token})
        if path == '/api/report':
            result = lab.snapshot()
            if result['status'] != 'complete':
                return self.send(409, {'error': 'no completed experiment'})
            return self.send(200, result['result'], download='release-lab-report.json')
        if path == '/api/diagnostics':
            state = lab.snapshot()
            if state['status'] != 'complete' or state['diagnostics'] is None:
                return self.send(409, {'error': 'no completed diagnostics'})
            return self.send(200, state['diagnostics'], download='release-lab-diagnostics.json')
        if path == '/api/evidence.zip':
            content = lab.evidence_zip()
            return self.send(409, {'error': 'no completed experiment'}) if content is None else self.send(200, content, 'application/zip', 'release-lab-evidence.zip')
        assets = {'/': ('index.html', 'text/html; charset=utf-8'),
                  '/lab.js': ('lab.js', 'text/javascript; charset=utf-8'),
                  '/lab.css': ('lab.css', 'text/css; charset=utf-8')}
        if path in assets:
            name, mime = assets[path]
            return self.send(200, (ROOT/'lab'/name).read_bytes(), mime)
        return self.send(404, {'error': 'not found'})

    def do_POST(self):
        if not self.valid_host():
            return self.send(403, {'error': 'loopback host required'})
        origin = self.headers.get('Origin')
        if origin is not None and origin not in (f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'):
            return self.send(403, {'error': 'same-origin request required'})
        if self.path != '/api/run':
            return self.send(404, {'error': 'not found'})
        if not hmac.compare_digest(self.headers.get('X-CSRF-Token', ''), self.server.lab.token):
            return self.send(403, {'error': 'invalid session token'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length < 0 or length > MAX_BODY:
                return self.send(413, {'error': 'request too large'})
            if self.headers.get_content_type() != 'application/json':
                return self.send(415, {'error': 'application/json required'})
            body = json.loads(self.rfile.read(length) or b'{}')
            if type(body) is not dict or body:
                return self.send(400, {'error': 'run uses the fixed checked-in protocol; expected an empty object'})
        except (ValueError, UnicodeDecodeError):
            return self.send(400, {'error': 'invalid request'})
        if not self.server.lab.start():
            return self.send(409, {'error': 'an experiment is already running'})
        return self.send(202, {'status': 'running'})

    def log_message(self, *args):
        pass


def make_server(comparator, output_root, port=0):
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.lab = Lab(comparator, output_root)
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaldelta', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--output-root', type=Path, default=ROOT/'results/release-lab')
    args = parser.parse_args()
    if not args.evaldelta.is_file():
        parser.error('--evaldelta must point to a trusted local comparator')
    server = make_server(args.evaldelta, args.output_root, args.port)
    print(f'Release Lab: http://127.0.0.1:{server.server_port}', flush=True)
    print('Local-only. Each click trains and evaluates real models. Ctrl+C to stop.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
