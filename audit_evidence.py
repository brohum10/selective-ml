"""Verify a Release Lab evidence ZIP without extracting or executing its files.

Hashes detect accidental changes relative to the included manifest. They are
NOT signatures: a malicious party can replace both the files and their hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile
import zlib

MAX_FILES = 200
MAX_BYTES = 64_000_000
HASH = re.compile(r'[0-9a-f]{64}\Z')


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key: '+key)
        result[key] = value
    return result


def safe_name(name):
    if not isinstance(name, str):
        return False
    path = PurePosixPath(name)
    return (isinstance(name, str) and bool(name) and not path.is_absolute()
            and '\\' not in name and ':' not in name
            and all(p not in ('', '.', '..') for p in name.split('/'))
            and all(ord(c) >= 32 for c in name))


def audit(path):
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError('archive exceeds 64 MB')
    with zipfile.ZipFile(path) as archive:
        items = archive.infolist()
        if not 1 <= len(items) <= MAX_FILES or sum(i.file_size for i in items) > MAX_BYTES:
            raise ValueError('archive exceeds file-count or expanded-size limit')
        names = [i.filename for i in items]
        if len(names) != len(set(names)):
            raise ValueError('duplicate archive member')
        for item in items:
            if not safe_name(item.filename) or item.is_dir() or item.flag_bits & 1:
                raise ValueError('unsafe archive member')
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError('symlinks are not allowed')
        if 'manifest.json' not in names:
            raise ValueError('manifest missing')
        manifest = json.loads(archive.read('manifest.json'), object_pairs_hook=unique_object)
        if type(manifest) is not dict or type(manifest.get('version')) is not int or manifest.get('version') != 1 or type(manifest.get('sha256')) is not dict:
            raise ValueError('unsupported manifest')
        expected = manifest['sha256']
        for name, digest in expected.items():
            if not safe_name(name) or not isinstance(digest, str) or not HASH.fullmatch(digest):
                raise ValueError('invalid manifest entry')
        if set(names) != set(expected) | {'manifest.json'} or 'manifest.json' in expected:
            raise ValueError('archive members differ from manifest')
        for name, digest in expected.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError('SHA-256 mismatch: '+name)
        # Require the core evidence, not merely an arbitrary self-consistent ZIP.
        required = {'summary.json', 'protocol.json'}
        if not required.issubset(expected):
            raise ValueError('core evidence missing')
        report = json.loads(archive.read('summary.json'), object_pairs_hook=unique_object)
        protocol = json.loads(archive.read('protocol.json'), object_pairs_hook=unique_object)
        if type(report) is not dict or report.get('version') != 2 or report.get('protocol') != protocol:
            raise ValueError('summary/protocol mismatch')
        canonical = json.dumps(protocol, sort_keys=True, separators=(',', ':'), allow_nan=False)
        if hashlib.sha256(canonical.encode()).hexdigest() != report.get('protocol_sha256'):
            raise ValueError('protocol fingerprint mismatch')
        if not isinstance(report.get('runs'), list) or not report['runs']:
            raise ValueError('run evidence missing')
        seeds = []
        for run in report['runs']:
            if not isinstance(run, dict):
                raise ValueError('invalid run record')
            seed = run.get('seed')
            if type(seed) is not int or seed < 0 or seed in seeds:
                raise ValueError('invalid or duplicate seed')
            seeds.append(seed)
            for name in ('splits.json', 'baseline-model.json', 'candidate-model.json',
                         'baseline.jsonl', 'candidate.jsonl', 'gate.json', 'control.json', 'predictions.json'):
                if f'seed-{seed}/{name}' not in expected:
                    raise ValueError('per-seed evidence missing: '+name)
        if seeds != protocol.get('seeds'):
            raise ValueError('planned seed evidence incomplete')
        return {'status': 'verified', 'files_checked': len(expected), 'expanded_bytes': sum(i.file_size for i in items),
                'seeds': seeds, 'protocol_sha256': report['protocol_sha256'],
                'scope': 'Integrity relative to an unsigned manifest; not authenticity, statistical validity or production safety.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.archive)
    except (OSError, ValueError, KeyError, TypeError, RecursionError, zipfile.BadZipFile, RuntimeError, NotImplementedError, zlib.error) as error:
        parser.exit(2, f'evidence audit: {error}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
