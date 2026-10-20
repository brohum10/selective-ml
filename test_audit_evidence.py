import hashlib
import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

import audit_evidence as audit

RECORDED = Path(__file__).parent/'media/release-lab-evidence.zip'


class EvidenceAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.path=Path(self.temp.name)/'test.zip'
        with zipfile.ZipFile(RECORDED) as archive:
            self.files={name:archive.read(name) for name in archive.namelist()}

    def tearDown(self): self.temp.cleanup()

    def write(self, update_manifest=False):
        if update_manifest:
            manifest={'version':1,'sha256':{name:hashlib.sha256(content).hexdigest() for name,content in self.files.items() if name!='manifest.json'}}
            self.files['manifest.json']=json.dumps(manifest).encode()
        with zipfile.ZipFile(self.path,'w',zipfile.ZIP_DEFLATED) as archive:
            for name,content in self.files.items(): archive.writestr(name,content)
        return self.path

    def test_recorded_bundle_verifies_without_extraction(self):
        result=audit.audit(RECORDED)
        self.assertEqual(result['status'],'verified')
        self.assertEqual(result['seeds'],[42,43,44])
        self.assertGreater(result['files_checked'],20)
        self.assertIn('unsigned',result['scope'])
        self.assertEqual(list(Path(self.temp.name).iterdir()),[])

    def test_changed_file_fails_its_hash(self):
        self.files['seed-42/predictions.json']=b'{}'
        with self.assertRaisesRegex(ValueError,'SHA-256 mismatch'): audit.audit(self.write())

    def test_missing_or_extra_file_rejected(self):
        del self.files['protocol.json']
        with self.assertRaisesRegex(ValueError,'differ'): audit.audit(self.write())
        self.files['surprise.txt']=b'unlisted'
        with self.assertRaisesRegex(ValueError,'differ'): audit.audit(self.write())

    def test_duplicate_zip_members_rejected(self):
        self.write()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning)
            with zipfile.ZipFile(self.path,'a') as archive: archive.writestr('manifest.json',b'{}')
        with self.assertRaisesRegex(ValueError,'duplicate archive'): audit.audit(self.path)

    def test_unsafe_member_paths_rejected(self):
        for name in ['../outside','/absolute','a/./b','a//b','C:/windows','a\\b']:
            self.files[name]=b'bad'
            with self.assertRaisesRegex(ValueError,'unsafe'): audit.audit(self.write())
            del self.files[name]

    def test_symlink_rejected_without_following_it(self):
        self.write()
        with zipfile.ZipFile(self.path,'a') as archive:
            item=zipfile.ZipInfo('link');item.create_system=3;item.external_attr=(stat.S_IFLNK|0o777)<<16
            archive.writestr(item,'/etc/passwd')
        with self.assertRaisesRegex(ValueError,'symlinks'): audit.audit(self.path)

    def test_duplicate_manifest_keys_rejected(self):
        self.files['manifest.json']=b'{"version":1,"version":1,"sha256":{}}'
        with self.assertRaisesRegex(ValueError,'duplicate JSON'): audit.audit(self.write())

    def test_manifest_digest_shape_is_validated(self):
        manifest=json.loads(self.files['manifest.json'])
        manifest['sha256']['protocol.json']='not-a-hash'
        self.files['manifest.json']=json.dumps(manifest).encode()
        with self.assertRaisesRegex(ValueError,'invalid manifest'): audit.audit(self.write())

    def test_core_evidence_required_even_with_self_consistent_manifest(self):
        del self.files['seed-43/candidate.jsonl']
        with self.assertRaisesRegex(ValueError,'per-seed'): audit.audit(self.write(update_manifest=True))

    def test_summary_protocol_mismatch_detected(self):
        self.files['protocol.json']=b'{}'
        with self.assertRaisesRegex(ValueError,'summary/protocol'): audit.audit(self.write(update_manifest=True))

    def test_size_and_member_limits(self):
        with patch.object(audit,'MAX_BYTES',1):
            with self.assertRaisesRegex(ValueError,'64 MB'): audit.audit(RECORDED)
        with patch.object(audit,'MAX_FILES',1):
            with self.assertRaisesRegex(ValueError,'file-count'): audit.audit(RECORDED)

    def test_cli_success_and_invalid_exit_codes(self):
        command=[sys.executable,str(Path(audit.__file__))]
        good=subprocess.run([*command,str(RECORDED)],capture_output=True,text=True)
        self.assertEqual(good.returncode,0)
        self.assertEqual(json.loads(good.stdout)['status'],'verified')
        self.path.write_text('not a ZIP')
        bad=subprocess.run([*command,str(self.path)],capture_output=True,text=True)
        self.assertEqual(bad.returncode,2)
        self.assertNotIn('Traceback',bad.stderr)


if __name__ == '__main__': unittest.main()
