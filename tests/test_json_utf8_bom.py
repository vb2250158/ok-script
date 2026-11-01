import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ok.util.config import Config
from ok.util.file import read_json_file


class TestReadJsonUtf8Bom(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory(prefix='ok-json-bom-')
        self.root = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _write_bom_json(self, name, payload):
        path = self.root / name
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        path.write_bytes(b'\xef\xbb\xbf' + body)
        return path

    def test_read_json_file_preserves_values_when_file_has_utf8_bom(self):
        payload = {
            'Saved Accounts': ['195****2369', '176****5513'],
            'Exit After Task': False,
        }
        path = self._write_bom_json('MultiAccountDailyTask.json', payload)

        self.assertEqual(path.read_bytes()[:3], b'\xef\xbb\xbf')
        self.assertEqual(payload, read_json_file(str(path)))

    def test_config_loads_bom_file_without_resetting_saved_accounts(self):
        payload = {
            'Saved Accounts': ['195****2369', '176****5513'],
            'Exit After Task': False,
        }
        configs = self.root / 'configs'
        configs.mkdir()
        path = configs / 'MultiAccountDailyTask.json'
        path.write_bytes(b'\xef\xbb\xbf' + json.dumps(payload, ensure_ascii=False).encode('utf-8'))

        with patch('ok.util.config.get_relative_path', side_effect=lambda *parts: str(self.root.joinpath(*parts))):
            config = Config('MultiAccountDailyTask', {
                'Saved Accounts': [],
                'Exit After Task': False,
            })

        self.assertEqual(payload['Saved Accounts'], config['Saved Accounts'])
        self.assertFalse(any(configs.glob('*.corrupt-*')))

    def test_config_backs_up_existing_corrupt_file_before_reset(self):
        configs = self.root / 'configs'
        configs.mkdir()
        path = configs / 'BrokenTask.json'
        original = b'{"Saved Accounts": ["keep-me"],'
        path.write_bytes(original)

        with patch('ok.util.config.get_relative_path', side_effect=lambda *parts: str(self.root.joinpath(*parts))):
            config = Config('BrokenTask', {
                'Saved Accounts': [],
                'Exit After Task': False,
            })

        backups = sorted(configs.glob('BrokenTask.json.corrupt-*'))
        self.assertEqual(1, len(backups))
        self.assertEqual(original, backups[0].read_bytes())
        self.assertEqual([], config['Saved Accounts'])
        self.assertFalse(config['Exit After Task'])
        rewritten = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual([], rewritten['Saved Accounts'])


if __name__ == '__main__':
    unittest.main()