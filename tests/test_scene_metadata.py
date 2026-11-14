import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType('scene_metadata_test_package')
package.__path__ = [str(ROOT)]
nodes = types.ModuleType(package.__name__ + '.nodes')
nodes._st_load = nodes._st_save = nodes._streams_from_latent = None
spec = importlib.util.spec_from_file_location(package.__name__ + '.project_nodes', ROOT / 'project_nodes.py')
module = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {package.__name__: package, nodes.__name__: nodes,
                             'folder_paths': types.ModuleType('folder_paths'),
                             'torch': types.ModuleType('torch'), 'av': types.ModuleType('av')}):
    spec.loader.exec_module(module)


class SceneMetadataTests(unittest.TestCase):
    def prompt(self, seed):
        packages = json.dumps({'MSG_A': {'seed': seed, 'transition': 'cut'}})
        return {'1': {'class_type': 'PrimitiveString', 'inputs': {'value': 'MSG_A'}},
                '2': {'class_type': 'FantasticSeeds', 'inputs': {'seed': 42}},
                '3': {'class_type': 'DynamicScriptNode',
                      'inputs': {'code': f'PACKAGES = _json.loads({packages!r})'}}}

    def test_invalid_optional_package_seed_keeps_resolved_seed(self):
        for seed in ('', 'invalid', [], {}, float('inf')):
            with self.subTest(seed=seed):
                self.assertEqual(module._scene_fields_from_prompt(self.prompt(seed)), ('MSG_A', 'cut', '42'))

    def test_valid_package_seed_overrides_resolved_seed(self):
        for seed in (0, 123, '456'):
            with self.subTest(seed=seed):
                self.assertEqual(module._scene_fields_from_prompt(self.prompt(seed)), ('MSG_A', 'cut', str(seed)))

    def test_save_metadata_uses_only_its_upstream_branch(self):
        prompt = self.prompt(123)
        prompt['3']['inputs']['input_0'] = ['1', 0]
        prompt['10'] = {'class_type': 'H3ProjectSave', 'inputs': {'images': ['3', 0]}}
        other = self.prompt(456)
        other['1']['inputs']['value'] = 'MSG_B'
        other['3']['inputs']['code'] = other['3']['inputs']['code'].replace('MSG_A', 'MSG_B')
        other['3']['inputs']['input_0'] = ['21', 0]
        prompt.update({str(int(k) + 20): v for k, v in other.items()})
        prompt['30'] = {'class_type': 'H3ProjectSave', 'inputs': {'images': ['23', 0]}}
        self.assertEqual(module._scene_fields_from_prompt(prompt, '10'), ('MSG_A', 'cut', '123'))
        self.assertEqual(module._scene_fields_from_prompt(prompt, '30'), ('MSG_B', 'cut', '456'))


if __name__ == '__main__':
    unittest.main()
