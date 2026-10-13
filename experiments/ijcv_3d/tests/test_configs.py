"""Configuration-level checks for capacity-controlled IJCV variants."""
import json
from pathlib import Path
import unittest


class ModelControlTests(unittest.TestCase):
    def test_dense_control_changes_only_identity_and_memory_source(self):
        root = Path(__file__).parent.parent
        baseline = json.loads((root / 'configs/complete.json').read_text())
        dense = json.loads((root / 'configs/dense_image.json').read_text())
        ignored = {'identity', 'transform_memory'}
        self.assertEqual({key: value for key, value in baseline.items() if key not in ignored},
                         {key: value for key, value in dense.items() if key not in ignored})
        self.assertEqual(dense['model_variant'], 'complete')
        self.assertEqual(dense['transform_memory'], 'dense_image')

    def test_3d_only_control_changes_only_supervision(self):
        root = Path(__file__).parent.parent
        baseline = json.loads((root / 'configs/complete.json').read_text())
        control = json.loads((root / 'configs/3d_only.json').read_text())
        ignored = {'identity', 'image_supervision', 'paired_matching'}
        self.assertEqual({key: value for key, value in baseline.items() if key not in ignored},
                         {key: value for key, value in control.items() if key not in ignored})
        self.assertFalse(control['image_supervision'])
        self.assertFalse(control['paired_matching'])

    def test_generic_control_changes_only_gradient_path(self):
        root = Path(__file__).parent.parent
        baseline = json.loads((root / 'configs/complete.json').read_text())
        control = json.loads((root / 'configs/generic_tokens.json').read_text())
        ignored = {'identity', 'transform_memory'}
        self.assertEqual({key: value for key, value in baseline.items() if key not in ignored},
                         {key: value for key, value in control.items() if key not in ignored})
        self.assertEqual(control['transform_memory'], 'generic_tokens')


if __name__ == '__main__':
    unittest.main()
