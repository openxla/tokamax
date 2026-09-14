# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import pathlib
from absl.testing import absltest

from tokamax._src.ops import op as op_lib
from tokamax._src.ops import registry

class RegistryTest(absltest.TestCase):

    def test_all_ops_registered_by_folder_name(self):
        ops_dir = pathlib.Path(registry.__file__).parent
        registered_names = set(reg.name for reg in registry.OPS)

        for path in ops_dir.iterdir():
            if path.is_dir() and not path.name.startswith('_') and path.name != 'experimental':
                if (path / 'base.py').exists() and (path / 'api.py').exists():
                    self.assertIn(
                        path.name,
                        registered_names,
                        f"Op folder '{path.name}' is missing from registry.OPS"
                    )

    def test_all_base_classes_registered(self):
        _ops = registry.OPS
        registered_base_classes = {op.base_class for op in _ops}

        def _get_all_subclasses(cls):
            visited = set()
            stack = [cls]
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    stack.extend(current.__subclasses__())
            return visited

        for subclass in _get_all_subclasses(op_lib.Op):
           # Simple heuristic: only test direct concrete implementations inside their base modules
           # Skip intermediate abstract classes or VJPs
           name = subclass.__name__
           if name.endswith('Vjp') or 'Base' in name or name == 'Op':
               continue
           # If the class defines an initialization or acts as a Base
           if subclass.__module__.endswith('.base'):
               self.assertIn(
                   subclass,
                   registered_base_classes,
                   f"Op subclass {subclass.__name__} in module {subclass.__module__} must be registered in registry.OPS!"
               )

if __name__ == '__main__':
    absltest.main()
