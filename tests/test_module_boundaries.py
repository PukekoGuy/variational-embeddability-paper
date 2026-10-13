import ast
import importlib
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class ModuleBoundaries(unittest.TestCase):
    def test_reusable_packages_exist(self):
        for name in ('solvers', 'experiments', 'plotting', 'paper'):
            self.assertTrue((ROOT / name / '__init__.py').exists(), name)

    def test_solver_imports_are_mathematical(self):
        files = list((ROOT / 'solvers').glob('*.py'))
        self.assertGreater(len(files), 5)
        for file in files:
            for node in ast.walk(ast.parse(file.read_text())):
                names = ([node.module or ''] if isinstance(node, ast.ImportFrom)
                         else [n.name for n in node.names] if isinstance(node, ast.Import) else [])
                for name in names:
                    self.assertNotIn(name.split('.')[0], ('experiments', 'plotting', 'matplotlib'), str(file))

    def test_imports_do_not_create_figures_or_files(self):
        import matplotlib.pyplot as plt
        before = set(ROOT.rglob('*'))
        for folder in ('solvers', 'experiments', 'plotting', 'paper'):
            for file in (ROOT / folder).glob('*.py'):
                importlib.import_module(folder + '.' + file.stem)
        after = set(ROOT.rglob('*'))
        created = [p for p in after - before if '__pycache__' not in p.parts]
        self.assertEqual(created, [])
        self.assertEqual(plt.get_fignums(), [])
