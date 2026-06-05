from pathlib import Path
import unittest


class ProjectStructureTest(unittest.TestCase):
    def test_expected_project_directories_exist(self):
        project_root = Path(__file__).resolve().parents[1]
        expected_dirs = [
            "data/raw",
            "data/interim",
            "data/processed",
            "notebooks",
            "src",
            "dashboard",
            "models",
            "reports/figures",
            "tests",
        ]

        for relative_path in expected_dirs:
            self.assertTrue((project_root / relative_path).is_dir(), relative_path)


if __name__ == "__main__":
    unittest.main()
