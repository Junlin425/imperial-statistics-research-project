import os
import tempfile
import unittest
from pathlib import Path

from project_paths import DATA_DIR, FIGURES_DIR, SCRIPT_DIR, script_output


class ProjectPathTests(unittest.TestCase):
    def test_paths_do_not_depend_on_current_working_directory(self):
        original_directory = Path.cwd()

        with tempfile.TemporaryDirectory() as temporary_directory:
            try:
                os.chdir(temporary_directory)
                output = script_output("example.csv")
            finally:
                os.chdir(original_directory)

        self.assertEqual(output, SCRIPT_DIR / "example.csv")
        self.assertEqual(DATA_DIR, SCRIPT_DIR.parent / "data")
        self.assertEqual(FIGURES_DIR, SCRIPT_DIR.parent / "figures")

    def test_script_output_rejects_parent_directory_escape(self):
        with self.assertRaises(ValueError):
            script_output("../outside.csv")


if __name__ == "__main__":
    unittest.main()
