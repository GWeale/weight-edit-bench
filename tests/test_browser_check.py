import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests import browser_check


class BrowserCheckSafetyTests(unittest.TestCase):
    def test_macos_rejects_launch_before_side_effects(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "output"
            with patch.object(sys, "argv", ["browser_check", "--chrome", "/unused/chrome",
                                           "--out", str(out)]), \
                    patch.object(sys, "platform", "darwin"), \
                    patch.object(browser_check.subprocess, "run") as launch:
                with self.assertRaises(SystemExit) as error:
                    browser_check.main()
                self.assertEqual(error.exception.code, 2)
                launch.assert_not_called()
                self.assertFalse(out.exists())

    def test_prepare_only_writes_fixture_without_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(sys, "argv", ["browser_check", "--prepare-only", "--out", folder]), \
                    patch.object(sys, "platform", "darwin"), \
                    patch.object(browser_check.subprocess, "run") as launch:
                browser_check.main()
                launch.assert_not_called()
            fixture = (Path(folder) / "fixture.html").read_text()
            self.assertIn("document.documentElement.dataset.qa='passed'", fixture)
            self.assertIn('id="weights"', fixture)
