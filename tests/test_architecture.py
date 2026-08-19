# TEMPORARY: deliberate ruff E501 violation to prove the notify-on-red path actually fires; reverted in the very next commit.
import subprocess


def test_import_contracts_hold():
    result = subprocess.run(["uv", "run", "lint-imports"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout
