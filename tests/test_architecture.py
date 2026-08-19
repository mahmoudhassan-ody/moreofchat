import subprocess


def test_import_contracts_hold():
    result = subprocess.run(["uv", "run", "lint-imports"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout
