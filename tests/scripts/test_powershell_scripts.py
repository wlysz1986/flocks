import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts"
POWERSHELL_SCRIPTS = ("install.ps1", "run.ps1")


def _parse_script(script_path: Path) -> subprocess.CompletedProcess[str]:
    parser_command = (
        "$tokens = $null; "
        "$errors = $null; "
        "[System.Management.Automation.Language.Parser]::ParseFile("
        f"'{script_path.as_posix()}', "
        "[ref]$tokens, "
        "[ref]$errors"
        ") | Out-Null; "
        "if ($errors.Count -gt 0) { "
        "$errors | ForEach-Object { $_.ToString() }; "
        "exit 1 "
        "}"
    )
    return subprocess.run(
        ["pwsh", "-NoProfile", "-Command", parser_command],
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is required to parse PowerShell scripts")
@pytest.mark.parametrize("script_name", POWERSHELL_SCRIPTS)
def test_powershell_scripts_parse_without_errors(script_name: str) -> None:
    result = _parse_script(SCRIPT_DIR / script_name)
    assert result.returncode == 0, result.stdout or result.stderr
