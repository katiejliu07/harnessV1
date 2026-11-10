"""Cross-platform test helpers: no /bin/sh, true/false, grep or shebangs."""
import os
import sys

PY=f'"{sys.executable}"'
TRUE=f'{PY} -c "raise SystemExit(0)"'
FALSE=f'{PY} -c "raise SystemExit(1)"'


def py_cmd(code):
    """A shell command running one line of Python (double quotes only, for cmd.exe)."""
    assert '"' not in code
    return f'{PY} -c "{code}"'


def fake_cli(path,body):
    """A fake agent CLI as a Python script; returns the list-form command."""
    path.write_text(body)
    return [sys.executable,str(path)]


def can_symlink(tmp_path):
    try:
        os.symlink("target",tmp_path/".symlink_probe")
        os.remove(tmp_path/".symlink_probe")
        return True
    except (OSError,NotImplementedError):
        return False
