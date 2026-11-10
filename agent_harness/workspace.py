"""Give every ladder rung the same starting files.

A rung that fails its check can leave half-finished edits behind. Without a
reset the next (stronger) rung starts from that broken state without being
told, and a later pass can owe credit to an earlier rung's edits. The ladder's
measured accuracy assumes each rung starts fresh, so the workdir is snapshotted
before the first rung and restored before each escalation.

Cost is paid only where it can matter: checked tasks with a workdir and more
than one rung. Directories in SKIP (VCS data, dependencies, caches) are neither
copied nor restored. A workdir over max_bytes is not snapshotted; the report
says so and the rungs share the workdir as before.

Restoring removes files created since the snapshot (outside SKIP). Do not
point two concurrent tasks at one workdir with isolation on.
"""
import filecmp
import os
import shutil
import tempfile
import time
from pathlib import Path

SKIP={".git",".hg",".svn","node_modules",".venv","venv","__pycache__",".pytest_cache",".mypy_cache",".ruff_cache",".tox"}


def _remove(p):
    if os.path.isdir(p) and not os.path.islink(p):
        shutil.rmtree(p)
    else:
        os.remove(p)


def _files(root,dirs=None):
    """Relative paths of files and symlinks under root, outside SKIP dirs.
    Real directories are added to dirs when it is given."""
    out=set()
    for dirpath,dirnames,filenames in os.walk(root):
        keep=[]
        for d in dirnames:
            if d in SKIP:
                continue
            if os.path.islink(os.path.join(dirpath,d)):
                filenames.append(d)          # a symlinked dir is restored as a link
            else:
                keep.append(d)
        dirnames[:]=keep
        rel=os.path.relpath(dirpath,root)
        if dirs is not None:
            dirs.add(os.path.normpath(rel))
        for f in filenames:
            out.add(os.path.normpath(os.path.join(rel,f)))
    return out


def _size(root,files,limit):
    total=0
    for f in files:
        try:
            total+=os.lstat(os.path.join(root,f)).st_size
        except OSError:
            continue
        if total>limit:
            return total
    return total


def _copy(src,dst):
    os.makedirs(os.path.dirname(dst) or ".",exist_ok=True)
    if os.path.islink(src):
        if os.path.lexists(dst):
            _remove(dst)
        os.symlink(os.readlink(src),dst)
    else:
        if os.path.islink(dst):
            os.remove(dst)
        shutil.copy2(src,dst)


class Snapshot:
    def __init__(self,workdir,max_bytes):
        start=time.perf_counter()
        self.workdir=str(workdir)
        self.dir=None
        self.skipped=None
        self.dirs=set()
        files=_files(self.workdir,self.dirs)
        size=_size(self.workdir,files,max_bytes)
        if size>max_bytes:
            self.skipped=f"workdir over {max_bytes} bytes"
        else:
            self.dir=tempfile.mkdtemp(prefix="harness-rung-")
            try:
                for f in files:
                    _copy(os.path.join(self.workdir,f),os.path.join(self.dir,f))
                self.files=files
            except OSError as exc:            # unreadable file etc.: run unisolated, and say so
                self.close()
                self.skipped=f"snapshot failed: {exc!r}"[:300]
        self.seconds=time.perf_counter()-start

    def restore(self):
        """Put the workdir back as it was at snapshot time. Returns seconds taken."""
        start=time.perf_counter()
        if self.dir is None:
            return 0.
        now=_files(self.workdir)
        for f in now-self.files:             # created by the failed rung
            _remove(os.path.join(self.workdir,f))
        for f in self.files:
            src,dst=os.path.join(self.dir,f),os.path.join(self.workdir,f)
            same=(os.path.lexists(dst) and not os.path.islink(src) and not os.path.islink(dst)
                  and os.path.isfile(dst) and filecmp.cmp(src,dst,shallow=False))
            if not same:
                if os.path.lexists(dst) and os.path.isdir(dst) and not os.path.islink(dst):
                    shutil.rmtree(dst)
                _copy(src,dst)
        # directories the failed rung created and left empty
        # (walked top-down so SKIP dirs like node_modules are pruned, removed deepest first)
        now_dirs=set()
        _files(self.workdir,now_dirs)
        for rel in sorted(now_dirs-self.dirs-{"."},key=lambda r:-len(Path(r).parts)):
            path=os.path.join(self.workdir,rel)
            if not os.listdir(path):
                os.rmdir(path)
        return time.perf_counter()-start

    def close(self):
        if self.dir:
            shutil.rmtree(self.dir,ignore_errors=True)
            self.dir=None
