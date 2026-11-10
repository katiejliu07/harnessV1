"""Give every ladder rung the same starting files.

A rung that fails its check can leave half-finished edits behind. Without a
reset the next (stronger) rung starts from that broken state without being
told, and a later pass can owe credit to an earlier rung's edits. The ladder's
measured accuracy assumes each rung starts fresh, so the workdir is snapshotted
before the first rung and restored before each escalation.

Kept cheap because it runs before every checked task with a workdir:
  - files are cloned copy-on-write where the filesystem allows it (APFS
    clonefile, Linux FICLONE on btrfs/xfs), else copied;
  - each file's stat fingerprint (size, mtime, ctime, inode, mode) is kept,
    and restore only rewrites files whose fingerprint changed. Any content
    write or utime() bumps ctime, and ctime cannot be set back by user tools,
    so an unchanged fingerprint means unchanged content without reading it.

Directories in SKIP (VCS data, dependencies, caches) are neither copied nor
restored. A workdir over max_bytes is not snapshotted; the report says so and
the rungs share the workdir as before.

Restoring removes files created since the snapshot (outside SKIP). Do not
point two concurrent tasks at one workdir with isolation on.
"""
import ctypes
import errno
import filecmp
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

SKIP={".git",".hg",".svn","node_modules",".venv","venv","__pycache__",".pytest_cache",".mypy_cache",".ruff_cache",".tox"}


# ---- copy-on-write cloning ---------------------------------------------------

_clonefile=None
if sys.platform=="darwin":
    try:
        _libc=ctypes.CDLL(None,use_errno=True)
        _clonefile=_libc.clonefile
        _clonefile.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_uint32]
        _clonefile.restype=ctypes.c_int
    except (OSError,AttributeError):
        _clonefile=None
FICLONE=0x40049409
_clone_ok={}                                   # st_dev -> bool, learned on first try


def _clone_or_copy(src,dst,dev):
    """Clone src to dst (dst must not exist); fall back to a plain copy."""
    if _clone_ok.get(dev,True):
        if _clonefile is not None:
            if _clonefile(os.fsencode(src),os.fsencode(dst),0)==0:
                _clone_ok[dev]=True
                return
            err=ctypes.get_errno()
            if err not in (errno.ENOTSUP,errno.EXDEV,errno.EINVAL,errno.ENOSYS):
                raise OSError(err,os.strerror(err),src)
            _clone_ok[dev]=False
        elif sys.platform.startswith("linux"):
            import fcntl
            with open(src,"rb") as s,open(dst,"wb") as d:
                try:
                    fcntl.ioctl(d.fileno(),FICLONE,s.fileno())
                    _clone_ok[dev]=True
                    shutil.copymode(src,dst)
                    return
                except OSError:
                    _clone_ok[dev]=False
        else:
            _clone_ok[dev]=False
    shutil.copyfile(src,dst)
    shutil.copymode(src,dst)


# ---- scanning ----------------------------------------------------------------

# On POSIX any content write or utime() bumps st_ctime, and user tools cannot
# set it back, so an equal fingerprint proves the file is unchanged. On Windows
# st_ctime is the *creation* time (and DirEntry.stat() leaves st_ino 0), so a
# write that restores the old mtime would go unseen: there, an equal fingerprint
# is confirmed by comparing content before a file is skipped.
FINGERPRINT_IS_PROOF=os.name!="nt"


def _sig(st):
    if FINGERPRINT_IS_PROOF:
        return (st.st_size,st.st_mtime_ns,st.st_ctime_ns,st.st_ino,st.st_mode)
    return (st.st_size,st.st_mtime_ns,st.st_mode)


def _scan(root,skipped=None):
    """({rel: ("file", sig) | ("link", target)}, {rel dirs}) outside SKIP.
    SKIP dirs found below the top level are added to skipped when given."""
    entries,dirs={},{"."}
    stack=[(root,"")]
    while stack:
        path,rel=stack.pop()
        with os.scandir(path) as it:
            for e in it:
                r=f"{rel}{e.name}"
                if e.is_symlink():
                    entries[r]=("link",os.readlink(e.path))
                elif e.is_dir(follow_symlinks=False):
                    if e.name in SKIP:
                        if skipped is not None and rel:
                            skipped.append(r)
                        continue
                    dirs.add(r)
                    stack.append((e.path,r+"/"))
                elif e.is_file(follow_symlinks=False):
                    entries[r]=("file",_sig(e.stat(follow_symlinks=False)))
    return entries,dirs


def _snapshot_parent(workdir):
    """A temp location on the workdir's filesystem, so clones can work."""
    dev=os.stat(workdir).st_dev
    tmp=tempfile.gettempdir()
    try:
        if os.stat(tmp).st_dev==dev:
            return tmp
    except OSError:
        pass
    parent=os.path.dirname(os.path.abspath(workdir))
    if os.access(parent,os.W_OK):
        return parent
    return tmp


def _remove(p):
    if os.path.isdir(p) and not os.path.islink(p):
        shutil.rmtree(p)
    elif os.path.lexists(p):
        os.remove(p)


class Snapshot:
    def __init__(self,workdir,max_bytes):
        start=time.perf_counter()
        self.workdir=os.path.abspath(str(workdir))
        self.dir=None
        self.skipped=None
        nested_skip=[]
        # Fingerprints first, copy second: a file changed in between is simply
        # restored to its newer copy, never to stale content.
        self.entries,self.dirs=_scan(self.workdir,nested_skip)
        size=sum(v[1][0] for v in self.entries.values() if v[0]=="file")
        self.method=None
        if size>max_bytes:
            self.skipped=f"workdir over {max_bytes} bytes"
        else:
            self.dev=os.stat(self.workdir).st_dev
            self.dir=tempfile.mkdtemp(prefix=".harness-rung-",dir=_snapshot_parent(self.workdir))
            try:
                self.method=self._copy_tree(nested_skip)
            except OSError as exc:            # unreadable file etc.: run unisolated, and say so
                self.close()
                self.skipped=f"snapshot failed: {exc!r}"[:300]
        self.seconds=time.perf_counter()-start

    def _copy_tree(self,nested_skip):
        """Fastest exact copy available. Whole-directory clones copy nested SKIP
        dirs too; that is free with clones, so only the plain-copy paths avoid it."""
        tops=[e for e in os.listdir(self.workdir) if e not in SKIP]
        if _clonefile is not None and _clone_ok.get(self.dev,True):
            # APFS: one clonefile() per top-level entry clones its whole subtree
            # in the kernel (5k files: ~36 ms vs ~730 ms file by file).
            done=[]
            for name in tops:
                src,dst=os.path.join(self.workdir,name),os.path.join(self.dir,name)
                if os.path.islink(src):
                    continue                          # links are restored from their target string
                if _clonefile(os.fsencode(src),os.fsencode(dst),0)!=0:
                    err=ctypes.get_errno()
                    if err in (errno.ENOTSUP,errno.EXDEV,errno.EINVAL,errno.ENOSYS) and not done:
                        _clone_ok[self.dev]=False
                        break
                    raise OSError(err,os.strerror(err),src)
                done.append(name)
            else:
                return "clone"
        if sys.platform.startswith("linux") and not nested_skip and shutil.which("cp"):
            # GNU cp: reflink where supported, else a native-speed copy.
            srcs=[os.path.join(self.workdir,n) for n in tops if not os.path.islink(os.path.join(self.workdir,n))]
            if not srcs:
                return "cp"
            proc=subprocess.run(["cp","-a","--reflink=auto","--",*srcs,self.dir+"/"],capture_output=True,text=True)
            if proc.returncode==0:
                return "cp"
            shutil.rmtree(self.dir,ignore_errors=True)
            os.mkdir(self.dir)
        for d in sorted(self.dirs,key=len):
            if d!=".":
                os.makedirs(os.path.join(self.dir,d),exist_ok=True)
        for rel,(kind,_) in self.entries.items():
            if kind=="file":
                _clone_or_copy(os.path.join(self.workdir,rel),os.path.join(self.dir,rel),self.dev)
        return "files"

    def restore(self):
        """Put the workdir back as it was at snapshot time. Returns seconds taken.
        Only entries whose fingerprint changed are touched."""
        start=time.perf_counter()
        if self.dir is None:
            return 0.
        now,now_dirs=_scan(self.workdir)
        for rel in now.keys()-self.entries.keys():      # created by the failed rung
            _remove(os.path.join(self.workdir,rel))
        for d in sorted(self.dirs-now_dirs,key=len):    # original dirs it deleted
            path=os.path.join(self.workdir,d)
            if os.path.lexists(path) and not os.path.isdir(path):
                os.remove(path)                          # replaced by a file or link
            os.makedirs(path,exist_ok=True)
        for rel,want in self.entries.items():
            if now.get(rel)==want and (FINGERPRINT_IS_PROOF or want[0]!="file"
                                       or filecmp.cmp(os.path.join(self.dir,rel),os.path.join(self.workdir,rel),shallow=False)):
                continue                                 # untouched: no read, no write (POSIX)
            dst=os.path.join(self.workdir,rel)
            if os.path.lexists(dst):
                _remove(dst)
            if want[0]=="link":
                os.symlink(want[1],dst)
                continue
            src=os.path.join(self.dir,rel)
            _clone_or_copy(src,dst,self.dev)
            shutil.copystat(src,dst)
            # the restored file has a new ctime/inode: remember that as "original"
            self.entries[rel]=("file",_sig(os.lstat(dst)))
        # directories the failed rung created and left empty, deepest first
        for d in sorted(now_dirs-self.dirs,key=lambda r:-r.count("/")):
            path=os.path.join(self.workdir,d)
            if os.path.isdir(path) and not os.listdir(path):
                os.rmdir(path)
        return time.perf_counter()-start

    def close(self,wait=False):
        """Delete the snapshot. Off the critical path by default: deleting 5k
        files takes ~300 ms that the task's caller should not wait for."""
        if self.dir:
            d,self.dir=self.dir,None
            if wait:
                shutil.rmtree(d,ignore_errors=True)
            else:
                t=threading.Thread(target=shutil.rmtree,args=(d,),kwargs={"ignore_errors":True},daemon=False)
                t.start()
                _cleanups.append(t)


_cleanups=[]


def wait_for_cleanup():
    """Join background snapshot deletions (tests, or before measuring disk)."""
    while _cleanups:
        _cleanups.pop().join()
