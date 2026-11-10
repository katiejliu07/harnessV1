"""Give every ladder rung the same starting files.

A rung that fails its check can leave half-finished edits behind. Without a
reset the next (stronger) rung starts from that broken state without being
told, and a later pass can owe credit to an earlier rung's edits. The ladder's
measured accuracy assumes each rung starts fresh, so the workdir is snapshotted
before the first rung and restored before each escalation.

Kept off the critical path because it runs before every checked task with a
workdir:
  - only the fingerprint scan runs before the first rung. The copy is built in
    the background while the agent starts up (seconds before its first edit),
    cloned copy-on-write where the filesystem allows it (APFS clonefile, Linux
    reflink) and otherwise copied by parallel threads. Every file is checked
    against its fingerprint just before and just after it is copied; if the
    agent got there first, the snapshot is marked unusable and the ladder runs
    unisolated (reported), so a restore never brings back wrong content;
  - restore only rewrites files whose fingerprint changed. The fingerprint
    includes the change time (POSIX st_ctime; NTFS ChangeTime on Windows,
    read per directory), which any content write or utime() bumps, so an
    unchanged fingerprint means unchanged content without reading it.

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
import struct
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

# Whether an equal fingerprint proves a file unchanged. POSIX: st_ctime is bumped
# by any content write or utime() and user tools cannot set it back. Windows:
# os.stat's st_ctime is the *creation* time, so the NTFS ChangeTime is read per
# directory instead (_win_change_times); if that fails, equal fingerprints are
# confirmed by comparing content. Tests set this False to force that fallback.
FINGERPRINT_IS_PROOF=True

_win=None
if os.name=="nt":
    try:
        from ctypes import wintypes as _wt
        _k32=ctypes.WinDLL("kernel32",use_last_error=True)
        _CreateFileW=_k32.CreateFileW
        _CreateFileW.argtypes=[_wt.LPCWSTR,_wt.DWORD,_wt.DWORD,ctypes.c_void_p,_wt.DWORD,_wt.DWORD,_wt.HANDLE]
        _CreateFileW.restype=_wt.HANDLE
        _GetInfo=_k32.GetFileInformationByHandleEx
        _GetInfo.argtypes=[_wt.HANDLE,ctypes.c_int,ctypes.c_void_p,_wt.DWORD]
        _GetInfo.restype=_wt.BOOL
        _CloseHandle=_k32.CloseHandle
        _CloseHandle.argtypes=[_wt.HANDLE]
        _win=True
    except (OSError,AttributeError):
        _win=None
_INVALID_HANDLE=ctypes.c_void_p(-1).value


def _win_change_times(path):
    """{name: NTFS ChangeTime} for one directory, in one enumeration
    (FILE_FULL_DIR_INFO: ChangeTime at offset 32, name length at 60, name at 68)."""
    h=_CreateFileW(path,0x0001,0x7,None,3,0x02000000,None)   # LIST_DIRECTORY, share all, OPEN_EXISTING, BACKUP_SEMANTICS
    if h in (None,_INVALID_HANDLE):
        raise OSError(ctypes.get_last_error(),"CreateFileW failed",path)
    out={}
    buf=ctypes.create_string_buffer(64*1024)
    cls=0xF                                                    # FileFullDirectoryRestartInfo, then FileFullDirectoryInfo
    try:
        while True:
            if not _GetInfo(h,cls,buf,len(buf)):
                err=ctypes.get_last_error()
                if err==18:                                    # ERROR_NO_MORE_FILES
                    break
                raise OSError(err,"GetFileInformationByHandleEx failed",path)
            cls=0xE
            raw=buf.raw
            off=0
            while True:
                nxt,=struct.unpack_from("<I",raw,off)
                change,=struct.unpack_from("<q",raw,off+32)
                nlen,=struct.unpack_from("<I",raw,off+60)
                out[raw[off+68:off+68+nlen].decode("utf-16-le")]=change
                if not nxt:
                    break
                off+=nxt
    finally:
        _CloseHandle(h)
    return out


def _sig(st,change=None):
    if os.name!="nt":
        return (st.st_size,st.st_mtime_ns,st.st_ctime_ns,st.st_ino,st.st_mode)
    return (st.st_size,st.st_mtime_ns,change,st.st_mode)


def _quick(path):
    """The fields of a fingerprint that one stat() can read, for checking a
    file just before and after it is copied."""
    st=os.lstat(path)
    return (st.st_size,st.st_mtime_ns,st.st_ctime_ns) if os.name!="nt" else (st.st_size,st.st_mtime_ns)


def _quick_of(sig):
    return sig[:3] if os.name!="nt" else sig[:2]


def _dir_change_times(path):
    if os.name!="nt" or not _win:
        return None
    try:
        return _win_change_times(path)
    except OSError:
        return None


def _scan(root,skipped=None):
    """({rel: ("file", sig) | ("link", target)}, {rel dirs}, proof) outside SKIP.
    proof says whether equal fingerprints prove equal content. SKIP dirs found
    below the top level are added to skipped when given."""
    entries,dirs={},{"."}
    proof=FINGERPRINT_IS_PROOF
    stack=[(root,"")]
    while stack:
        path,rel=stack.pop()
        changes=_dir_change_times(path)
        if os.name=="nt" and changes is None:
            proof=False
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
                    change=changes.get(e.name) if changes else None
                    if os.name=="nt" and change is None:
                        proof=False
                    entries[r]=("file",_sig(e.stat(follow_symlinks=False),change))
    return entries,dirs,proof


def _sig_of(path):
    st=os.lstat(path)
    if os.name!="nt":
        return _sig(st)
    changes=_dir_change_times(os.path.dirname(path)) or {}
    return _sig(st,changes.get(os.path.basename(path)))


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
    """Snapshot(workdir, max_bytes, background=True).

    seconds: time spent on the caller's path (the scan, plus the copy when
    background=False). build_seconds: time the copy took. restore() waits for
    the copy and returns None when the snapshot is unusable (see skipped)."""

    def __init__(self,workdir,max_bytes,background=True,jobs=8):
        start=time.perf_counter()
        self.workdir=os.path.abspath(str(workdir))
        self.dir=None
        self.skipped=None
        self.method=None
        self.build_seconds=None
        self.jobs=jobs
        self._cancel=threading.Event()
        self._thread=None
        nested_skip=[]
        # Fingerprints first, before the agent can touch anything: they are the
        # originals every copied file is checked against.
        self.entries,self.dirs,self.proof=_scan(self.workdir,nested_skip)
        size=sum(v[1][0] for v in self.entries.values() if v[0]=="file")
        if size>max_bytes:
            self.skipped=f"workdir over {max_bytes} bytes"
        else:
            self.dev=os.stat(self.workdir).st_dev
            self.root=tempfile.mkdtemp(prefix=".harness-rung-",dir=_snapshot_parent(self.workdir))
            self.dir=self.root                        # None once closed; the copier keeps using root
            if background:
                self._thread=threading.Thread(target=self._build,args=(nested_skip,),daemon=True)
                self._thread.start()
            else:
                self._build(nested_skip)
        self.seconds=time.perf_counter()-start

    def _build(self,nested_skip):
        start=time.perf_counter()
        try:
            self.method=self._copy_tree(nested_skip)
        except Exception as exc:          # noqa: BLE001 - unreadable file, the agent got there first, closed
            self.skipped=f"snapshot failed: {exc}"[:300]
        self.build_seconds=time.perf_counter()-start

    def ready(self):
        """Wait for the copy; True if it can be restored from."""
        if self._thread is not None:
            self._thread.join()
        return self.dir is not None and self.skipped is None

    def _verify_tree(self):
        """After a whole-tree copy: every original still has its fingerprint, so
        the copy was taken from the originals."""
        now,_,_=_scan(self.workdir)
        for rel,want in self.entries.items():
            if now.get(rel)!=want:
                raise OSError(f"{rel} changed while the snapshot was taken")

    def _copy_tree(self,nested_skip):
        """Fastest exact copy available. Whole-directory clones copy nested SKIP
        dirs too; that is free with clones, so only the plain-copy paths avoid it."""
        tops=[e for e in os.listdir(self.workdir) if e not in SKIP]
        if _clonefile is not None and _clone_ok.get(self.dev,True):
            # APFS: one clonefile() per top-level entry clones its whole subtree
            # in the kernel (5k files: ~36 ms vs ~730 ms file by file).
            done=[]
            for name in tops:
                if self._cancel.is_set():
                    raise OSError("cancelled")
                src,dst=os.path.join(self.workdir,name),os.path.join(self.root,name)
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
                self._verify_tree()
                return "clone"
        if sys.platform.startswith("linux") and not nested_skip and shutil.which("cp"):
            # GNU cp: reflink where supported, else a native-speed copy.
            srcs=[os.path.join(self.workdir,n) for n in tops if not os.path.islink(os.path.join(self.workdir,n))]
            if srcs:
                proc=subprocess.run(["cp","-a","--reflink=auto","--",*srcs,self.root+"/"],capture_output=True,text=True)
                if proc.returncode==0:
                    self._verify_tree()
                    return "cp"
                shutil.rmtree(self.root,ignore_errors=True)
                os.mkdir(self.root)
            else:
                return "cp"
        for d in sorted(self.dirs,key=len):
            if d!=".":
                os.makedirs(os.path.join(self.root,d),exist_ok=True)
        files=[rel for rel,(kind,_) in self.entries.items() if kind=="file"]

        def one(rel):
            if self._cancel.is_set():
                raise OSError("cancelled")
            src=os.path.join(self.workdir,rel)
            want=_quick_of(self.entries[rel][1])
            if _quick(src)!=want:
                raise OSError(f"{rel} changed before it was copied")
            _clone_or_copy(src,os.path.join(self.root,rel),self.dev)
            if _quick(src)!=want:
                raise OSError(f"{rel} changed while it was copied")

        if self.jobs>1 and len(files)>32:
            # Per-file copies are bound by open/close and antivirus scanning
            # (Windows: ~1 ms per file), not bandwidth: threads overlap them.
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.jobs) as pool:
                list(pool.map(one,files))
        else:
            for rel in files:
                one(rel)
        return "files"

    def restore(self):
        """Put the workdir back as it was at snapshot time. Returns seconds taken,
        or None if the snapshot is unusable. Only entries whose fingerprint
        changed are touched."""
        if not self.ready():
            return None
        start=time.perf_counter()
        now,now_dirs,now_proof=_scan(self.workdir)
        proof=self.proof and now_proof
        for rel in now.keys()-self.entries.keys():      # created by the failed rung
            _remove(os.path.join(self.workdir,rel))
        for d in sorted(self.dirs-now_dirs,key=len):    # original dirs it deleted
            path=os.path.join(self.workdir,d)
            if os.path.lexists(path) and not os.path.isdir(path):
                os.remove(path)                          # replaced by a file or link
            os.makedirs(path,exist_ok=True)
        for rel,want in self.entries.items():
            if now.get(rel)==want and (proof or want[0]!="file"
                                       or filecmp.cmp(os.path.join(self.root,rel),os.path.join(self.workdir,rel),shallow=False)):
                continue                                 # untouched: no read, no write when proof holds
            dst=os.path.join(self.workdir,rel)
            if os.path.lexists(dst):
                _remove(dst)
            if want[0]=="link":
                os.symlink(want[1],dst)
                continue
            src=os.path.join(self.root,rel)
            _clone_or_copy(src,dst,self.dev)
            shutil.copystat(src,dst)
            # the restored file has a new change time/inode: that is now "original"
            self.entries[rel]=("file",_sig_of(dst))
        # directories the failed rung created and left empty, deepest first
        for d in sorted(now_dirs-self.dirs,key=lambda r:-r.count("/")):
            path=os.path.join(self.workdir,d)
            if os.path.isdir(path) and not os.listdir(path):
                os.rmdir(path)
        return time.perf_counter()-start

    def close(self,wait=False):
        """Stop any copy in progress and delete the snapshot. Off the critical
        path by default: deleting 5k files takes ~300 ms that the task's caller
        should not wait for."""
        self._cancel.set()
        if self.dir:
            d,self.dir=self.dir,None
            thread=self._thread

            def cleanup():
                if thread is not None:
                    thread.join()
                shutil.rmtree(d,ignore_errors=True)

            if wait:
                cleanup()
            else:
                t=threading.Thread(target=cleanup,daemon=False)
                t.start()
                _cleanups.append(t)


_cleanups=[]


def wait_for_cleanup():
    """Join background snapshot deletions (tests, or before measuring disk)."""
    while _cleanups:
        _cleanups.pop().join()
