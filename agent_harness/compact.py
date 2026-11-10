"""Deterministic context compaction: drop repetition and noise, never facts.

No model call, so compacting costs zero tokens. Every removal leaves a marker
with a count, so an agent can tell something was there. Code fences pass
through untouched. Prose is not summarized or reworded: rewording is where
compression methods lose information, and a dropped number or identifier is
a silent accuracy loss. What goes:
  - terminal escape codes and carriage-return progress-bar redraws
  - runs of identical lines (log spam) -> first line + count
  - repeated multi-line blocks (recursive stack frames) -> one copy + count
  - lines that repeat apart from a leading timestamp -> first and last + count
  - exact duplicate paragraphs (>= 80 characters) -> a back-reference
  - long base64/hex blobs -> length and short hash
  - trailing whitespace and runs of blank lines
"""
import hashlib
import re

ANSI=re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07")
FENCE=re.compile(r"^(```|~~~)")
TIMESTAMP=re.compile(r"^\s*\[?\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?\]?\s*"
                     r"|^\s*\[?\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\]?\s*"
                     r"|^\s*[A-Z][a-z]{2} +\d{1,2} \d{2}:\d{2}:\d{2}\s+")
BLOB=re.compile(r"(?<![A-Za-z0-9+/])(?:[A-Za-z0-9+/]{200,}={0,2}|[0-9a-fA-F]{200,})(?![A-Za-z0-9+/=])")


def _blob(m):
    s=m.group(0)
    return f"[blob: {len(s)} chars, sha1 {hashlib.sha1(s.encode()).hexdigest()[:10]}]"


def _segments(text):
    """Split into (is_code, lines) runs; fenced code is kept verbatim."""
    out,cur,code=[],[],False
    for line in text.split("\n"):
        if FENCE.match(line.strip()):
            if code:
                cur.append(line)
                out.append((True,cur))
                cur,code=[],False
            else:
                if cur:
                    out.append((False,cur))
                cur,code=[line],True
            continue
        cur.append(line)
    if cur:
        out.append((code,cur))
    return out


def _clean_line(line):
    line=ANSI.sub("",line)
    if "\r" in line:                          # progress bars redraw with \r: keep the final state
        line=line.rstrip("\r").split("\r")[-1]
    return BLOB.sub(_blob,line.rstrip())


def _collapse_runs(lines):
    out,i=[],0
    while i<len(lines):
        line=lines[i]
        j=i+1
        while j<len(lines) and lines[j]==line:
            j+=1
        if j-i>1 and line.strip():
            out.append(f"{line}  [line repeated {j-i}x]")
            i=j
            continue
        key=TIMESTAMP.sub("",line,count=1) if TIMESTAMP.match(line) else None
        if key:
            j=i+1
            while j<len(lines) and TIMESTAMP.match(lines[j]) and TIMESTAMP.sub("",lines[j],count=1)==key:
                j+=1
            if j-i>2:
                out+= [lines[i],f"  [same message {j-i-2}x more, timestamps vary]",lines[j-1]]
                i=j
                continue
        out.append(line)
        i+=1
    return out


def _collapse_blocks(lines,max_period=6):
    """Repeated multi-line blocks (e.g. recursive stack frames) -> one copy + count."""
    out,i=[],0
    while i<len(lines):
        for k in range(2,max_period+1):
            # A period-k repeat needs lines[i+k]==lines[i]; checking that first
            # skips building slices for the (common) lines that start no repeat.
            if i+k>=len(lines) or lines[i+k]!=lines[i]:
                continue
            block=lines[i:i+k]
            if len(block)<k or not any(l.strip() for l in block):
                continue
            reps=1
            while lines[i+reps*k:i+(reps+1)*k]==block:
                reps+=1
            if reps>1:
                out+=block+[f"  [previous {k} lines repeated {reps}x]"]
                i+=reps*k
                break
        else:
            out.append(lines[i])
            i+=1
    return out


def _squeeze_blank(lines):
    out=[]
    for line in lines:
        if not line.strip() and out and not out[-1].strip():
            continue
        out.append(line)
    return out


def compact(text):
    """Return (compacted_text, report). The report counts what was removed."""
    if not text:
        return text,{"chars_in":0,"chars_out":0}
    pieces=[]
    for is_code,lines in _segments(text.replace("\r\n","\n")):
        if is_code:
            pieces.append("\n".join(lines))
            continue
        lines=_squeeze_blank(_collapse_blocks(_collapse_runs([_clean_line(l) for l in lines])))
        pieces.append("\n".join(lines))
    body="\n".join(pieces)
    # exact duplicate paragraphs outside code
    seen,paras,dupes={},[],0
    in_code=False
    for para in body.split("\n\n"):
        fences=sum(1 for l in para.split("\n") if FENCE.match(l.strip()))
        key=para.strip()
        if not in_code and len(key)>=80 and not fences and key in seen:
            paras.append(f"[duplicate of paragraph {seen[key]+1} removed]")
            dupes+=1
        else:
            if not in_code and len(key)>=80 and not fences:
                seen.setdefault(key,len(paras))
            paras.append(para)
        if fences%2:
            in_code=not in_code
    out="\n\n".join(paras).strip("\n")
    return out,{"chars_in":len(text),"chars_out":len(out),"duplicate_paragraphs":dupes}


# Facts that must survive compaction; used by the retention tests and as a
# runtime guard (see compact_checked).
FACT=re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|https?://\S+|[A-Za-z_][A-Za-z0-9_]*(?:[./:][A-Za-z0-9_]+)+|[A-Z][A-Za-z]*(?:Error|Exception)\b")


def facts(text):
    # Leading timestamps are excluded on purpose: collapsing a repeated message
    # keeps its first and last timestamps and knowingly drops the ones between.
    lines=[TIMESTAMP.sub("",l,count=1) for l in ANSI.sub("",text).split("\n")]
    return set(FACT.findall("\n".join(lines)))


def _line_facts(lines):
    out=set()
    for l in lines:
        out.update(FACT.findall(TIMESTAMP.sub("",l,count=1)))
    return out


def lost_facts(text,out):
    """facts(BLOB.sub("",text)) - facts(out), without rescanning lines both share.

    No FACT match spans a newline, so facts are a per-line union, and a line
    present verbatim in both texts contributes the same facts to each side.
    Only input lines missing from the output can hold a lost fact. Escape codes
    can span lines (OSC sequences), so text containing ESC takes the full path."""
    if "\x1b" in text or "\x1b" in out:
        return facts(BLOB.sub("",text))-facts(out)
    out_lines=out.split("\n")
    out_set=set(out_lines)
    in_set=set(BLOB.sub("",text).split("\n"))
    candidates=_line_facts(in_set-out_set)
    if not candidates:
        return set()
    # Cheapest first: lines the compactor wrote (markers, collapsed runs) usually
    # carry the facts of the lines they replaced.
    candidates-=_line_facts(out_set-in_set)
    if candidates:
        candidates-=_line_facts(out_set&in_set)
    return candidates


def compact_checked(text):
    """Compact, but fall back to the original if any fact would disappear.
    A guard against a compaction bug ever costing information."""
    out,report=compact(text)
    if len(out)>=len(text):                    # markers can outweigh tiny savings: never grow the text
        return text,{**report,"chars_out":len(text),"fallback":"no saving"}
    lost=lost_facts(text,out)
    if lost:
        report={**report,"fallback":"fact lost","lost_examples":sorted(lost)[:5]}
        return text,report
    return out,report


if __name__=="__main__":
    # Pipe filter for agents: `noisy-command 2>&1 | python3 /path/to/compact.py`
    import sys
    out,report=compact_checked(sys.stdin.read())
    sys.stdout.write(out+"\n")
    if report.get("chars_out",0)<report.get("chars_in",0):
        sys.stdout.write(f"[compacted {report['chars_in']} -> {report['chars_out']} chars]\n")
