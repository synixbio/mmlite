"""Generate src/mmlite/pipeline/context_triggers.py from MetaMapLite's context-2012.jar.

Usage: python scripts/gen_context_triggers.py path/to/context-2012.jar

The ConText lexicon is not a resource file: the 2012 Java implementation compiles its trigger
table into ``context/implementation/ConText.class`` as string constants of the form
``phrase,position,type`` (e.g. ``absence of ,pre,neg``).  The constructor loads them into a
``String[]`` with one ``ldc`` each, and this script reads that sequence from ``javap -c`` — the
array order, which is regex alternation order and so decides which trigger wins when two overlap.
One entry, ``"history, physical",pseudo,hist``, carries a comma inside its phrase; Java splits on
the first and last comma, gets a nonsense position and drops it, and the port must do the same,
so it is kept here verbatim.  Needs a JDK on PATH for ``javap``.
"""

from __future__ import annotations

import re
import subprocess
import sys
import zipfile
from pathlib import Path

jar = Path(sys.argv[1])
CLASS = "context/implementation/ConText.class"
work = Path(__file__).resolve().parent / "_context_tmp"
work.mkdir(exist_ok=True)
with zipfile.ZipFile(jar) as zf:
    zf.extract(CLASS, work)
dump = subprocess.run(
    ["javap", "-c", "-p", str(work / CLASS)],
    capture_output=True,
    text=True,
    check=True,
    encoding="utf-8",
).stdout
ctor = dump.split("public context.implementation.ConText();", 1)[1].split("putfield", 1)[0]

entries: list[str] = []
for m in re.finditer(r"ldc(?:_w)?\s+#\d+\s+// String (.*)$", ctor, re.M):
    value = m.group(1).replace('\\"', '"')  # javap escapes embedded quotes
    if re.search(r",[a-z]+,[a-z]+$", value):
        entries.append(value)
if len(entries) < 300:
    sys.exit(f"only {len(entries)} lexicon entries found; expected ~355")

out = [
    '"""ConText trigger lexicon, generated from ``context-2012.jar`` by',
    "scripts/gen_context_triggers.py.  Each entry is the raw ``phrase,position,type`` string",
    "exactly as the Java class stores it -- trailing spaces included, since they change what the",
    "generated regex requires after the phrase, and one entry with a comma inside its phrase that",
    "Java's first/last-comma split turns into nothing (see pipeline/context.py).  Order is the",
    'Java array order, which is regex alternation order."""',
    "",
    "CONTEXT_TRIGGERS: tuple[str, ...] = (",
]
for entry in entries:
    out.append(f"    {entry!r},")
out += [")", ""]
dest = Path(__file__).resolve().parents[1] / "src" / "mmlite" / "pipeline" / "context_triggers.py"
dest.write_text("\n".join(out), encoding="utf-8")
print(f"{len(entries)} entries -> {dest}")
