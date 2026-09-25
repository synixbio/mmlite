"""Generate src/mmlite/pipeline/negex_triggers.py from MetaMapLite's NegExKeyMap.java.

Usage: python scripts/gen_negex_triggers.py path/to/NegExKeyMap.java
"""

import re
import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text(encoding="utf-8")
entries = re.findall(r'put\(Arrays\.asList\(([^)]*)\),\s*"(\w+)"\)', src)
if not entries:
    sys.exit("no entries found")

seen: dict[tuple[str, ...], str] = {}
for words, kind in entries:
    phrase = tuple(re.findall(r'"([^"]*)"', words))
    seen[phrase] = kind  # later puts override earlier ones, as in a Java HashMap

out = [
    '"""NegEx trigger phrases, generated from MetaMapLite ``NegExKeyMap`` by',
    "scripts/gen_negex_triggers.py.  Types: nega/negb (pre/post negation), pnega/pnegb",
    '(pseudo-negation), conj (scope terminator)."""',
    "",
    "NEGATION_PHRASE_TYPES: dict[tuple[str, ...], str] = {",
]
for phrase, kind in seen.items():
    out.append(f"    {phrase!r}: {kind!r},")
out += ["}", ""]
dest = Path(__file__).resolve().parents[1] / "src" / "mmlite" / "pipeline" / "negex_triggers.py"
dest.write_text("\n".join(out), encoding="utf-8")
print(f"{len(seen)} phrases ({len(entries)} puts) -> {dest}")
