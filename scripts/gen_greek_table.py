"""Generate src/mmlite/greek.py from MetaMapLite's GreekCharacters.java.

Usage: python scripts/gen_greek_table.py path/to/GreekCharacters.java
"""

import re
import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text(encoding="utf-8")
pairs = re.findall(r"put\('\\u([0-9A-Fa-f]{4})',\s*\"([^\"]*)\"\)", src)
if not pairs:
    sys.exit("no entries found")

out = [
    '"""Greek character -> ASCII expansion table, generated from MetaMapLite',
    "``gov.nih.nlm.nls.metamap.prefix.utf8.GreekCharacters`` by scripts/gen_greek_table.py.",
    '"""',
    "",
    "GREEK_TO_ASCII: dict[str, str] = {",
]
for cp, exp in pairs:
    out.append(f'    "\\u{cp.upper()}": {exp!r},  # {chr(int(cp, 16))}')
out += [
    "}",
    "",
    "",
    "def greek_to_ascii(text: str) -> str:",
    '    """Expand Greek letters to their names (alpha, beta, ...), as MetaMapLite does."""',
    "    if not any(ch in GREEK_TO_ASCII for ch in text):",
    "        return text",
    '    return "".join(GREEK_TO_ASCII.get(ch, ch) for ch in text)',
    "",
]
dest = Path(__file__).resolve().parents[1] / "src" / "mmlite" / "greek.py"
dest.write_text("\n".join(out), encoding="utf-8")
print(f"{len(pairs)} entries -> {dest}")
