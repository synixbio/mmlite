"""SQLite schema for the mmlite index (mirrors MetaMapLite's ivf tables)."""

DB_FILENAME = "mmlite.sqlite"
TERMMAP_FILENAME = (
    "termmap.pkl"  # pickled key -> (cui, ...) cache built by IndexLookup.load_in_memory
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- cuiconcept: one preferred name per CUI
CREATE TABLE IF NOT EXISTS cuiconcept (
    cui            TEXT PRIMARY KEY,
    preferred_name TEXT NOT NULL
);

-- cuisourceinfo: every English string for a CUI, with its normalized form
CREATE TABLE IF NOT EXISTS cuisourceinfo (
    cui  TEXT NOT NULL,
    sui  TEXT NOT NULL,
    str  TEXT NOT NULL,
    norm TEXT NOT NULL,
    sab  TEXT NOT NULL,
    tty  TEXT NOT NULL
);

-- cuist: semantic types per CUI
CREATE TABLE IF NOT EXISTS cuist (
    cui    TEXT NOT NULL,
    tui    TEXT NOT NULL,
    abbrev TEXT NOT NULL
);

-- meshtcrelaxed: MeSH tree codes keyed by normalized MeSH preferred term, from MRSAT ATN=MN
CREATE TABLE IF NOT EXISTS meshtcrelaxed (
    term     TEXT NOT NULL,
    treecode TEXT NOT NULL
);
"""

INDEXES = """
CREATE INDEX IF NOT EXISTS ix_csi_norm  ON cuisourceinfo(norm);
CREATE INDEX IF NOT EXISTS ix_csi_cui   ON cuisourceinfo(cui);
CREATE INDEX IF NOT EXISTS ix_cuist_cui ON cuist(cui);
CREATE INDEX IF NOT EXISTS ix_mesh_term ON meshtcrelaxed(term);
"""
