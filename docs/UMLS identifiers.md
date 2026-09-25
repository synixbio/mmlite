# UMLS Unique Identifiers (UIs) Interlocking Examples

In the [Unified Medical Language System](https://www.nlm.nih.gov/research/umls/index.html) (UMLS), concept nesting follows a strict top-down structural hierarchy: **CUI → LUI → SUI → AUI**, which then maps out to the original source vocabulary structures (**SCUI**, **CODE**) and the broader **TUI** category.

## The Interlocking Hierarchy: "Atrial Fibrillation"
To see how these variables interconnect, let's look at the concrete example of the cardiac condition **Atrial Fibrillation**.

```
[ TUI: T047 (Disease or Syndrome) ]
       │
       ▼
[ CUI: C0004154 (Concept: Atrial Fibrillation) ]
       ├── [ LUI: L0004154 (Lexical: "Atrial Fibrillation") ]
       │     ├── [ SUI: S0014791 (String: "Atrial Fibrillation") ]
       │     │     ├── [ AUI: A2822452 ] ──► [ CODE: 164889003 ] (SNOMED CT)
       │     │     └── [ AUI: A0027961 ] ──► [ CODE: I48.91 ]    (ICD-10-CM)
       │     │
       │     └── [ SUI: S0014792 (String: "atrial fibrillation") ]
       │           └── [ AUI: A3154812 ] ──► [ SCUI: M0002166 ] (MeSH Descriptor)
       │
       └── [ LUI: L0337821 (Lexical: "Auricular Fibrillation" - Historical Synonym) ]
             └── [ SUI: S0421890 (String: "Auricular fibrillation") ]
                   └── [ AUI: A0561924 ] ──► [ CODE: 427.31 ]    (ICD-9-CM)
```

## Breaking Down the Connections

* **TUI (Type Unique Identifier):** The concept is categorized broadly under **`T047`** (Disease or Syndrome) in the Semantic Network.
* **CUI (Concept Unique Identifier):** Everything below resolves to **`C0004154`**. No matter what term a clinician uses, the computer knows they mean the exact same clinical concept.
* **LUI (Lexical Unique Identifier):** Split by base word structure. 
  * **`L0004154`** represents the modern wording "Atrial Fibrillation". 
  * **`L0337821`** is generated for "Auricular Fibrillation" because it uses a completely different base word ("Auricular" vs "Atrial"), even though it means the same thing.
* **SUI (String Unique Identifier):** Inside the modern LUI, a unique SUI isolates exact textual variations. Capitalized "**A**trial **F**ibrillation" (**`S0014791`**) gets a separate ID from lowercase "**a**trial **f**ibrillation" (**`S0014792`**).
* **AUI (Atom Unique Identifier):** The precise point where a specific string is imported from a specific source. If SNOMED CT and ICD-10-CM both use the exact string "Atrial Fibrillation", they each receive their own unique AUI (**`A2822452`** vs **`A0027961`**) to preserve data lineage.
* **CODE / SCUI (Source Assertions):** These point directly back to the original source systems. The UMLS atoms link directly to the native SNOMED code (`164889003`) or the MeSH identifier (`M0002166`).

## Relationship Interlocking (`RUI` and `ATUI`)
When this concept connects to another one, **RUI** and **ATUI** manage the layout. 

If **`C0004154`** (Atrial Fibrillation) is treated by **`C0010255`** (Digoxin), the database records a relationship row and stamps it with a **RUI** (Relationship Unique Identifier). If a source specifies that this treatment relationship is "highly common," that extra note is stamped with an **ATUI** (Attribute Unique Identifier) and attached to the RUI.