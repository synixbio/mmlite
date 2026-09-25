"""UMLS semantic type TUI <-> MetaMap abbreviation mapping (SemanticTypes_2018AB).

:func:`to_tui_set` and :func:`to_source_set` are the shared readings of a restriction set —
used by both the dictionary lookup and the entity-lookup filter so ``dsyn``, ``DSYN``, ``T047``
and ``t047`` all select the same concepts wherever a restriction is accepted.
"""

from __future__ import annotations

from collections.abc import Iterable

_RAW = """
T001 orgm Organism
T002 plnt Plant
T004 fngs Fungus
T005 virs Virus
T007 bact Bacterium
T008 anim Animal
T010 vtbt Vertebrate
T011 amph Amphibian
T012 bird Bird
T013 fish Fish
T014 rept Reptile
T015 mamm Mammal
T016 humn Human
T017 anst Anatomical Structure
T018 emst Embryonic Structure
T019 cgab Congenital Abnormality
T020 acab Acquired Abnormality
T021 ffas Fully Formed Anatomical Structure
T022 bdsy Body System
T023 bpoc Body Part, Organ, or Organ Component
T024 tisu Tissue
T025 cell Cell
T026 celc Cell Component
T028 gngm Gene or Genome
T029 blor Body Location or Region
T030 bsoj Body Space or Junction
T031 bdsu Body Substance
T032 orga Organism Attribute
T033 fndg Finding
T034 lbtr Laboratory or Test Result
T037 inpo Injury or Poisoning
T038 biof Biologic Function
T039 phsf Physiologic Function
T040 orgf Organism Function
T041 menp Mental Process
T042 ortf Organ or Tissue Function
T043 celf Cell Function
T044 moft Molecular Function
T045 genf Genetic Function
T046 patf Pathologic Function
T047 dsyn Disease or Syndrome
T048 mobd Mental or Behavioral Dysfunction
T049 comd Cell or Molecular Dysfunction
T050 emod Experimental Model of Disease
T051 evnt Event
T052 acty Activity
T053 bhvr Behavior
T054 socb Social Behavior
T055 inbe Individual Behavior
T056 dora Daily or Recreational Activity
T057 ocac Occupational Activity
T058 hlca Health Care Activity
T059 lbpr Laboratory Procedure
T060 diap Diagnostic Procedure
T061 topp Therapeutic or Preventive Procedure
T062 resa Research Activity
T063 mbrt Molecular Biology Research Technique
T064 gora Governmental or Regulatory Activity
T065 edac Educational Activity
T066 mcha Machine Activity
T067 phpr Phenomenon or Process
T068 hcpp Human-caused Phenomenon or Process
T069 eehu Environmental Effect of Humans
T070 npop Natural Phenomenon or Process
T071 enty Entity
T072 phob Physical Object
T073 mnob Manufactured Object
T074 medd Medical Device
T075 resd Research Device
T077 cnce Conceptual Entity
T078 idcn Idea or Concept
T079 tmco Temporal Concept
T080 qlco Qualitative Concept
T081 qnco Quantitative Concept
T082 spco Spatial Concept
T083 geoa Geographic Area
T085 mosq Molecular Sequence
T086 nusq Nucleotide Sequence
T087 amas Amino Acid Sequence
T088 crbs Carbohydrate Sequence
T089 rnlw Regulation or Law
T090 ocdi Occupation or Discipline
T091 bmod Biomedical Occupation or Discipline
T092 orgt Organization
T093 hcro Health Care Related Organization
T094 pros Professional Society
T095 shro Self-help or Relief Organization
T096 grup Group
T097 prog Professional or Occupational Group
T098 popg Population Group
T099 famg Family Group
T100 aggp Age Group
T101 podg Patient or Disabled Group
T102 grpa Group Attribute
T103 chem Chemical
T104 chvs Chemical Viewed Structurally
T109 orch Organic Chemical
T114 nnon Nucleic Acid, Nucleoside, or Nucleotide
T116 aapp Amino Acid, Peptide, or Protein
T120 chvf Chemical Viewed Functionally
T121 phsu Pharmacologic Substance
T122 bodm Biomedical or Dental Material
T123 bacs Biologically Active Substance
T125 horm Hormone
T126 enzy Enzyme
T127 vita Vitamin
T129 imft Immunologic Factor
T130 irda Indicator, Reagent, or Diagnostic Aid
T131 hops Hazardous or Poisonous Substance
T167 sbst Substance
T168 food Food
T169 ftcn Functional Concept
T170 inpr Intellectual Product
T171 lang Language
T184 sosy Sign or Symptom
T185 clas Classification
T190 anab Anatomical Abnormality
T191 neop Neoplastic Process
T192 rcpt Receptor
T194 arch Archaeon
T195 antb Antibiotic
T196 elii Element, Ion, or Isotope
T197 inch Inorganic Chemical
T200 clnd Clinical Drug
T201 clna Clinical Attribute
T203 drdd Drug Delivery Device
T204 euka Eukaryote
"""

TUI_TO_ABBREV: dict[str, str] = {}
TUI_TO_NAME: dict[str, str] = {}
ABBREV_TO_TUI: dict[str, str] = {}

for _line in _RAW.strip().splitlines():
    _tui, _abbr, _name = _line.split(" ", 2)
    TUI_TO_ABBREV[_tui] = _abbr
    TUI_TO_NAME[_tui] = _name
    ABBREV_TO_TUI[_abbr] = _tui

# Case-insensitive views, so a user may write dsyn, DSYN, T047 or t047 interchangeably.
_ABBREV_TO_TUI_CI = {k.lower(): v for k, v in ABBREV_TO_TUI.items()}
_TUI_TO_ABBREV_CI = {k.lower(): v for k, v in TUI_TO_ABBREV.items()}


def to_abbrev(tui: str) -> str:
    """TUI -> MetaMap abbreviation; unknown values are returned unchanged."""
    return _TUI_TO_ABBREV_CI.get(tui.lower(), tui)


def to_tui(abbrev_or_tui: str) -> str:
    """Abbreviation or TUI -> TUI (canonical upper case); unknown values are unchanged."""
    s = abbrev_or_tui.lower()
    if s in _ABBREV_TO_TUI_CI:
        return _ABBREV_TO_TUI_CI[s]
    if s in _TUI_TO_ABBREV_CI:
        return abbrev_or_tui.upper()
    return abbrev_or_tui


def unknown_semantic_types(semantic_types: Iterable[str] | None) -> list[str]:
    """The values in a restriction set that are neither a known abbreviation nor a TUI.

    Filtering itself is permissive - an unknown value simply matches nothing, as in Java - which
    is exactly how a typo like ``dysn`` turns into an empty, exit-0 result.  Callers that take
    the restriction from a user (the CLI, the example scripts) check this first and refuse.
    ``all`` is accepted, as :func:`to_tui_set` treats it as "no restriction".
    """
    if not semantic_types:
        return []
    return sorted(
        v.strip()
        for v in semantic_types
        if v
        and v.strip()
        and v.strip().lower() not in _ABBREV_TO_TUI_CI
        and v.strip().lower() not in _TUI_TO_ABBREV_CI
        and v.strip().lower() != "all"
    )


def to_tui_set(semantic_types: Iterable[str] | None) -> set[str] | None:
    """A restriction set as TUIs, accepting abbreviations or TUIs in any case.

    Returns ``None`` when the restriction is empty or contains ``"all"`` (no filtering) —
    the single place both :mod:`mmlite.index.lookup` and
    :mod:`mmlite.pipeline.entity_lookup` interpret a semantic-type restriction.
    """
    if not semantic_types:
        return None
    values = {s.strip() for s in semantic_types if s and s.strip()}
    if not values or any(v.lower() == "all" for v in values):
        return None
    return {to_tui(v) for v in values}


def to_source_set(sources: Iterable[str] | None) -> set[str] | None:
    """A source restriction set, upper-cased (UMLS SABs are upper case); ``None`` = no filter."""
    if not sources:
        return None
    values = {s.strip().upper() for s in sources if s and s.strip()}
    if not values or "ALL" in values:
        return None
    return values
