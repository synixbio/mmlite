from mmlite.semtypes import to_tui, to_tui_set, unknown_semantic_types


def test_to_tui_accepts_abbreviations_and_tuis_in_any_case():
    assert to_tui("dsyn") == to_tui("DSYN") == to_tui("t047") == "T047"
    assert to_tui("nope") == "nope"  # unknown values pass through unchanged


def test_to_tui_set_treats_all_and_empty_as_no_restriction():
    assert to_tui_set(None) is None
    assert to_tui_set([]) is None
    assert to_tui_set(["dsyn", "all"]) is None
    assert to_tui_set(["dsyn", " sosy "]) == {"T047", "T184"}


def test_unknown_semantic_types_names_only_the_bad_values():
    assert unknown_semantic_types(None) == []
    assert unknown_semantic_types(["dsyn", "T047", "ALL", " sosy "]) == []
    assert unknown_semantic_types(["dysn", "dsyn", "zzz", "T999"]) == ["T999", "dysn", "zzz"]
