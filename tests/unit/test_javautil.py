"""Tests for the Java collection semantics that leak into MetaMapLite's output.

These pin behaviour that is only observable by diffing against the real Java tool, so they are
written as statements about Java rather than about convenience.
"""

from mmlite.javautil import (
    ev_hash,
    hash_set_order,
    hash_set_order_by,
    string_hash,
    table_capacity,
)


def test_string_hash_matches_java_string_hashcode():
    # Values taken from Java: "".hashCode() == 0, "a" == 97, "MTH" == 76673, "LNC" == 75521.
    assert string_hash("") == 0
    assert string_hash("a") == 97
    assert string_hash("MTH") == 76673
    assert string_hash("LNC") == 75521


def test_table_capacity_doubles_past_the_load_factor():
    assert table_capacity(0) == 16
    assert table_capacity(12) == 16  # 16 * 0.75
    assert table_capacity(13) == 32
    assert table_capacity(25) == 64


def test_colliding_sources_keep_insertion_order_within_their_bucket():
    """``MTH`` and ``LNC`` both land in bucket 0, so whichever went in first prints first.

    This is why ``IndexLookup.sources`` must preserve MRCONSO order rather than sort: sorting
    would always put ``LNC`` first, and Java's output has ``MTH`` first whenever MRCONSO does.
    """
    assert hash_set_order(["MTH", "LNC", "HL7V2.5"]) == ["MTH", "LNC", "HL7V2.5"]
    assert hash_set_order(["LNC", "MTH", "HL7V2.5"]) == ["LNC", "MTH", "HL7V2.5"]


def test_hash_set_order_dedupes_keeping_the_first_occurrence():
    assert hash_set_order(["MTH", "LNC", "MTH"]) == ["MTH", "LNC"]


def test_hash_set_order_is_bucket_order_not_insertion_order_across_buckets():
    """Elements in different buckets are reordered into table order regardless of insertion."""
    assert hash_set_order(["HL7V2.5", "MTH"]) == ["MTH", "HL7V2.5"]


def test_ev_hash_is_length_plus_start_plus_cui_hash():
    assert ev_hash(5, 3, "C0011849") == (3 + 5 + string_hash("C0011849")) & 0xFFFFFFFF


def test_ev_hash_wraps_at_32_bits_as_java_int_addition_does():
    # A CUI whose hash is large enough that adding to it overflows a signed 32-bit int.
    big = "zzzzzzzzzzzz"
    assert 0 <= ev_hash(2**31, 2**31, big) <= 0xFFFFFFFF


def test_hash_set_order_by_orders_arbitrary_elements_by_their_java_hash():
    items = [("a", 1), ("b", 2), ("c", 3)]
    ordered = hash_set_order_by(items, lambda t: t[1])
    assert sorted(ordered) == sorted(items)
    assert ordered == sorted(items, key=lambda t: t[1] % 16)
