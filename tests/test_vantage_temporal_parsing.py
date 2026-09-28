"""Unit tests for the timestamp parsing helpers in vlmeval/dataset/vantage_temporal.py.

VANTAGE_Temporal.parse_timestamps_json and timestamp_to_seconds are
@staticmethod, so no dataset instance is constructed here.

Two input forms are covered:
  * "hh:mm:ss" strings convert to seconds, so "00:01:05" is 65.0.
  * numeric JSON values ({"start": 4.5, "end": 9.0}) are read as seconds
    directly.
The 'ss' and 'mm:ss' forms, list-wrapped answers and truncated output
keep their established results, and unparseable input keeps the existing
fallback policy.
"""
import pytest

from vlmeval.dataset.vantage_temporal import VANTAGE_Temporal

parse_timestamps_json = VANTAGE_Temporal.parse_timestamps_json
timestamp_to_seconds = VANTAGE_Temporal.timestamp_to_seconds

DURATION = 30.0


# ---------------------------------------------------------------------------
# timestamp_to_seconds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("4.5", 4.5),
    ("9", 9.0),
    ("00:05", 5.0),
    ("1:05.5", 65.5),
    ("00:01:05", 65.0),
    ("01:02:03", 3723.0),
    (" 00:09 ", 9.0),
    (4.5, 4.5),
    (9, 9.0),
])
def test_timestamp_to_seconds(value, expected):
    assert timestamp_to_seconds(value) == pytest.approx(expected)


@pytest.mark.parametrize("value", ["", "abc", "1:2:3:4", None, True, [1, 2], {"a": 1}])
def test_timestamp_to_seconds_rejects_garbage(value):
    with pytest.raises(ValueError):
        timestamp_to_seconds(value)


# ---------------------------------------------------------------------------
# parse_timestamps_json: 'ss' / 'mm:ss' values, list wrapping, truncated output
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ('{"start": "00:05", "end": "00:09"}', [5.0, 9.0]),
    ('{"start": "4.5", "end": "9"}', [4.5, 9.0]),
    ('{"start": "1:05.5", "end": "1:10"}', [65.5, 70.0]),
    ('[{"start": "00:05", "end": "00:09"}]', [5.0, 9.0]),
    # Truncated output: the repair step closes the quote and the brace.
    ('{"start": "00:05", "end": "00:09', [5.0, 9.0]),
    # Truncated list: recovered by the regex fallback.
    ('[{"start": "00:05", "end": "00:09"}', [5.0, 9.0]),
    # Unquoted start value followed by a quoted end value.
    ('{"start": 5, "end": "00:09"}', [5.0, 9.0]),
])
def test_parse_timestamps_json_parity(text, expected):
    assert parse_timestamps_json(text, DURATION) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# parse_timestamps_json: hh:mm:ss and numeric values
# ---------------------------------------------------------------------------

def test_hh_mm_ss_is_converted_to_seconds():
    assert parse_timestamps_json('{"start": "00:01:05", "end": "00:01:10"}', 120.0) == pytest.approx([65.0, 70.0])


def test_numeric_json_values_are_accepted():
    assert parse_timestamps_json('{"start": 4.5, "end": 9.0}', DURATION) == pytest.approx([4.5, 9.0])
    assert parse_timestamps_json('{"start": 4, "end": 9}', DURATION) == pytest.approx([4.0, 9.0])
    assert parse_timestamps_json('[{"start": 4.5, "end": 9.0}]', DURATION) == pytest.approx([4.5, 9.0])


# ---------------------------------------------------------------------------
# parse_timestamps_json: unparseable input keeps the existing fallback policy
# ---------------------------------------------------------------------------

def test_unparseable_non_strict_falls_back_to_whole_video():
    assert parse_timestamps_json('no timestamps here', DURATION) == [0, DURATION]


def test_unparseable_strict_raises():
    with pytest.raises(ValueError):
        parse_timestamps_json('no timestamps here', DURATION, strict=True)


def test_non_numeric_string_values_still_raise_for_caller_fallback():
    # The evaluate() caller catches this and falls back to the regex parser.
    with pytest.raises(ValueError):
        parse_timestamps_json('{"start": "beginning", "end": "the end"}', DURATION)
