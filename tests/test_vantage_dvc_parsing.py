"""Unit tests for the DVC parsing helpers in vlmeval/dataset/vantage_dvc.py.

These exercise VANTAGE_DVC.parse_timestamp and
VANTAGE_DVC.parse_events_from_json in isolation. Both are @staticmethod, so
no dataset instance is constructed and no video/torch machinery is touched
directly by this test (importing the module does pull in vlmeval's usual
dependency chain, but nothing here constructs a VANTAGE_DVC instance or
calls into torch/video code paths).
"""
import math

import pytest

from vlmeval.dataset.vantage_dvc import VANTAGE_DVC

parse_timestamp = VANTAGE_DVC.parse_timestamp
parse_events_from_json = VANTAGE_DVC.parse_events_from_json


# ---------------------------------------------------------------------------
# parse_timestamp: parity controls (must stay byte-identical to old behavior)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ts_str,expected", [
    (None, 0.0),
    ('', 0.0),
    ('4.72', 4.72),
    ('00:04.72', 4.72),
    ('00:01:04.72', 64.72),
    ('1:02', 62.0),
    ('01:02:03', 3723.0),
    (5, 5.0),
    (5.5, 5.5),
])
def test_parse_timestamp_parity(ts_str, expected):
    assert parse_timestamp(ts_str) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# parse_timestamp: previously-crashing inputs must now return None, never raise
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ts_str", [
    'Frame 1',
    'Image 1',
    'The scene opens showing the automated...',
])
def test_parse_timestamp_unparseable_returns_none(ts_str):
    assert parse_timestamp(ts_str) is None


# ---------------------------------------------------------------------------
# parse_timestamp: tolerated real-world formats
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ts_str,expected", [
    ('4.72s', 4.72),
    ('4.72 sec', 4.72),
    ('4.72 seconds', 4.72),
    ('1m5s', 65.0),
    ('2h3m4s', 7384.0),
    ('00:00:00,000', 0.0),
    ('00:00:04,720', 4.72),
    ('00:04.72 - 00:09.31', 4.72),
    ('00:04:32:10', 272.0),  # hh:mm:ss:ff -- frames dropped, see comment in source
])
def test_parse_timestamp_tolerated_formats(ts_str, expected):
    result = parse_timestamp(ts_str)
    assert result is not None
    assert result == pytest.approx(expected)


def test_parse_timestamp_never_raises_on_garbage():
    garbage = [
        'Frame 1', 'Image 1', 'not a timestamp at all', '::::', '1:2:3:4:5',
        'abc', '4.72x', '-', ' - ', 'm5s2h', '00:00:00,000,000',
    ]
    for g in garbage:
        # Must never raise -- either a float or None.
        result = parse_timestamp(g)
        assert result is None or isinstance(result, float)


# ---------------------------------------------------------------------------
# parse_events_from_json
# ---------------------------------------------------------------------------

def test_parse_events_from_json_nan():
    assert parse_events_from_json(float('nan')) == []


def test_parse_events_from_json_none():
    assert parse_events_from_json(None) == []


def test_parse_events_from_json_empty_string():
    assert parse_events_from_json('') == []


def test_parse_events_from_json_number():
    # Defensive: any other non-string type should also coerce, not raise.
    assert parse_events_from_json(0) == []


def test_parse_events_from_json_list_of_strings_does_not_raise():
    result = parse_events_from_json('["a", "b"]')
    assert result == ["a", "b"]


def test_parse_events_from_json_fenced_block():
    text = (
        "Here are the events:\n"
        "```json\n"
        "[\n"
        "  {\"start\": \"00:00.00\", \"end\": \"00:05.00\", \"caption\": \"A man walks in.\"},\n"
        "  {\"start\": \"00:05.00\", \"end\": \"00:10.00\", \"caption\": \"He sits down.\"}\n"
        "]\n"
        "```\n"
    )
    result = parse_events_from_json(text)
    assert result == [
        {"start": "00:00.00", "end": "00:05.00", "caption": "A man walks in."},
        {"start": "00:05.00", "end": "00:10.00", "caption": "He sits down."},
    ]


# ---------------------------------------------------------------------------
# parse_timestamp: 'start - end' ranges pick the endpoint the field asks for
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ts_str,start_expected,end_expected", [
    ('00:04.72 - 00:09.31', 4.72, 9.31),
    ('4 - 9', 4.0, 9.0),
    ('00:00:04,720 - 00:00:09,310', 4.72, 9.31),
    ('1m5s - 1m10s', 65.0, 70.0),
])
def test_parse_timestamp_range_endpoints(ts_str, start_expected, end_expected):
    # Default (and explicit which='start') takes the first endpoint.
    assert parse_timestamp(ts_str) == pytest.approx(start_expected)
    assert parse_timestamp(ts_str, which='start') == pytest.approx(start_expected)
    # An 'end' field takes the second endpoint.
    assert parse_timestamp(ts_str, which='end') == pytest.approx(end_expected)


@pytest.mark.parametrize("ts_str,expected", [
    ('4.72', 4.72),
    ('00:01:04.72', 64.72),
    (5.5, 5.5),
    (None, 0.0),
    ('', 0.0),
])
def test_parse_timestamp_which_end_is_noop_without_a_range(ts_str, expected):
    assert parse_timestamp(ts_str, which='end') == pytest.approx(expected)


def test_parse_timestamp_which_end_unparseable_returns_none():
    assert parse_timestamp('Frame 1 - Frame 2', which='end') is None
