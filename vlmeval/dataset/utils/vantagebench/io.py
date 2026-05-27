"""JSONL read/write helpers for the canonical schema layer.

Thin wrappers around the JSONL serialization conventions used elsewhere in
VLMEvalKit. Kept here so canonical modules have a single import point and so
encoding/line-ending choices are explicit.
"""

import json


def read_jsonl(path):
    """Read a JSONL file. One JSON object per line. UTF-8."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, start=1):
            line = line.rstrip('\n').rstrip('\r')
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON on line {lineno} of {path}: {e}") from e
    return records


def write_jsonl(records, path):
    """Write a list of JSON-serializable records to a JSONL file.

    UTF-8, ensure_ascii=False, LF line endings, no trailing whitespace.
    Order is preserved verbatim (no sorting).
    """
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write('\n')
