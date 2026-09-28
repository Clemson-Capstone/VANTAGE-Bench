"""Per-sample error handling in the inference loops.

When model.generate raises for one sample (for example an unreadable clip or
an error inside the model's preprocessing), the error is recorded as the
prediction ('Failed to obtain answer: <Type>: <message>'), the traceback is
logged once per exception type, a per-dataset count is reported, and the
remaining samples are still run, so every dataset yields a complete
prediction file. VANTAGE_FAIL_FAST=1 raises on the first error instead.
Both the image and the video loops are exercised end-to-end with fake
models and datasets, no GPU needed.
"""
import logging

import pandas as pd
import pytest

from vlmeval import inference, inference_video
from vlmeval.inference import GEN_FAIL_MSG, generate_or_record, report_failed_samples
from vlmeval.smp import get_logger, load


class FlakyModel:
    """generate() raises for the indices in `bad`, echoes the prompt otherwise."""
    is_api = False
    VIDEO_LLM = True

    def __init__(self, bad):
        self.bad = bad
        self.calls = []

    def set_dump_image(self, func):
        pass

    def generate(self, message, dataset=None):
        text = message[-1]['value']
        self.calls.append(text)
        if text in self.bad:
            raise self.bad[text]
        return f'answer to {text}'


class FakeDataset:
    dataset_name = 'FakeDS'
    nframe = 8
    fps = 0

    def __init__(self, n):
        self.data = pd.DataFrame({'index': list(range(n)), 'question': [f'q{i}' for i in range(n)]})

    def __len__(self):
        return len(self.data)

    def dump_image(self, line):
        return []

    def build_prompt(self, line, video_llm=False):
        if isinstance(line, int):
            line = self.data.iloc[line]
        return [dict(type='text', value=line['question'])]


@pytest.fixture
def run_log(caplog):
    # get_logger sets propagate=False, so attach caplog's handler directly.
    logger = get_logger('RUN')
    logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger='RUN')
    yield caplog
    logger.removeHandler(caplog.handler)


def test_generate_or_record_formats_failure_and_logs_traceback_once(run_log, monkeypatch):
    monkeypatch.delenv('VANTAGE_FAIL_FAST', raising=False)
    model = FlakyModel({'a': ValueError('nframes should in interval [2, 5], but got 8.'),
                        'b': ValueError('second'), 'c': KeyError('other')})
    seen = set()
    out_a = generate_or_record(model, [dict(type='text', value='a')], 'DS', 1, seen)
    out_b = generate_or_record(model, [dict(type='text', value='b')], 'DS', 2, seen)
    out_c = generate_or_record(model, [dict(type='text', value='c')], 'DS', 3, seen)
    out_ok = generate_or_record(model, [dict(type='text', value='d')], 'DS', 4, seen)

    assert out_a == f'{GEN_FAIL_MSG}: ValueError: nframes should in interval [2, 5], but got 8.'
    assert out_b == f'{GEN_FAIL_MSG}: ValueError: second'
    assert out_c == f"{GEN_FAIL_MSG}: KeyError: 'other'"
    assert out_ok == 'answer to d'
    assert seen == {'ValueError', 'KeyError'}

    errors = [r for r in run_log.records if r.levelno == logging.ERROR]
    warnings_ = [r for r in run_log.records if r.levelno == logging.WARNING]
    # Full traceback once per exception type, one-liner for the repeat.
    assert len(errors) == 2
    assert all('Traceback (most recent call last)' in r.getMessage() for r in errors)
    assert 'Sample 1 of DS failed with ValueError' in errors[0].getMessage()
    assert 'VANTAGE_FAIL_FAST=1' in errors[0].getMessage()
    assert len(warnings_) == 1
    assert warnings_[0].getMessage() == 'Sample 2 of DS failed with ValueError: second'


def test_generate_or_record_fail_fast_raises(monkeypatch):
    monkeypatch.setenv('VANTAGE_FAIL_FAST', '1')
    model = FlakyModel({'a': RuntimeError('boom')})
    with pytest.raises(RuntimeError, match='boom'):
        generate_or_record(model, [dict(type='text', value='a')], 'DS', 0, set())


def test_report_failed_samples_counts_prefix(run_log):
    res = {0: 'A', 1: f'{GEN_FAIL_MSG}: ValueError: x', 2: 'B', 3: f'{GEN_FAIL_MSG}: KeyError: y'}
    assert report_failed_samples(res, 'M', 'DS') == 2
    assert 'M/DS: 2 of 4 samples failed' in run_log.text
    assert report_failed_samples({0: 'A'}, 'M', 'DS') == 0


def test_video_infer_data_completes_and_records_failures(tmp_path, run_log, monkeypatch):
    monkeypatch.delenv('VANTAGE_FAIL_FAST', raising=False)
    dataset = FakeDataset(5)
    model = FlakyModel({'q1': ValueError('bad clip'), 'q3': ValueError('bad clip too')})
    out_file = str(tmp_path / 'rank0.pkl')
    inference_video.infer_data(model, 'FakeModel', str(tmp_path), dataset, out_file)
    res = load(out_file)
    assert sorted(res) == [0, 1, 2, 3, 4]
    assert res[0] == 'answer to q0'
    assert res[1] == f'{GEN_FAIL_MSG}: ValueError: bad clip'
    assert res[3] == f'{GEN_FAIL_MSG}: ValueError: bad clip too'
    assert res[4] == 'answer to q4'
    # Every sample was attempted, none skipped after the first failure.
    assert model.calls == ['q0', 'q1', 'q2', 'q3', 'q4']
    assert 'FakeModel/FakeDS: 2 of 5 samples failed' in run_log.text


def test_video_infer_data_fail_fast_aborts(tmp_path, monkeypatch):
    monkeypatch.setenv('VANTAGE_FAIL_FAST', '1')
    dataset = FakeDataset(3)
    model = FlakyModel({'q1': ValueError('bad clip')})
    with pytest.raises(ValueError, match='bad clip'):
        inference_video.infer_data(model, 'FakeModel', str(tmp_path), dataset, str(tmp_path / 'rank0.pkl'))
    assert model.calls == ['q0', 'q1']


def test_image_infer_data_completes_and_records_failures(tmp_path, run_log, monkeypatch):
    monkeypatch.delenv('VANTAGE_FAIL_FAST', raising=False)
    dataset = FakeDataset(4)
    model = FlakyModel({'q2': RuntimeError('CUDA error: device-side assert')})
    out_file = str(tmp_path / 'rank0.pkl')
    inference.infer_data(model, 'FakeModel', str(tmp_path), dataset, out_file)
    res = load(out_file)
    assert sorted(res) == [0, 1, 2, 3]
    assert res[2] == f'{GEN_FAIL_MSG}: RuntimeError: CUDA error: device-side assert'
    assert res[3] == 'answer to q3'
    assert model.calls == ['q0', 'q1', 'q2', 'q3']
    assert 'FakeModel/FakeDS: 1 of 4 samples failed' in run_log.text


def test_image_infer_data_fail_fast_aborts(tmp_path, monkeypatch):
    monkeypatch.setenv('VANTAGE_FAIL_FAST', '1')
    dataset = FakeDataset(3)
    model = FlakyModel({'q0': RuntimeError('boom')})
    with pytest.raises(RuntimeError, match='boom'):
        inference.infer_data(model, 'FakeModel', str(tmp_path), dataset, str(tmp_path / 'rank0.pkl'))
