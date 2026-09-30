"""Unit tests for the SOT metric helpers in vlmeval/dataset/vantage_sot.py.

compute_sot_metrics returns the same key set whether or not a sequence
has any evaluable frame. A sequence with zero evaluable frames (object
absent in every non-init GT frame and no predicted box) is scored 0.0 on
every metric, including success_auc / precision_25 / precision_75, so the
per-sequence table printed by VANTAGE_SOT.evaluate() covers every
sequence of the submission.

The end-to-end test builds a VANTAGE_SOT instance the same way
adapter_sot.py does (__new__ plus a synthetic _gt_cache), so no prepared
SOT directory or video decoding is needed.
"""
import json

import pandas as pd
import pytest

from vlmeval.dataset.vantage_sot import VANTAGE_SOT, compute_sot_metrics
from vlmeval.smp import dump


def _metrics_with_frames():
    frame_ids = [0, 1, 2]
    gt = {0: [100, 100, 200, 200], 1: [100, 100, 200, 200], 2: [110, 110, 210, 210]}
    pred = {1: [100, 100, 200, 200], 2: [110, 110, 210, 210]}
    return compute_sot_metrics(gt, pred, frame_ids, occluded={})


def _metrics_without_frames():
    # Object present only in the init frame, absent afterwards, model
    # predicted nothing: zero evaluable frames.
    frame_ids = [0, 1, 2]
    gt = {0: [100, 100, 200, 200], 1: None, 2: None}
    return compute_sot_metrics(gt, {}, frame_ids, occluded={})


def test_perfect_track_is_unchanged():
    m = _metrics_with_frames()
    assert m['mean_iou'] == pytest.approx(1.0)
    assert m['success_auc'] == pytest.approx(1.0)
    assert m['precision'] == pytest.approx(1.0)
    assert m['n_eval_frames'] == 2


def test_zero_evaluable_frames_has_full_key_set():
    empty = _metrics_without_frames()
    full = _metrics_with_frames()
    assert set(empty) == set(full)
    assert empty['n_eval_frames'] == 0
    for k, v in empty.items():
        if k != 'n_eval_frames':
            assert v == 0.0, (k, v)


def test_evaluate_survives_zero_evaluable_sequence(tmp_path):
    ds = VANTAGE_SOT.__new__(VANTAGE_SOT)
    ds.verbose = True  # also exercise the per-sequence verbose line
    ds.model_family = 'cr'
    ds.data = pd.DataFrame([])
    ds._gt_cache = {
        'seq_ok': {
            'frame_ids': [0, 1, 2],
            'gt_bboxes': {0: [100, 100, 200, 200], 1: [100, 100, 200, 200], 2: [110, 110, 210, 210]},
            'occluded': {},
            'init_bbox': [100, 100, 200, 200],
            'object_type': 'Person',
            'label': 'Warehouse_000/Camera_0000/1/obj1',
            'video_path': '', 'frame_paths': [], 'crop_path': '', 'seq_dir': '',
        },
        'seq_empty': {
            'frame_ids': [0, 1, 2],
            'gt_bboxes': {0: [100, 100, 200, 200], 1: None, 2: None},
            'occluded': {},
            'init_bbox': [100, 100, 200, 200],
            'object_type': 'Person',
            'label': 'Warehouse_000/Camera_0000/1/obj2',
            'video_path': '', 'frame_paths': [], 'crop_path': '', 'seq_dir': '',
        },
    }
    good = json.dumps({'frame_1': [100, 100, 200, 200], 'frame_2': [110, 110, 210, 210]})
    absent = json.dumps({'frame_1': None, 'frame_2': None})
    pred_path = tmp_path / 'preds.xlsx'
    dump(pd.DataFrame([
        {'index': 'seq_ok', 'prediction': good},
        {'index': 'seq_empty', 'prediction': absent},
    ]), str(pred_path))

    result = ds.evaluate(str(pred_path))

    assert set(result) == {'mean_iou', 'success_auc', 'precision_at_0_5'}
    # Both sequences are scored: the perfect one at 1.0 and the empty one at 0.
    assert result['mean_iou'] == pytest.approx(0.5)
    assert result['success_auc'] == pytest.approx(0.5)
    assert result['precision_at_0_5'] == pytest.approx(0.5)
    per_seq = json.loads((tmp_path / 'preds_sot_results.json').read_text())
    assert per_seq['Warehouse_000/Camera_0000/1/obj2']['success_auc'] == 0.0
    assert per_seq['Warehouse_000/Camera_0000/1/obj2']['n_eval_frames'] == 0
