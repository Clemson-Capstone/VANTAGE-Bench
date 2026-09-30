"""Frame sampling for short clips in the Qwen video paths.

A clip with fewer decodable frames than requested is sampled at its full
length: the requested nframes, including a dataset-supplied value, is
clamped to the clip's decodable frame count (rounded down to the frame
factor) before process_vision_info sees it. qwen_vl_utils' decord reader
accepts nframes only in [2, total_frames], so this keeps an eight-frame
request on a five-frame clip within range. FORCE_QWENVL_VIDEO_READER is
pinned to decord, which keeps the reader independent of
torchvision.io.read_video (not present in torchvision 0.29).
The fixture is a five-frame mp4 written with cv2.
"""
import functools
import importlib.util
import os
import sys
import types

import numpy as np
import pytest

from vlmeval.smp import clamp_video_nframes, pin_qwen_video_reader, video_frame_count

cv2 = pytest.importorskip('cv2')


def write_clip(path, n_frames, size=(64, 48), fps=5.0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    if not writer.isOpened():
        pytest.skip('cv2 cannot write mp4v files in this environment')
    for i in range(n_frames):
        frame = np.full((size[1], size[0], 3), (8 * i) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    if video_frame_count(str(path)) != n_frames:
        pytest.skip(f'could not write a {n_frames}-frame clip with cv2')
    return str(path)


@pytest.fixture
def short_clip(tmp_path):
    return write_clip(tmp_path / 'short.mp4', 5)


@pytest.fixture
def long_clip(tmp_path):
    return write_clip(tmp_path / 'long.mp4', 30)


def test_video_frame_count_reads_decodable_frames(short_clip, long_clip, tmp_path):
    assert video_frame_count(short_clip) == 5
    assert video_frame_count(long_clip) == 30
    assert video_frame_count(str(tmp_path / 'missing.mp4')) is None


def test_clamp_video_nframes(short_clip, long_clip, tmp_path):
    assert clamp_video_nframes(short_clip, 8) == 4      # 5 frames, rounded down to the frame factor
    assert clamp_video_nframes(short_clip, 5) == 5      # fits: unchanged
    assert clamp_video_nframes(short_clip, 4) == 4
    assert clamp_video_nframes(short_clip, 3) == 3
    assert clamp_video_nframes(long_clip, 8) == 8
    assert clamp_video_nframes(long_clip, 64) == 30
    assert clamp_video_nframes(str(tmp_path / 'missing.mp4'), 8) == 8  # unreadable: leave it to the reader


def _bare(cls, **attrs):
    obj = cls.__new__(cls)
    defaults = dict(min_pixels=None, max_pixels=None, total_pixels=None, fps=None, nframe=8, FRAME_FACTOR=2,
                    limit_mm_per_prompt=24)
    defaults.update(attrs)
    for k, v in defaults.items():
        setattr(obj, k, v)
    return obj


def _video_items(content):
    return [c for c in content if c['type'] == 'video']


def test_qwen3_vl_clamps_dataset_supplied_nframes(short_clip, long_clip):
    from vlmeval.vlm.qwen3_vl.model import Qwen3VLChat
    model = _bare(Qwen3VLChat)
    # The VANTAGE video datasets put nframes into the message (EventVerification, VQA, Temporal, DVC).
    msg = [dict(type='video', value=short_clip, nframes=8), dict(type='text', value='q')]
    (item,) = _video_items(model._prepare_content(msg))
    assert item['nframes'] == 4
    assert item['video'] == f'file://{short_clip}'
    # A clip that has enough frames keeps the request.
    (item,) = _video_items(model._prepare_content([dict(type='video', value=long_clip, nframes=8)]))
    assert item['nframes'] == 8
    # fps requests are bounded by qwen_vl_utils itself and are left alone.
    (item,) = _video_items(model._prepare_content([dict(type='video', value=short_clip, fps=2)]))
    assert item['fps'] == 2 and 'nframes' not in item


def test_qwen3_vl_clamps_model_default_nframes(short_clip):
    from vlmeval.vlm.qwen3_vl.model import Qwen3VLChat
    model = _bare(Qwen3VLChat, nframe=8, fps=None)
    (item,) = _video_items(model._prepare_content([dict(type='video', value=short_clip)]))
    assert item['nframes'] == 4


def test_qwen2_vl_clamps_nframes_in_both_content_builders(short_clip, long_clip):
    from vlmeval.vlm.qwen2_vl.model import Qwen2VLChat
    model = _bare(Qwen2VLChat, nframe=8, fps=None)
    for builder in (model._prepare_content, model._prepare_content_vllm):
        (item,) = _video_items(builder([dict(type='video', value=short_clip), dict(type='text', value='q')]))
        assert item['nframes'] == 4, builder.__name__
        (item,) = _video_items(builder([dict(type='video', value=long_clip), dict(type='text', value='q')]))
        assert item['nframes'] == 8, builder.__name__


def test_pin_qwen_video_reader_respects_user_setting(monkeypatch):
    monkeypatch.setenv('FORCE_QWENVL_VIDEO_READER', 'torchcodec')
    fake = types.SimpleNamespace(FORCE_QWENVL_VIDEO_READER='torchcodec')
    monkeypatch.setitem(sys.modules, 'qwen_vl_utils.vision_process', fake)
    assert pin_qwen_video_reader() == 'torchcodec'
    assert os.environ['FORCE_QWENVL_VIDEO_READER'] == 'torchcodec'
    assert fake.FORCE_QWENVL_VIDEO_READER == 'torchcodec'


@pytest.mark.skipif(importlib.util.find_spec('decord') is None, reason='decord not installed')
def test_pin_qwen_video_reader_forces_decord_and_patches_loaded_module(monkeypatch):
    monkeypatch.delenv('FORCE_QWENVL_VIDEO_READER', raising=False)
    calls = []

    @functools.lru_cache(maxsize=1)
    def get_video_reader_backend():
        calls.append(1)
        return 'torchvision'

    # Mimic qwen_vl_utils imported before the model was built, with the env var read as None.
    fake = types.SimpleNamespace(FORCE_QWENVL_VIDEO_READER=None, get_video_reader_backend=get_video_reader_backend)
    fake.get_video_reader_backend()
    monkeypatch.setitem(sys.modules, 'qwen_vl_utils.vision_process', fake)

    assert pin_qwen_video_reader() == 'decord'
    assert os.environ['FORCE_QWENVL_VIDEO_READER'] == 'decord'
    assert fake.FORCE_QWENVL_VIDEO_READER == 'decord'
    assert fake.get_video_reader_backend.cache_info().currsize == 0  # cached choice was cleared


@pytest.mark.skipif(importlib.util.find_spec('qwen_vl_utils') is None or importlib.util.find_spec('decord') is None,
                    reason='qwen_vl_utils and decord are needed for the end-to-end check')
def test_qwen_vl_utils_decord_reader_accepts_clamped_request(short_clip):
    from qwen_vl_utils import vision_process
    # An eight-frame request on the five-frame clip is outside the reader's range.
    with pytest.raises(ValueError, match=r'nframes should in interval \[2, 5\], but got 8'):
        vision_process._read_video_decord({'video': short_clip, 'nframes': 8})
    video = vision_process._read_video_decord({'video': short_clip, 'nframes': clamp_video_nframes(short_clip, 8)})[0]
    assert video.shape[0] == 4
