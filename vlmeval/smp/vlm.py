import os
import io
import pandas as pd
import numpy as np
import string
from uuid import uuid4
import os.path as osp
import base64
from PIL import Image
import sys

Image.MAX_IMAGE_PIXELS = 1e9


def rescale_img(img, tgt=None):
    assert isinstance(tgt, tuple) and -1 in tgt
    w, h = img.size
    if tgt[0] != -1:
        new_w, new_h = tgt[0], int(tgt[0] / w * h)
    elif tgt[1] != -1:
        new_w, new_h = int(tgt[1] / h * w), tgt[1]
    img = img.resize((new_w, new_h))
    return img


def concat_images_vlmeval(images, target_size=-1, mode='h', return_image=False):
    from .file import md5

    ims = [Image.open(im) for im in images]
    if target_size != -1:
        ims = [
            rescale_img(im, (-1, target_size) if mode == 'h' else (target_size, -1))
            for im in ims
        ]

    ws, hs = [x.width for x in ims], [x.height for x in ims]
    if mode == 'h':
        new_w, new_h = sum(ws), max(hs)
        dst = Image.new('RGB', (new_w, new_h))
        for i, im in enumerate(ims):
            dst.paste(im, (sum(ws[:i]), 0))
    elif mode == 'v':
        new_w, new_h = max(ws), sum(hs)
        dst = Image.new('RGB', (new_w, new_h))
        for i, im in enumerate(ims):
            dst.paste(im, (sum(ws[:i], 0)))
    if return_image:
        return dst
    else:
        _str = '\n'.join(images)
        str_md5 = md5(_str)
        tgt = osp.join('/tmp', str_md5 + '.jpg')
        dst.save(tgt)
        return tgt


def mmqa_display(question, target_size=-1):
    question = {k.lower(): v for k, v in question.items()}
    keys = list(question.keys())
    keys = [k for k in keys if k not in ['index', 'image']]

    if 'image' in question:
        images = question.pop('image')
        if images[0] == '[' and images[-1] == ']':
            images = eval(images)
        else:
            images = [images]
    else:
        images = question.pop('image_path')
        if images[0] == '[' and images[-1] == ']':
            images = eval(images)
        else:
            images = [images]
        images = [encode_image_file_to_base64(x) for x in images]

    idx = question.pop('index', 'XXX')
    print(f'INDEX: {idx}')

    for im in images:
        image = decode_base64_to_image(im, target_size=target_size)
        display(image)  # noqa: F821

    for k in keys:
        try:
            if not pd.isna(question[k]):
                print(f'{k.upper()}. {question[k]}')
        except ValueError:
            if False in pd.isna(question[k]):
                print(f'{k.upper()}. {question[k]}')


def resize_image_by_factor(img, factor=1):
    w, h = img.size
    new_w, new_h = int(w * factor), int(h * factor)
    img = img.resize((new_w, new_h))
    return img


def encode_image_to_base64(img, target_size=-1, fmt='JPEG'):
    # if target_size == -1, will not do resizing
    # else, will set the max_size ot (target_size, target_size)
    if img.mode in ('RGBA', 'P', 'LA'):
        img = img.convert('RGB')
    if target_size > 0:
        img.thumbnail((target_size, target_size))
    img_buffer = io.BytesIO()
    img.save(img_buffer, format=fmt)
    image_data = img_buffer.getvalue()
    ret = base64.b64encode(image_data).decode('utf-8')
    max_size = os.environ.get('VLMEVAL_MAX_IMAGE_SIZE', 1e9)
    min_edge = os.environ.get('VLMEVAL_MIN_IMAGE_EDGE', 1e2)
    max_size = int(max_size)
    min_edge = int(min_edge)

    if min(img.size) < min_edge:
        factor = min_edge / min(img.size)
        image_new = resize_image_by_factor(img, factor)
        img_buffer = io.BytesIO()
        image_new.save(img_buffer, format=fmt)
        image_data = img_buffer.getvalue()
        ret = base64.b64encode(image_data).decode('utf-8')

    factor = 1
    while len(ret) > max_size:
        factor *= 0.7  # Half Pixels Per Resize, approximately
        image_new = resize_image_by_factor(img, factor)
        img_buffer = io.BytesIO()
        image_new.save(img_buffer, format=fmt)
        image_data = img_buffer.getvalue()
        ret = base64.b64encode(image_data).decode('utf-8')

    if factor < 1:
        new_w, new_h = image_new.size
        print(
            f'Warning: image size is too large and exceeds `VLMEVAL_MAX_IMAGE_SIZE` {max_size}, '
            f'resize to {factor:.2f} of original size: ({new_w}, {new_h})'
        )

    return ret


def encode_image_file_to_base64(image_path, target_size=-1, fmt='JPEG'):
    image = Image.open(image_path)
    return encode_image_to_base64(image, target_size=target_size, fmt=fmt)


def decode_base64_to_image(base64_string, target_size=-1):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    if image.mode in ('RGBA', 'P', 'LA'):
        image = image.convert('RGB')
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image


def decode_base64_to_image_file(base64_string, image_path, target_size=-1):
    image = decode_base64_to_image(base64_string, target_size=target_size)
    base_dir = osp.dirname(image_path)
    if not osp.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    image.save(image_path)


def build_option_str(option_dict):
    s = 'There are several options: \n'
    for c, content in option_dict.items():
        if not pd.isna(content):
            s += f'{c}. {content}\n'
    return s


def isimg(s):
    return osp.exists(s) or s.startswith('http')


def read_ok(img_path):
    if not osp.exists(img_path):
        return False
    try:
        im = Image.open(img_path)
        assert im.size[0] > 0 and im.size[1] > 0
        return True
    except:
        return False


def gpt_key_set():
    openai_key = os.environ.get('OPENAI_API_KEY', None)
    if openai_key is None:
        openai_key = os.environ.get('AZURE_OPENAI_API_KEY', None)
        return isinstance(openai_key, str)
    return isinstance(openai_key, str) and openai_key.startswith('sk-')


def apiok(wrapper):
    s = wrapper.generate('Hello!')
    return wrapper.fail_msg not in s


def video_frame_count(video_path):
    """Number of decodable frames in a video file, or None if it cannot be opened.

    decord is tried first because it is what the dataset frame extraction and
    the qwen_vl_utils decord reader count with (len(VideoReader)); cv2's
    CAP_PROP_FRAME_COUNT comes from container metadata and can exceed what
    actually decodes. cv2 is the fallback when decord is not installed.
    """
    try:
        import decord
        return len(decord.VideoReader(video_path))
    except ImportError:
        pass
    except Exception:
        return None
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        try:
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        return n if n > 0 else None
    except Exception:
        return None


def clamp_video_nframes(video_path, nframes, frame_factor=2):
    """Clamp a requested frame count to what the clip has.

    qwen_vl_utils rejects nframes outside [frame_factor, total_frames], so a
    clip shorter than the requested count is sampled at its full length,
    rounded down to a multiple of frame_factor. Requests that fit, and clips
    whose length cannot be read, are returned unchanged.
    """
    total = video_frame_count(video_path)
    if total is None or nframes <= total:
        return nframes
    clamped = total // frame_factor * frame_factor
    if clamped < frame_factor:
        return nframes
    print(f'Video {video_path} has {total} frames, sampling {clamped} instead of the requested {nframes}')
    return clamped


def pin_qwen_video_reader():
    """Pin qwen_vl_utils to a video reader that exists in this environment.

    When its preferred reader raises, qwen_vl_utils falls back to torchvision.io.read_video,
    which torchvision 0.29 (installed with torch 2.14) removed. FORCE_QWENVL_VIDEO_READER
    is honoured if the user set it; otherwise decord (the reader the harness uses
    everywhere else) or torchcodec is forced. qwen_vl_utils reads the variable at import
    time, so the already-imported module is patched as well. Returns the reader name, or
    None when neither reader is installed.
    """
    import importlib.util
    backend = os.environ.get('FORCE_QWENVL_VIDEO_READER')
    if not backend:
        for candidate in ('decord', 'torchcodec'):
            if importlib.util.find_spec(candidate) is not None:
                backend = candidate
                break
        if backend is None:
            return None
        os.environ['FORCE_QWENVL_VIDEO_READER'] = backend
    vp = sys.modules.get('qwen_vl_utils.vision_process')
    if vp is not None and getattr(vp, 'FORCE_QWENVL_VIDEO_READER', None) != backend:
        vp.FORCE_QWENVL_VIDEO_READER = backend
        cache_clear = getattr(getattr(vp, 'get_video_reader_backend', None), 'cache_clear', None)
        if cache_clear is not None:
            cache_clear()
    return backend
