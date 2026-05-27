"""
CosmosReason2 (and base CosmosReason) API for OpenAI-compatible endpoints (e.g. Lepton).

Set COSMOS_REASON2_API_BASE (your Lepton or compatible endpoint URL).
Key: COSMOS_API_KEY or NVIDIA_API_KEY (e.g. for Lepton endpoints).
No API keys or URLs are stored in code.
"""
import base64
import copy
import json
import os
import re
import subprocess

import numpy as np
import requests
import torch
from qwen_vl_utils import process_vision_info

from ..smp import *
from ..dataset import DATASET_TYPE
from .base import BaseAPI


def _tensor_video_to_base64(video_tensor, fps, crf=14):
    """Convert (T,C,H,W) RGB tensor to MP4 base64 using ffmpeg."""
    assert video_tensor.ndim == 4 and video_tensor.shape[1] == 3
    if video_tensor.dtype != torch.uint8:
        v = video_tensor
        if torch.is_floating_point(v) and v.max() <= 1.0:
            v = (v * 255.0)
        video_u8 = v.clamp(0, 255).to(torch.uint8)
    else:
        video_u8 = video_tensor
    T, C, H, W = video_u8.shape
    video_thwc = video_u8.permute(0, 2, 3, 1).contiguous()
    cmd = [
        "ffmpeg", "-threads", "1", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
        "-i", "-", "-an", "-vcodec", "libx264", "-preset", "fast", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "frag_keyframe+empty_moov", "-f", "mp4", "-"
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**6)
    try:
        out, err = proc.communicate(input=video_thwc.numpy().tobytes())
    except Exception as e:
        raise RuntimeError(f"ffmpeg communication failed: {e}") from e
    if proc.returncode != 0:
        err_msg = err.decode("utf-8", errors="ignore") if err else "No error output"
        raise RuntimeError(f"ffmpeg failed (returncode={proc.returncode}): {err_msg}")
    return base64.b64encode(out).decode("utf-8")


class _NoopVideoCache:
    """No-op cache so we don't require a separate video_cache module."""

    def get(self, video_path, image_patch_size=None, **kwargs):
        return (None, None)

    def set(self, video_path, video_url, video_kwargs, image_patch_size=None, **kwargs):
        pass


def process_video_info_to_video_url(video_path, image_patch_size=16, use_cache=True, **kwargs):
    """Process video file to base64 data URL for API. use_cache is ignored (no-op cache)."""
    messages = [{"role": "user", "content": [{"type": "video", "video": video_path, **kwargs}]}]
    _, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True, image_patch_size=image_patch_size
    )
    video_base64 = _tensor_video_to_base64(video_inputs[0], video_kwargs["fps"][0])
    video_url = f"data:video/mp4;base64,{video_base64}"
    video_kwargs = copy.deepcopy(video_kwargs)
    video_kwargs.update(dict(do_sample_frames=False, do_resize=False))
    del video_kwargs["fps"]
    return video_url, video_kwargs


def build_multi_choice_prompt(line, dataset=None):
    question = line["question"]
    hint = line.get("hint")
    if hint is not None and not (isinstance(hint, float) and np.isnan(hint)):
        question = str(hint) + "\n" + question
    options = {
        c: line[c] for c in "ABCD"
        if c in line and not (isinstance(line[c], float) and np.isnan(line[c]))
    }
    for k, v in options.items():
        question += f"\n{k}. {v}"
    if options:
        question += "\nPlease answer directly with only the letter of the correct option and nothing else."
    else:
        question += "\nAnswer the question directly."
    return question


class CosmosReason(BaseAPI):
    is_api = True
    VIDEO_LLM = True
    nframes = None
    fps = None
    total_pixels = None
    max_pixels = None
    min_pixels = None
    max_frames = None

    def __init__(
        self,
        model="nvidia/Cosmos-Reason2-8B",
        retry=5,
        wait=5,
        key=None,
        verbose=False,
        system_prompt=None,
        temperature=0,
        top_p=None,
        top_k=None,
        repetition_penalty=None,
        timeout=300,
        api_base=None,
        max_tokens=2048,
        seed=1,
        img_size=-1,
        img_detail="high",
        use_video_cache=False,
        **kwargs,
    ):
        self.model = model
        self.cur_idx = 0
        self.fail_msg = "Failed to obtain answer via API. "
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.seed = seed
        key = key or os.environ.get("COSMOS_API_KEY") or os.environ.get("NVIDIA_API_KEY", "")
        self.key = key
        self.img_size = img_size
        self.img_detail = img_detail
        self.timeout = timeout
        self.use_video_cache = use_video_cache
        self.nframes = kwargs.pop("nframes", None)
        self.fps = kwargs.pop("fps", None)
        self.total_pixels = kwargs.pop("total_pixels", None)
        self.max_pixels = kwargs.pop("max_pixels", None)
        self.min_pixels = kwargs.pop("min_pixels", None)
        self.max_frames = kwargs.pop("max_frames", None)
        self.image_patch_size = kwargs.pop("image_patch_size", 16)
        api_base = api_base or os.environ.get("COSMOS_REASON2_API_BASE", "").strip()
        if not api_base:
            raise ValueError(
                "Set COSMOS_REASON2_API_BASE to your Lepton (or compatible) endpoint URL, e.g. "
                "https://your-deployment.lepton.run/v1/chat/completions"
            )
        self.api_base = api_base
        super().__init__(wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)
        self.logger.info(f"Using API base: {self.api_base}")

    def use_custom_prompt(self, dataset):
        if dataset is None:
            return False
        return not listinstr(
            ["SparBench", "EgoPlanBench2", "CVBench", "EmbSpatialBench", "RoboSpatialHome", "RefSpatialBench", "SATBench", "Where2Place", "ERQA"],
            dataset,
        )

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)
        if dataset is not None and DATASET_TYPE(dataset) == "Y/N":
            prompt = line["question"] + " Answer the question using a single word or phrase."
        elif dataset is not None and (DATASET_TYPE(dataset) == "MCQ" or (DATASET_TYPE(dataset) or "").startswith("MCQ")):
            prompt = build_multi_choice_prompt(line, dataset)
        elif dataset is not None and listinstr(['Astro2D', 'VANTAGE_2D'], dataset):
            prompt = line["question"]
        elif dataset is not None and DATASET_TYPE(dataset) == "VQA":
            prompt = line["question"] + "\nAnswer the question using a single word or phrase."
        else:
            prompt = line["question"]
        message = [dict(type="image", value=s) for s in (tgt_path if isinstance(tgt_path, list) else [tgt_path])]
        message.append(dict(type="text", value=prompt))
        return message

    def parse_answer(self, answer: str) -> str:
        return answer

    def _get_video_processing_kwargs(self, msg):
        out = copy.deepcopy(msg)
        out.pop("type", None)
        out.pop("value", None)
        keys = ["nframes", "fps", "total_pixels", "max_pixels", "min_pixels", "max_frames"]
        if any(k in out for k in keys):
            return out
        for k in keys:
            v = getattr(self, k, None)
            if v is not None:
                out[k] = v
        return out

    def prepare_itlist(self, inputs):
        video_kwargs = None
        assert all(isinstance(x, dict) for x in inputs)
        has_mm = sum(1 for x in inputs if x.get("type") in ("image", "video", "video_base64"))
        if has_mm:
            content_list = []
            for msg in inputs:
                if msg["type"] == "text":
                    content_list.append(dict(type="text", text=msg["value"]))
                elif msg["type"] == "image":
                    from PIL import Image
                    img = Image.open(msg["value"])
                    b64 = encode_image_to_base64(img, target_size=self.img_size)
                    content_list.append(dict(type="image_url", image_url=dict(url=f"data:image/jpeg;base64,{b64}", detail=self.img_detail)))
                elif msg["type"] == "video_base64":
                    content_list.append(dict(type="video_url", video_url=dict(url=msg["value"])))
                elif msg["type"] == "video":
                    video_url, video_kwargs = process_video_info_to_video_url(
                        msg["value"],
                        image_patch_size=self.image_patch_size,
                        use_cache=self.use_video_cache,
                        **self._get_video_processing_kwargs(msg),
                    )
                    content_list.append(dict(type="video_url", video_url=dict(url=video_url)))
        else:
            text = "\n".join(x["value"] for x in inputs)
            content_list = [dict(type="text", text=text)]
        return content_list, video_kwargs

    def prepare_inputs(self, inputs):
        input_msgs = []
        video_kwargs = None
        if self.system_prompt is not None:
            input_msgs.append(dict(role="system", content=self.system_prompt))
        assert isinstance(inputs, list) and len(inputs) and isinstance(inputs[0], dict)
        if inputs[0].get("role") is not None:
            for item in inputs:
                content_list, vk = self.prepare_itlist(item["content"])
                if video_kwargs is None:
                    video_kwargs = vk
                input_msgs.append(dict(role=item["role"], content=content_list))
        else:
            content_list, video_kwargs = self.prepare_itlist(inputs)
            input_msgs.append(dict(role="user", content=content_list))
        return input_msgs, video_kwargs

    def generate_inner(self, inputs, **kwargs):
        input_msgs, video_kwargs = self.prepare_inputs(inputs)
        temperature = kwargs.pop("temperature", self.temperature)
        top_p = kwargs.pop("top_p", self.top_p)
        top_k = kwargs.pop("top_k", self.top_k)
        repetition_penalty = kwargs.pop("repetition_penalty", self.repetition_penalty)
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        seed = kwargs.pop("seed", self.seed)
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"}
        payload = {
            "model": self.model,
            "messages": input_msgs,
            "n": 1,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "max_tokens": max_tokens,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            **kwargs,
        }
        if video_kwargs is not None:
            payload["mm_processor_kwargs"] = video_kwargs
        try:
            response = requests.post(
                self.api_base, headers=headers, data=json.dumps(payload), timeout=int(self.timeout * 1.1)
            )
        except requests.exceptions.Timeout:
            self.logger.error(f"Endpoint timeout: {self.api_base}")
            return 408, self.fail_msg, None
        except requests.exceptions.ConnectionError as e:
            self.logger.error(f"Cannot connect to {self.api_base}: {e}")
            return 503, self.fail_msg, None
        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            return 500, self.fail_msg, None
        ret_code = 0 if 200 <= response.status_code < 300 else response.status_code
        answer = self.fail_msg
        try:
            data = response.json()
            answer = data["choices"][0]["message"]["content"].strip()
            answer = self.parse_answer(answer)
        except Exception as err:
            if self.verbose:
                self.logger.error(f"Parse error: {err}\n{response.text[:500]}")
        return ret_code, answer, response


class CosmosReason2(CosmosReason):
    def __init__(self, model="nvidia/Cosmos-Reason2-8B", **kwargs):
        kwargs.setdefault("image_patch_size", 16)
        super().__init__(model=model, **kwargs)

    def parse_answer(self, answer: str) -> str:
        # Strip <think>...</think> and return the rest as answer
        match = re.search(r"^<think>[\s\S]*?</think>\s*([\s\S]*)$", answer, re.DOTALL)
        if match:
            return match.group(1).strip()
        return answer.strip()
