from PIL import Image
import torch

from .base import BaseModel
from ..smp import *

from io import BytesIO
import base64
from mimetypes import guess_type


class PaliGemma(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='google/paligemma-3b-mix-448', **kwargs):
        try:
            from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
        except Exception as e:
            logging.critical('Please install the latest version transformers.')
            raise e

        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
        ).eval()
        self.model = model.cuda()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.kwargs = kwargs

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        image = Image.open(image_path).convert('RGB')

        model_inputs = self.processor(
            text=prompt, images=image, return_tensors='pt'
        ).to('cuda')
        input_len = model_inputs['input_ids'].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **model_inputs, max_new_tokens=512, do_sample=False
            )
            generation = generation[0][input_len:]
            res = self.processor.decode(generation, skip_special_tokens=True)
        return res


class Gemma3(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='google/gemma-3-4b-it', **kwargs):
        logging.info(
            "Please install transformers via \n"
            "pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3"
        )
        try:
            from transformers import AutoProcessor, Gemma3ForConditionalGeneration
            import torch
        except Exception as e:
            logging.critical('Please install torch and transformers')
            raise e

        self.use_vllm = kwargs.get('use_vllm', False)
        self.limit_mm_per_prompt = 24
        if self.use_vllm:
            from vllm import LLM, SamplingParams
            # Set tensor_parallel_size [8, 4, 2, 1] based on the number of available GPUs
            gpu_count = torch.cuda.device_count()
            if gpu_count >= 8:
                tp_size = 8
            elif gpu_count >= 4:
                tp_size = 4
            elif gpu_count >= 2:
                tp_size = 2
            else:
                tp_size = 1
            logging.info(
                f'Using vLLM for Llama4 inference with {tp_size} GPUs (available: {gpu_count})'
            )
            import os
            if os.environ.get('VLLM_WORKER_MULTIPROC_METHOD') != 'spawn':
                logging.warning(
                    'VLLM_WORKER_MULTIPROC_METHOD is not set to spawn.'
                    'Use \'export VLLM_WORKER_MULTIPROC_METHOD=spawn\' to avoid potential multi-process issues'
                )
            self.llm = LLM(
                model=model_path,
                max_num_seqs=4,
                max_model_len=16384,
                limit_mm_per_prompt={"image": self.limit_mm_per_prompt},
                tensor_parallel_size=tp_size,
                gpu_memory_utilization=kwargs.get("gpu_utils", 0.9),
            )
            # export VLLM_WORKER_MULTIPROC_METHOD=spawn
        else:
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                model_path, device_map="cuda", attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
            ).eval()
            self.device = self.model.device

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.system_prompt = kwargs.pop('system_prompt', 'You are a helpful assistant. ')
        default_kwargs = {
            'do_sample': False,
            'max_new_tokens': 4096
        }
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs

    def message2pipeline(self, message):
        ret = []
        if hasattr(self, 'system_prompt') and self.system_prompt is not None:
            ret = [
                dict(role='system', content=[dict(type='text', text=self.system_prompt)])
            ]
        content = []
        for m in message:
            if m['type'] == 'text':
                content.append(dict(type='text', text=m['value']))
            elif m['type'] == 'image':
                content.append(dict(type='image', url=m['value']))
        ret.append(dict(role='user', content=content))
        return ret

    def encode_image(self, image_path):
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = "image/jpeg"
        image_format = mime_type.split("/")[-1].upper() if mime_type else "JPEG"
        image = Image.open(image_path)
        # Handle the alpha channel
        if image.mode == "RGBA":
            image = self._rgba_to_rgb(image)

        encoded_image = self._encode_image(image, image_format)

        return encoded_image

    def _encode_image(self, image, image_format):
        with BytesIO() as output:
            image.convert("RGB").save(output, format=image_format)
            base64_encoded_data = base64.b64encode(output.getvalue()).decode("utf-8")
        return base64_encoded_data

    @staticmethod
    def _rgba_to_rgb(image):
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, image).convert("RGB")

    def message_to_promptimg_vllm(self, message, dataset=None):
        processed_message = []
        images = []
        num_images = 0
        for item in message:
            if item['type'] == 'text':
                processed_message.append({
                    "type": "text",
                    "text": item['value']
                })
            elif item['type'] == 'image':
                if num_images < self.limit_mm_per_prompt:
                    image_path = item['value']
                    encoded_image = self.encode_image(image_path)
                    image = Image.open(BytesIO(base64.b64decode(encoded_image)))
                    image.load()
                    processed_message.append({
                        "type": "image",
                        "image": "",
                    })
                    images.append(image)
                    num_images += 1
        if num_images >= self.limit_mm_per_prompt:
            logging.warning(
                f"Number of images exceeds the limit of {self.limit_mm_per_prompt}."
                f"Only the first {self.limit_mm_per_prompt} images will be used."
            )
        return processed_message, images

    def generate_inner_transformers(self, message, dataset=None):
        messages = self.message2pipeline(message)
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.device, dtype=torch.bfloat16)

        input_len = inputs['input_ids'].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**inputs, **self.kwargs)
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded

    def generate_inner_vllm(self, message, dataset=None):
        from vllm import LLM, SamplingParams
        prompt, images = self.message_to_promptimg_vllm(message, dataset=dataset)
        messages = [
            {'role': 'user', 'content': prompt}
        ]
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        sampling_params = SamplingParams(temperature=0.0,
                                         max_tokens=self.kwargs['max_new_tokens'])
        outputs = self.llm.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": images
                },
            },
            sampling_params=sampling_params
        )
        for o in outputs:
            generated_text = o.outputs[0].text
        return generated_text

    def generate_inner(self, message, dataset=None):
        if self.use_vllm:
            return self.generate_inner_vllm(message, dataset=dataset)
        else:
            return self.generate_inner_transformers(message, dataset=dataset)


class Gemma4(BaseModel):
    """Unified wrapper for the Gemma 4 model family.

    Supports gemma-4-E2B-it, gemma-4-26B-A4B-it, gemma-4-31B-it and any
    other google/gemma-4-*-it checkpoint.  Video is handled by extracting
    frames with OpenCV so VANTAGE video benchmarks work out of the box
    (VIDEO_LLM=True: dataset sends the raw video path; this wrapper samples
    it into images).
    """

    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    # Datasets where a short decode budget is sufficient (MCQ / binary answers)
    _MCQ_DATASET_PREFIXES = (
        'VANTAGE_VQA',
        'VANTAGE_EventVerification',
        'VANTAGE_2DGrounding',
        'VANTAGE_2DPointing',
    )
    _MCQ_MAX_NEW_TOKENS = 128

    # Valid visual token budgets supported by Gemma4ImageProcessor.
    # 70 is the native video-mode budget (used by Gemma4's own VideoProcessor).
    # 280 is the image-mode default and backward-compatible baseline.
    _VALID_TOKEN_BUDGETS = (70, 140, 280, 560, 1120)


    def __init__(
        self,
        model_path='google/gemma-4-9b-it',
        nframe=16,
        fps=None,
        max_new_tokens=4096,
        max_frames=64,
        image_tokens_per_frame=280,
        system_prompt='You are a helpful assistant.',
        use_vllm=False,
        **kwargs,
    ):
        if image_tokens_per_frame not in self._VALID_TOKEN_BUDGETS:
            raise ValueError(
                f'image_tokens_per_frame must be one of {self._VALID_TOKEN_BUDGETS}, '
                f'got {image_tokens_per_frame}'
            )
        self.nframe = nframe
        self.fps = fps
        self.max_frames = max_frames
        self.image_tokens_per_frame = image_tokens_per_frame
        self.system_prompt = system_prompt
        self.use_vllm = use_vllm
        self.limit_mm_per_prompt = 64
        self.model_path = model_path

        if use_vllm:
            from vllm import LLM, SamplingParams  # noqa: F401
            gpu_count = torch.cuda.device_count()
            tp_size = max(1, gpu_count)
            import os as _os
            _os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
            llm_kwargs = dict(
                model=model_path,
                max_num_seqs=4,
                limit_mm_per_prompt={'image': self.limit_mm_per_prompt},
                tensor_parallel_size=tp_size,
                gpu_memory_utilization=kwargs.get('gpu_utils', 0.9),
                trust_remote_code=True,
            )
            if 'max_model_len' in kwargs:
                llm_kwargs['max_model_len'] = kwargs['max_model_len']
            self.llm = LLM(**llm_kwargs)
        else:
            try:
                from transformers import AutoModelForImageTextToText
            except ImportError as e:
                logging.critical('Please install transformers >= 4.51.')
                raise e
            try:
                import flash_attn  # noqa: F401
                attn_impl = 'flash_attention_2'
            except ImportError:
                attn_impl = 'sdpa'
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map='auto',
                attn_implementation=attn_impl,
            ).eval()
            self.device = self.model.device

        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_path, padding_side='left')

        default_kwargs = {'do_sample': False, 'max_new_tokens': max_new_tokens}
        default_kwargs.update({
            k: v for k, v in kwargs.items()
            if k not in ('gpu_utils', 'max_model_len')
        })
        self.kwargs = default_kwargs

    # ------------------------------------------------------------------
    # Video frame extraction
    # ------------------------------------------------------------------

    def _extract_frames_to_paths(self, video_path, nframes):
        """Extract ``nframes`` uniformly sampled frames from *video_path*.

        Frames are written to a deterministic tmp directory so repeated calls
        on the same video are cached.
        """
        import cv2
        import hashlib
        import os as _os

        cache_key = hashlib.md5(f'{video_path}_{nframes}'.encode()).hexdigest()[:12]
        cache_dir = _os.path.join(_os.environ.get('TMPDIR', '/tmp'), 'gemma4_frames', cache_key)
        _os.makedirs(cache_dir, exist_ok=True)

        frame_paths = [_os.path.join(cache_dir, f'{i:04d}.jpg') for i in range(nframes)]
        if all(_os.path.exists(p) for p in frame_paths):
            return frame_paths

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            logging.warning(f'Gemma4: could not read video {video_path}')
            return []

        actual_n = min(nframes, total)
        indices = [int(i * total / actual_n) for i in range(actual_n)]
        frame_paths = frame_paths[:actual_n]

        for out_path, frame_idx in zip(frame_paths, indices):
            if _os.path.exists(out_path):
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(out_path, frame)
        cap.release()
        return [p for p in frame_paths if _os.path.exists(p)]

    # ------------------------------------------------------------------
    # Frame count computation
    # ------------------------------------------------------------------

    def _compute_nframes(self, msg):
        """Return how many frames to extract for a video message.

        Priority for frame count:
          1. explicit msg['nframes'] (dataset nframe passed through build_prompt)
          2. fps-derived count — msg['fps'] takes precedence over self.fps
          3. self.nframe fallback (wrapper/model config default)

        Frame cap applied last: self.max_frames (hard OOM protection ceiling).
        """
        import cv2 as _cv2
        nframes = int(msg.get('nframes', 0) or 0)

        # Effective fps: dataset fps (msg key) takes precedence over wrapper fps.
        # Effective fps: dataset fps (msg key) takes precedence over wrapper fps.
        # rather than silently falling back to the wrapper's nframe default.
        effective_fps = float(msg.get('fps', 0) or 0)
        if effective_fps <= 0 and self.fps and self.fps > 0:
            effective_fps = self.fps

        if nframes <= 0 and effective_fps > 0:
            cap = _cv2.VideoCapture(msg['value'])
            total = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
            src_fps = cap.get(_cv2.CAP_PROP_FPS) or 30.0
            cap.release()
            if total > 0:
                nframes = max(1, round((total / src_fps) * effective_fps))
        if nframes <= 0:
            nframes = self.nframe or 16
        return min(nframes, self.max_frames)

    # ------------------------------------------------------------------
    # Message formatting
    # ------------------------------------------------------------------

    def message2pipeline(self, message):
        """Convert the VLMEvalKit message list to (messages, pil_images).

        Returns a tuple of:
        - messages: chat-template messages list with {type: 'image'} placeholders
        - pil_images: list of PIL Images in order of appearance
        """
        ret = []
        if self.system_prompt:
            ret.append({'role': 'system', 'content': [{'type': 'text', 'text': self.system_prompt}]})

        content = []
        pil_images = []
        for m in message:
            if m['type'] == 'text':
                content.append({'type': 'text', 'text': m['value']})
            elif m['type'] == 'image':
                pil_images.append(Image.open(m['value']).convert('RGB'))
                content.append({'type': 'image'})
            elif m['type'] == 'video':
                nframes = self._compute_nframes(m)
                frame_paths = self._extract_frames_to_paths(m['value'], nframes)
                for fp in frame_paths:
                    pil_images.append(Image.open(fp).convert('RGB'))
                    content.append({'type': 'image'})
            else:
                logging.warning(f'Gemma4: unknown message type {m["type"]}, skipping.')

        ret.append({'role': 'user', 'content': content})
        return ret, pil_images

    # ------------------------------------------------------------------
    # vLLM path
    # ------------------------------------------------------------------

    def _collect_images_for_vllm(self, message):
        """Return (processed_content, pil_images) for vLLM input."""
        content = []
        images = []
        for m in message:
            if m['type'] == 'text':
                content.append({'type': 'text', 'text': m['value']})
            elif m['type'] == 'image':
                img = Image.open(m['value']).convert('RGB')
                images.append(img)
                content.append({'type': 'image', 'image': ''})
            elif m['type'] == 'video':
                nframes = self._compute_nframes(m)
                frame_paths = self._extract_frames_to_paths(m['value'], nframes)
                for fp in frame_paths:
                    if len(images) < self.limit_mm_per_prompt:
                        images.append(Image.open(fp).convert('RGB'))
                        content.append({'type': 'image', 'image': ''})
        return content, images

    def generate_inner_vllm(self, message, dataset=None):
        from vllm import SamplingParams

        content, images = self._collect_images_for_vllm(message)
        messages = []
        if self.system_prompt:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': content})
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        max_tokens = self.kwargs.get('max_new_tokens', 4096)
        if dataset and any(dataset.startswith(p) for p in self._MCQ_DATASET_PREFIXES):
            max_tokens = min(max_tokens, self._MCQ_MAX_NEW_TOKENS)
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
        )
        req = {'prompt': prompt}
        if images:
            req['multi_modal_data'] = {'image': images}
        outputs = self.llm.generate([req], sampling_params=sampling_params)
        return outputs[0].outputs[0].text

    # ------------------------------------------------------------------
    # HuggingFace Transformers path
    # ------------------------------------------------------------------

    def generate_inner_transformers(self, message, dataset=None):
        messages, pil_images = self.message2pipeline(message)
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        proc_kwargs = dict(text=text, return_tensors='pt')
        if pil_images:
            proc_kwargs['images'] = pil_images
            proc_kwargs['max_soft_tokens'] = self.image_tokens_per_frame
        inputs = self.processor(**proc_kwargs).to(self.device, dtype=torch.bfloat16)

        input_len = inputs['input_ids'].shape[-1]

        gen_kwargs = dict(self.kwargs)
        if dataset and any(dataset.startswith(p) for p in self._MCQ_DATASET_PREFIXES):
            gen_kwargs['max_new_tokens'] = min(
                gen_kwargs.get('max_new_tokens', 4096), self._MCQ_MAX_NEW_TOKENS
            )

        with torch.inference_mode():
            generation = self.model.generate(**inputs, **gen_kwargs)
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded

    def generate_inner(self, message, dataset=None):
        if self.use_vllm:
            return self.generate_inner_vllm(message, dataset=dataset)
        return self.generate_inner_transformers(message, dataset=dataset)
