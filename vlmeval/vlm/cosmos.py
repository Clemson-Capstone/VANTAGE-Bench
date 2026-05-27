# vlmeval/vlm/cosmos.py
import os
import torch
from .base import BaseModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vlmeval.dataset import DATASET_TYPE

def _message_for_promptimg(message, max_pixels=None):
    processed_messages = []
    for part in message:
        if part["type"] == "text":
            processed_messages.append({"type": "text", "text": part["value"].strip()})
            
        elif part["type"] == "image":
            img_msg = {"type": "image", "image": part["value"]}
            if max_pixels is not None:
                img_msg["max_pixels"] = max_pixels
            processed_messages.append(img_msg)
        elif part["type"] == "video":
            video_msg = {
                "type": "video",
                "video": part["value"],
            }
            # Forward valid video kwargs
            for key in ["fps", "nframes", "total_pixels", "max_pixels", "min_pixels", "max_frames"]:
                if key in part and part[key] is not None and part[key] > 0:
                    video_msg[key] = part[key]
            # Default fps if neither fps nor nframes set
            if "fps" not in video_msg and "nframes" not in video_msg:
                video_msg["fps"] = 4
            processed_messages.append(video_msg)
    return processed_messages

class Cosmos(BaseModel):
    """Cosmos via vLLM with Configurable Resolution."""
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(self, model_path="nvidia/Cosmos-Reason1-7B", **kwargs):
        import os
        home = os.path.expanduser("~")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(home, ".cache", "torchinductor")
        os.environ["TRITON_CACHE_DIR"] = os.path.join(home, ".cache", "triton")
        from vllm import LLM, SamplingParams
        
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        os.environ['VLLM_SAMPLER_WARMUP_ITERATIONS'] = '1'
        self.use_vllm = kwargs.get("use_vllm", True)
        print(f"Using vLLM: {self.use_vllm}")
        self.model_path = model_path
        self.processor = AutoProcessor.from_pretrained(model_path)
        llm_kwargs = dict(
            model=model_path,
            limit_mm_per_prompt={"image": 64, "video": 10},
        )
        if "max_model_len" in kwargs:
            llm_kwargs["max_model_len"] = kwargs["max_model_len"]
        if "tensor_parallel_size" in kwargs:
            llm_kwargs["tensor_parallel_size"] = kwargs["tensor_parallel_size"]
        if "gpu_memory_utilization" in kwargs:
            llm_kwargs["gpu_memory_utilization"] = kwargs["gpu_memory_utilization"]
        self.llm = LLM(**llm_kwargs)
        self.enable_thinking = kwargs.get("enable_thinking", True)
        temperature = kwargs.get("temperature", 0.6)
        sp_kwargs = dict(
            temperature=temperature,
            max_tokens=kwargs.get("max_new_tokens", 16384),
            repetition_penalty=kwargs.get("repetition_penalty", 1.05),
        )
        if temperature > 0:
            sp_kwargs["top_p"] = kwargs.get("top_p", 0.95)
        self.sampling_params = SamplingParams(**sp_kwargs)
        self.max_pixels = kwargs.get("max_pixels", None)

    def message_for_promptimg(self, message):
        return _message_for_promptimg(message, max_pixels=self.max_pixels)

    def generate_inner(self, message, dataset=None):
        content = self.message_for_promptimg(message)

        messages = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        # Use return_video_metadata=True for newer vLLM that requires video metadata
        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True, return_video_metadata=True
            )
        except TypeError:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True
            )


        mm_data = {}
        if image_inputs: mm_data["image"] = image_inputs
        if video_inputs: mm_data["video"] = video_inputs

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs if video_kwargs else {},
        }
        outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params)
        text = outputs[0].outputs[0].text
        if self.enable_thinking:
            import re
            match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
            if match:
                return match.group(1).strip()
            # Strip <think>...</think> block if no <answer> tag
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        return text


COT_SYSTEM_PROMPT = (
    "Answer the question in the following format: "
    "<think>\nyour reasoning\n</think>\n\n<answer>\nyour answer\n</answer>."
)

class CosmosHF(BaseModel):
    """Cosmos (Reason1/Reason2) via HuggingFace transformers. No vLLM required."""
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(self, model_path="nvidia/Cosmos-Reason2-8B", max_new_tokens=4096,
                 temperature=0.0, top_p=0.95, repetition_penalty=1.05,
                 system_prompt=COT_SYSTEM_PROMPT, **kwargs):
        from transformers import AutoModelForImageTextToText
        self.model_path = model_path
        self.system_prompt = system_prompt
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        self.model.eval()
        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            repetition_penalty=repetition_penalty,
        )
        if temperature > 0:
            self.generate_kwargs["temperature"] = temperature
            self.generate_kwargs["top_p"] = top_p

    def message_for_promptimg(self, message):
        return _message_for_promptimg(message)

    def _extract_answer(self, response):
        import re
        match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: strip think block and return remainder
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        return response.strip()

    def _use_cot(self, dataset):
        if dataset is None or self.system_prompt is None:
            return False
        dtype = DATASET_TYPE(dataset, default=None)
        # MCQ and Y/N benchmarks score by exact string match — CoT breaks them
        return dtype not in ('MCQ', 'Y/N', 'BCQ')

    def generate_inner(self, message, dataset=None):
        messages = []
        if self._use_cot(dataset):
            messages.append({"role": "system", "content": self.system_prompt})
        user_message = {"role": "user", "content": self.message_for_promptimg(message)}
        messages.append(user_message)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        try:
            images, videos, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
        except TypeError:
            images, videos, video_kwargs = process_vision_info(
                messages, image_patch_size=16, return_video_kwargs=True
            )

        video_metadatas = None
        if videos is not None and videos and not isinstance(videos[0], tuple):
            pass
        elif videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)

        proc_kw = dict(
            text=text,
            images=images,
            videos=videos,
            do_resize=False,
            return_tensors="pt",
            **(video_kwargs or {}),
        )
        if video_metadatas is not None:
            proc_kw["video_metadata"] = video_metadatas
        inputs = self.processor(**proc_kw)
        inputs = inputs.to(self.model.device)
        if hasattr(self.model, "dtype"):
            inputs = inputs.to(self.model.dtype)

        generated_ids = self.model.generate(**inputs, **self.generate_kwargs)
        generated_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        if self._use_cot(dataset):
            return self._extract_answer(response)
        return response
