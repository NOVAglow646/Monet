# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from collections import defaultdict
from io import BytesIO
import pdb
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF
import random
import base64, io, re
from PIL import Image
import glob

def dataset_name_from_path(data_path: str) -> str:
    if "math3k" in data_path or "geometry3k" in data_path:
        return "Geometry3K"
    elif "math" in data_path:
        return "Math3K"
    elif "Thyme-RL" in data_path and 'val' not in data_path:
        return "Thyme-train"
    elif "Thyme-RL" in data_path and 'val' in data_path:
        return "Thyme-val"
    else:
        raise NotImplementedError(f"Dataset {data_path} not supported yet.")

def b64_to_pil(s: str) -> Image.Image:
    """Decode a Base64 (optionally data-URL) image string to a PIL Image lazily.

    Note: Do not convert to RGB here to avoid forcing a full decode; conversion
    (and potential downscale) will be applied later in process_image.
    """
    # Remove optional data URL header and surrounding whitespace
    s = re.sub(r'^\s*data:image/[^;]+;base64,', '', s.strip(), flags=re.I)
    # Add missing padding if needed
    s += '=' * (-len(s) % 4)
    # Lazy open without convert to avoid immediate full decode
    return Image.open(io.BytesIO(base64.b64decode(s)))

def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)
    
    return {**tensors, **non_tensors}


class ImageProcessMixin:
    max_pixels: int
    min_pixels: int

    def process_image(self, image: Union[Dict[str, Any], ImageObject]) -> ImageObject:
        if isinstance(image, dict):
            image = Image.open(BytesIO(image["bytes"]))
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))

        if (image.width * image.height) > self.max_pixels:
            resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            # Hint decoder to downsample at decode time for JPEG/WEBP to reduce memory
            fmt = (getattr(image, "format", None) or "").upper()
            if fmt in {"JPEG", "JPG", "WEBP"}:
                try:
                    image.draft("RGB", (width, height))
                except Exception:
                    pass
            image = image.resize((width, height))

        if (image.width * image.height) < self.min_pixels:
            resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image


class RLHFDataset(Dataset, ImageProcessMixin):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        max_pixels: Optional[int] = None,
        min_pixels: Optional[int] = None,
        filter_overlong_and_invalid_prompts: bool = True,
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.filter_overlong_and_invalid_prompts = filter_overlong_and_invalid_prompts

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            # Ignore any validation/val subfolders and load all parquet files at the root level
            root_parquets = sorted(glob.glob(os.path.join(data_path, "*.parquet")))
            #pdb.set_trace()
            if root_parquets:
                # Treat all found shards as a single training split
                self.dataset = load_dataset("parquet", data_files=root_parquets, split="train")
            else:
                # Fallback: let datasets infer from data_dir (may require dataset_info.json)
                # Note: still using train split because we intentionally ignore validation here
                self.dataset = load_dataset("parquet", data_dir=data_path, split="train")
        elif os.path.isfile(data_path):
            # Respect the requested split when a single parquet file is given
            self.dataset = load_dataset("parquet", data_files=data_path, split=data_split)
        else:
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt_path = format_prompt
        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        self._valid_ids_cache_path = None
        if self.filter_overlong_and_invalid_prompts:
            # Build cache path per dataset name
            dataset_name = dataset_name_from_path(self.data_path)
            cache_dir = os.path.join("./examples/dataset_valid_ids", dataset_name)
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, "valid_ids.txt")
            self._valid_ids_cache_path = cache_file
            #pdb.set_trace()
            if os.path.exists(cache_file):
                # Fast path: reuse cached valid ids
                with open(cache_file, "r") as f:
                    valid_ids = [int(line.strip()) for line in f if line.strip()]
                # Safety check: make sure ids are within range
                max_id = max(valid_ids) if valid_ids else -1
                if max_id >= len(self.dataset):
                    raise ValueError(
                        f"Cached ids out of range: max_id={max_id}, dataset_len={len(self.dataset)}. "
                        "Make sure the dataset order/size matches the cache."
                    )
                self.dataset = self.dataset.select(valid_ids)
            else:
                # Slow path: run filter once and cache original indices of kept rows
                orig_idx_col = "__orig_idx__"
                # Remove stale helper column if present
                if orig_idx_col in self.dataset.column_names:
                    self.dataset = self.dataset.remove_columns([orig_idx_col])

                # Attach original row indices to keep track across filtering
                self.dataset = self.dataset.add_column(orig_idx_col, list(range(len(self.dataset))))

                # Run your predicate to filter; if you use num_proc, add it here
                filtered = self.dataset.filter(
                    self._filter_overlong_and_invalid_prompts,
                    desc="Filtering overlong prompts"
                    # , num_proc=...
                )

                # Extract kept original indices and persist to cache (atomic write)
                valid_ids = filtered[orig_idx_col]
                tmp_file = cache_file + ".tmp"
                with open(tmp_file, "w") as f:
                    for idx in valid_ids:
                        f.write(f"{int(idx)}\n")
                os.replace(tmp_file, cache_file)
                
                # Drop helper column and finalize
                self.dataset = filtered.remove_columns([orig_idx_col])

            print(f"Dataset size for training: {len(self.dataset)}")

    def _build_messages(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)
        #breakpoint()
        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            #breakpoint()
            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]

    def _filter_overlong_and_invalid_prompts(self, example: Dict[str, Any]) -> bool:
        if 'geometry3k' not in self.data_path:
            if 'Thyme-RL' in self.data_path:
                example = self._build_example(example, dataset_name="Thyme")
            else:
                raise NotImplementedError(f"Dataset {self.data_path} not supported yet.")
            if not example:
                return False
        messages = self._build_messages(example)
        processing_class = self.processor if self.processor is not None else self.tokenizer
        return (
            len(processing_class.apply_chat_template(messages, add_generation_prompt=True)) <= self.max_prompt_length
        )

    def __len__(self):
        return len(self.dataset)


    def _build_example(self, example, dataset_name):
        data = {}
        '''{'images': [<PIL.PngImagePlugin....EABCE3F50>], 'problem': '<image>Find $x$ so t... $m || n$.', 'answer': '63'}'''
        if dataset_name == "Thyme":
            if not example["images"] or len(example["images"]) > 1:
                return {}
            img = b64_to_pil(example["images"][0])
            data["images"] = [img]
            #img.save("/mmu_vcg_ssd/shiyang06-temp/Latent_Think/Easyr1-temp/debug_thyme_image.png")
            data["problem"] = "<image>" + example["question"]
            data["answer"] = example["solution"]
        return data

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        
        if 'geometry3k' not in self.data_path:
            if 'Thyme-RL' in self.data_path:
                example = self._build_example(example, dataset_name="Thyme")
            else:
                raise NotImplementedError(f"Dataset {self.data_path} not supported yet.")
        
        messages = self._build_messages(example)

        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = [self.process_image(image) for image in example.pop(self.image_key)]
            model_inputs = self.processor(images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"image": images}
            example["multi_modal_inputs"] = dict(model_inputs)
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
            # qwen2vl mrope
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw"),
                attention_mask=attention_mask,
            )  # (3, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["ground_truth"] = example.pop(self.answer_key)
        example["prompt_key"] = self.prompt_key
        example["answer_key"] = self.answer_key
        example["image_key"] = self.image_key
        example["prompt_before_processor"] = prompt
        example["global_index"] = index
        return example


class CorrectAnswerDataset(Dataset):
    def __init__(self, base_dataset, correct_pool):
        self.base_dataset = base_dataset            
        self.correct_pool = correct_pool            
        self.question_ids = list(correct_pool.keys())

    def __len__(self):
        return len(self.question_ids)

    def __getitem__(self, idx):
        qid = self.question_ids[idx]
        sample = self.base_dataset[qid]
        answer = random.choice(self.correct_pool[qid])
        sample["prev_correct_answer"] = answer
        return sample
