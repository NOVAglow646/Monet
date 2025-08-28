import os as _early_os
# Avoid tokenizer parallelism deadlocks in forked envs
_early_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import os
import json
import math
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import copy

import torch
from PIL import Image
from tqdm import tqdm

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLConfig,
    AutoProcessor,
)

from qwen_vl_utils import process_vision_info

# Project utils and task-specific preprocessing
from src.utils import get_args
from src.task import task_preporcess_config


def _device() -> torch.device:
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
        try:
            torch.cuda.set_device(local_rank)
        except Exception:
            pass
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


@dataclass
class SamplePack:
    # Core tensors for teacher sequence
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor

    # Text and meta
    text: str
    metadata: Dict[str, Any]

    # Per-image info aligned to image_pad occurrences
    image_roles: List[str]  # 'user' or 'assistant'
    image_paths: List[str]
    image_token_spans: List[Tuple[int, int]]  # [start_idx, end_idx] inclusive over input_ids
    messages: List[Dict[str, Any]]


def build_processor_and_model(args):
    config = Qwen2_5_VLConfig.from_pretrained(args.load_model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.load_model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        max_memory=({i: f"{int(torch.cuda.get_device_properties(i).total_memory // (1024**3) * 0.9)}GiB" for i in range(torch.cuda.device_count())}
                    if torch.cuda.is_available() else None),
    )
    processor = AutoProcessor.from_pretrained(args.load_model_path, use_fast=False)

    # Add special tokens used in AVT pipelines if missing
    for tok in [
        "<abs_vis_token_pad>", "<abs_vis_token>", "</abs_vis_token>",
        "<observation>", "</observation>",
    ]:
        try:
            processor.tokenizer.add_tokens(tok, special_tokens=True)
        except Exception:
            pass

    # Resize embeddings if tokenizer grew
    try:
        new_vocab_size = len(processor.tokenizer)
        model.resize_token_embeddings(new_vocab_size)
        model.config.vocab_size = new_vocab_size
    except Exception:
        pass

    # Cache special ids
    toks = processor.tokenizer
    special = {
        'v_start': toks("<|vision_start|>", return_tensors="pt")["input_ids"][0][0].item(),
        'v_end': toks("<|vision_end|>", return_tensors="pt")["input_ids"][0][0].item(),
        'img_pad': toks("<|image_pad|>", return_tensors="pt")["input_ids"][0][0].item(),
        'answer_start_pattern': toks("<|im_start|>assistant", return_tensors="pt")["input_ids"][0].tolist(),
    }

    # Eval-mode, freeze vision to match usual eval
    for p in model.visual.parameters():
        p.requires_grad = False
    model.eval()
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass

    return model, processor, special


def collect_dataset(args, preprocess_function, num_samples=-1) -> List[Dict[str, Any]]:
    all_data = []
    for p in args.data_path:
        if p.endswith('.jsonl'):
            from src.utils import load_jsonl_dataset
            dataset = load_jsonl_dataset(p)
            # datasets.Dataset -> list of dict
            try:
                dataset = list(dataset)
            except Exception:
                dataset = [x for x in dataset]
        elif p.endswith('.json'):
            from src.utils import load_json_dataset
            dataset = load_json_dataset(p)
        else:
            continue
        all_data.extend(dataset)

    # Preprocess to model-ready chat format with metadata
    processed = []
    total_num = len(all_data) if num_samples < 0 else min(num_samples, len(all_data))
    for sample in tqdm(all_data[:total_num], desc="preprocess", total=total_num):
        try:
            if 'avt' in args.stage:
                ex = preprocess_function(sample, dataset_root=args.dataset_root)
            else:
                ex = preprocess_function(sample)
        except Exception:
            ex = None
        if ex is not None:
            processed.append(ex)
    return processed


def extract_images_from_messages(messages: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """Return (paths, roles) in the same order as <|image_pad|> occurrences.
    This assumes each message is {role: 'user'|'assistant', 'content': [ ... ]}
    and each image item is {'type':'image', 'image': <path or PIL.Image>}.
    If an item is a PIL.Image, we store '<PIL>' as placeholder path.
    """
    paths: List[str] = []
    roles: List[str] = []
    for msg in messages:
        role = msg.get('role', 'user')
        for part in msg.get('content', []):
            if isinstance(part, dict) and part.get('type') == 'image':
                img = part.get('image', None)
                if isinstance(img, Image.Image):
                    paths.append('<PIL>')
                else:
                    # Could be str path or url
                    paths.append(str(img) if img is not None else '<UNKNOWN>')
                roles.append('assistant' if role == 'assistant' else 'user')
    return paths, roles


def find_image_token_spans(input_ids: torch.Tensor, v_start: int, v_end: int) -> List[Tuple[int, int]]:
    """Scan input_ids for contiguous (v_start ... v_end) spans; return [start,end] inclusive per image.
    We do not enforce there is an image_pad in the middle; we just use start/end delimiters.
    """
    ids = input_ids.tolist()
    spans: List[Tuple[int, int]] = []
    i = 0
    while i < len(ids):
        if ids[i] == v_start:
            # find next v_end
            j = i + 1
            while j < len(ids) and ids[j] != v_end:
                j += 1
            if j < len(ids) and ids[j] == v_end:
                spans.append((i, j))
                i = j + 1
                continue
        i += 1
    return spans


def pack_sample(example: Dict[str, Any], processor, special) -> SamplePack:
    messages = example['data']
    metadata = example.get('metadata', {})

    # Raw text via chat template
    text = processor.apply_chat_template(messages, tokenize=False)

    # Convert to model batch (teacher-style: include all images)
    image_inputs, _ = process_vision_info(messages)
    teacher_batch = processor(text=text, images=image_inputs, return_tensors="pt", padding=True)

    # Map images to roles/paths in the order they appear
    image_paths, image_roles = extract_images_from_messages(messages)
    # Sanity: lengths should match number of image_pad occurrences
    # We don't assert strictly to be tolerant; if mismatch, plotting will still work partially

    # Find token spans for each image (aligned to order)
    image_token_spans = find_image_token_spans(
        teacher_batch['input_ids'][0], special['v_start'], special['v_end']
    )

    return SamplePack(
        input_ids=teacher_batch['input_ids'],
        attention_mask=teacher_batch['attention_mask'],
        pixel_values=teacher_batch['pixel_values'],
        image_grid_thw=teacher_batch['image_grid_thw'],
        text=text,
        metadata=metadata,
        image_roles=image_roles,
        image_paths=image_paths,
        image_token_spans=image_token_spans,
    messages=messages,
    )


def softmax_prob_for_labels(logits: torch.Tensor, input_ids: torch.Tensor) -> List[float]:
    """Compute p(y_t | x_{<=t-1}) over the whole sequence for each token y_t = input_ids[t].
    Returns a python list of length seq_len with probs; for t=0 we set NaN.
    """
    # logits: [1, L, V]
    l = logits.float().squeeze(0)
    ids = input_ids.squeeze(0)
    L = ids.shape[0]
    probs: List[float] = [float('nan')] * L
    if L <= 1:
        return probs
    # Compute per-position selected logit then softmax row-wise
    # For numeric stability, subtract max before exp
    for t in range(1, L):
        row = l[t - 1]
        sel_id = int(ids[t].item())
        m = torch.max(row)
        num = torch.exp(row[sel_id] - m)
        den = torch.exp(row - m).sum()
        probs[t] = float((num / den).item())
    return probs


def remove_segment_from_sequence(seq: torch.Tensor, mask: torch.Tensor, seg: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Remove indices [s,e] inclusive from seq and its mask (1D tensors)."""
    s, e = seg
    keep = list(range(0, s)) + list(range(e + 1, seq.numel()))
    return seq[keep], mask[keep]


def remove_image_by_global_index(messages: List[Dict[str, Any]], global_img_index: int) -> List[Dict[str, Any]]:
    """Return a deep-copied messages list with the N-th image (in appearance order) removed."""
    msg_copy = copy.deepcopy(messages)
    count = 0
    for mi, msg in enumerate(msg_copy):
        new_content = []
        for part in msg.get('content', []):
            if isinstance(part, dict) and part.get('type') == 'image':
                if count == global_img_index:
                    # skip this image (remove)
                    count += 1
                    continue
                count += 1
            new_content.append(part)
        msg['content'] = new_content
    return msg_copy


def main():
    args = get_args()
    # Mode: if NO_QUESTION_IMAGE env is set truthy, we first remove the initial question image
    env_noq = os.environ.get('NO_QUESTION_IMAGE', '').lower()
    no_question_image = env_noq in ('1', 'true', 'yes', 'y') or bool(getattr(args, 'no_question_image', False))
    # Extra args via env or defaults
    out_jsonl = os.environ.get('COT_IMAGE_ABLATION_OUT', None)
    if out_jsonl is None:
        # Default under save_model_path/cot_image_ablation.jsonl
        base = args.save_model_path if args.save_model_path else './checkpoints'
        out_jsonl = os.path.join(base, 'cot_image_ablation.jsonl')

    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)

    model, processor, special = build_processor_and_model(args)
    device = _device()
    #model.to(device)

    preprocess_function = task_preporcess_config[args.task]
    data = collect_dataset(args, preprocess_function, args.num_samples)

    # Iterate samples one by one (bs=1) for clarity
    with open(out_jsonl, 'w', encoding='utf-8') as fout:
        for idx, ex in enumerate(tqdm(data, desc='analyze (per sample)')):
            pack = pack_sample(ex, processor, special)

            # Compute base sequence depending on mode
            if not no_question_image:
                # Base = original (all images)
                base_messages = pack.messages
                base_text = pack.text
                base_image_inputs, _ = process_vision_info(base_messages)
                base_batch = processor(text=base_text, images=base_image_inputs, return_tensors="pt", padding=True)
                base_ids = base_batch['input_ids'][0]
                base_mask = base_batch['attention_mask'][0]
                base_pixel = base_batch.get('pixel_values', None)
                base_grid = base_batch.get('image_grid_thw', None)
                base_roles = pack.image_roles
                base_paths = pack.image_paths
                # Probabilities with base
                with torch.inference_mode():
                    model_inputs = {
                        'stage': 'avt_v2_stage2',
                        'latent_mode': True,
                        'input_ids': base_ids.unsqueeze(0).to(device),
                        'attention_mask': base_mask.unsqueeze(0).to(device),
                        'labels': None,
                        'return_dict': True,
                    }
                    if base_pixel is not None:
                        model_inputs['pixel_values'] = base_pixel.to(device)
                    if base_grid is not None:
                        model_inputs['image_grid_thw'] = base_grid.to(device)
                    out_base = model(**model_inputs)
                probs_base = softmax_prob_for_labels(out_base.logits, base_ids.unsqueeze(0))
                base_image_spans = find_image_token_spans(base_ids, special['v_start'], special['v_end'])
                question_idx = None
            else:
                # Base = remove the first question image (first 'user' role)
                question_idx = None
                for ii, r in enumerate(pack.image_roles):
                    if r == 'user':
                        question_idx = ii
                        break
                if question_idx is None:
                    question_idx = 0  # fallback to the first image
                base_messages = remove_image_by_global_index(pack.messages, question_idx)
                base_text = processor.apply_chat_template(base_messages, tokenize=False)
                base_image_inputs, _ = process_vision_info(base_messages)
                base_batch = processor(text=base_text, images=base_image_inputs, return_tensors="pt", padding=True)
                base_ids = base_batch['input_ids'][0]
                base_mask = base_batch['attention_mask'][0]
                base_pixel = base_batch.get('pixel_values', None)
                base_grid = base_batch.get('image_grid_thw', None)
                base_paths, base_roles = extract_images_from_messages(base_messages)
                with torch.inference_mode():
                    model_inputs = {
                        'stage': 'avt_v2_stage2',
                        'latent_mode': True,
                        'input_ids': base_ids.unsqueeze(0).to(device),
                        'attention_mask': base_mask.unsqueeze(0).to(device),
                        'labels': None,
                        'return_dict': True,
                    }
                    if base_pixel is not None:
                        model_inputs['pixel_values'] = base_pixel.to(device)
                    if base_grid is not None:
                        model_inputs['image_grid_thw'] = base_grid.to(device)
                    out_base = model(**model_inputs)
                probs_base = softmax_prob_for_labels(out_base.logits, base_ids.unsqueeze(0))
                base_image_spans = find_image_token_spans(base_ids, special['v_start'], special['v_end'])

            # Determine assistant images in base sequence
            Nimg = len(base_image_spans)
            roles = base_roles
            paths = base_paths

            if len(roles) != Nimg:
                roles = roles[:Nimg] + ['unknown'] * max(0, Nimg - len(roles))
                paths = paths[:Nimg] + ['<UNKNOWN>'] * max(0, Nimg - len(paths))

            abl_indices = [i for i, r in enumerate(roles) if r == 'assistant']
            if not abl_indices and Nimg > 1 and not no_question_image:
                abl_indices = list(range(1, Nimg))

            if not abl_indices:
                # No assistant images; record minimal info and continue
                record = {
                    'index': idx,
                    'metadata': pack.metadata,
                    'cot_text': pack.text,
                    'token_ids': pack.input_ids.squeeze(0).tolist(),
                    'image_paths': paths,
                    'image_roles': roles,
                    'message': 'no_assistant_images',
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            # For each assistant image, ablate and compute delta after that span (relative to base)
            for j, img_idx in enumerate(abl_indices):
                seg = base_image_spans[img_idx]  # [s,e] in base ids
                s, e = seg

                # Build messages_removed: base minus the img_idx-th image (in base)
                msgs_removed = remove_image_by_global_index(base_messages, img_idx)
                text_removed = processor.apply_chat_template(msgs_removed, tokenize=False)
                image_inputs_removed, _ = process_vision_info(msgs_removed)
                batch_removed = processor(text=text_removed, images=image_inputs_removed, return_tensors="pt", padding=True)

                new_ids = batch_removed['input_ids'][0]
                new_mask = batch_removed['attention_mask'][0]
                pv_removed = batch_removed.get('pixel_values', None)
                grid_removed = batch_removed.get('image_grid_thw', None)

                with torch.inference_mode():
                    model_inputs = {
                        'stage': 'avt_v2_stage2',
                        'latent_mode': True,
                        'input_ids': new_ids.unsqueeze(0).to(device),
                        'attention_mask': new_mask.unsqueeze(0).to(device),
                        'labels': None,
                        'return_dict': True,
                    }
                    if pv_removed is not None:
                        model_inputs['pixel_values'] = pv_removed.to(device)
                    if grid_removed is not None:
                        model_inputs['image_grid_thw'] = grid_removed.to(device)
                    out_drop = model(**model_inputs)

                probs_drop = softmax_prob_for_labels(out_drop.logits, new_ids.unsqueeze(0))

                # Align positions after e in base sequence
                Lseg = e - s + 1
                ids_base = base_ids
                L = ids_base.numel()
                deltas = [None] * L
                probs_with = [None] * L
                probs_wo = [None] * L
                for t in range(e + 1, L):
                    t_mod = t - Lseg
                    if t_mod <= 0 or t_mod >= len(probs_drop):
                        continue
                    p_with = probs_base[t]
                    p_wo = probs_drop[t_mod]
                    if isinstance(p_with, float) and isinstance(p_wo, float) and not (math.isnan(p_with) or math.isnan(p_wo)):
                        deltas[t] = float(p_with - p_wo)
                        probs_with[t] = float(p_with)
                        probs_wo[t] = float(p_wo)

                record = {
                    'index': idx,
                    'metadata': pack.metadata,
                    'mode': 'no_question_image' if no_question_image else 'default',
                    'question_removed_original_index': int(question_idx) if no_question_image and question_idx is not None else None,
                    'cot_text': base_text if no_question_image else pack.text,
                    'token_ids': ids_base.tolist(),
                    'prob_with': probs_with,
                    'prob_without': probs_wo,
                    'delta': deltas,
                    'removed_image_global_index': int(img_idx),  # index within base sequence
                    'removed_image_role': roles[img_idx] if img_idx < len(roles) else 'unknown',
                    'removed_image_path': paths[img_idx] if img_idx < len(paths) else '<UNKNOWN>',
                    'removed_image_token_span': [int(s), int(e)],
                    'image_paths': paths,
                    'image_roles': roles,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    logging.info(f"Saved ablation results to {out_jsonl}")


if __name__ == "__main__":
    main()
