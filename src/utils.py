import torch
import logging
import os
import numpy as np
import json
import random
import argparse
from datasets import Dataset
from typing import List, Union
import math
from PIL import Image
import math

def get_args():
    parser = argparse.ArgumentParser()
    # ===== Basic arguments =====
    parser.add_argument("--load_model_path", type=str, default='./checkpoints/model_stage1')
    parser.add_argument("--data_path", type=str, default='PathToJsonlData', nargs='+')
    parser.add_argument("--stage", type=str, default="avt_stage1", choices=['avt_sft', 'avt_stage1', 'avt_v2_stage1', 'avt_v2_precompute_latent', 'avt_v2_stage2'])
    parser.add_argument("--task", type=str, default="vsp-spatial-reasoning", choices=["vsp-spatial-reasoning", "vsp-spatial-planning", "blink-jigsaw", "sat", "mm-reasoning"])
    parser.add_argument("--save_model_path", type=str, default='./checkpoints/',help="Path to save the model checkpoints.")
    parser.add_argument("--resume_from_checkpoint", default=False, action="store_true")
    parser.add_argument("--dataset_root", type=str, default="./new", help="Root directory for the dataset.")
    parser.add_argument("--deepspeed", type=str, default="",
                        help="Path to DeepSpeed config JSON, e.g., ./deepspeed/ds_zero2_cpu_offload.json")
    parser.add_argument("--num_samples", default=-1, help="-1 means all data", type=int)

    # ===== Basic training hyperparameters =====
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for training.")
    parser.add_argument("--bsz", type=int, default=1, help="Batch size for training.")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--epochs", type=int, default=10)  
    parser.add_argument("--shuffle_train", action='store_true', default=False, help="Whether to shuffle the training dataset.")

    # ===== AVT SFT & AVT stage1 arguments =====
    parser.add_argument("--ce_emphasize_factor", default=1.0, type=float)
    parser.add_argument("--ce_emphasize_warmup_steps", default=50, type=int)

    # ===== AVT stage1 arguments =====
    parser.add_argument("--latent_size", type=int, default=4)
    parser.add_argument("--min_latent_size", type=int, default=6, help="AVT minimum latent size")
    parser.add_argument("--min_latent_compress_factor", type=int, default=10, help="the minimum of the range of the AVT compress factor")
    parser.add_argument("--max_latent_compress_factor", type=int, default=20, help="the maximum of the range of the AVT compress factor")
    parser.add_argument("--alignment_weight", default=1.0, help="Weight of the alignment loss in avt_stage1.")
    parser.add_argument("--alignment", type=str, default="observation_all", choices=["observation_end", "boxed_start", "observation_all"], help="The alignment strategy for AVT.")

    # ===== Training record arguments =====
    parser.add_argument("--log_file", type=str, default='./log.txt')
    parser.add_argument("--wandb_name", default=None, help="Name for the Weights & Biases run. If None, no W&B logging is done.")

    # ===== SFT representation analysis related arguments =====
    parser.add_argument("--sft_analysis_enable", action='store_true', default=False,
                        help="Enable tracking cosine similarity between baseline (pre-SFT) and current hidden states on a sampled subset.")
    parser.add_argument("--sft_analysis_ratio", type=float, default=0.02,
                        help="Proportion (0~1] of the whole training dataset to sample for representation analysis. Ignored if disable.")
    parser.add_argument("--sft_analysis_seed", type=int, default=1234,
                        help="Random seed for sampling analysis subset.")
    parser.add_argument("--sft_analysis_max_samples", type=int, default=200,
                        help="Maximum number of samples to track even if ratio yields more.")
    parser.add_argument("--sft_analysis_save_dir", type=str, default="./sft_analysis",
                        help="Directory to save analysis artifacts (subset ids, per-epoch cosine stats).")
    parser.add_argument("--sft_analysis_categories", type=str, nargs='+', default=["boxed_start_poss","observation_poss"],
                        help="Token position categories to aggregate: boxed_start_poss, observation_poss, non_observation_poss.")
    # ===== Eval SFT =====
    parser.add_argument("--mask_latent", action='store_true', default=False,
                        help="If set, make latent tokens (A_i) invisible to all subsequent tokens in build_additive_bias.")
    parser.add_argument("--eval_on_teacher_sequence", action='store_true', default=False)
    parser.add_argument("--eval_on_observation_tokens", action='store_true', default=False)
    parser.add_argument("--observation_tokens_only_see_image_tokens", action='store_true', default=False)
    parser.add_argument("--observation_tokens_only_see_latent_tokens", action='store_true', default=False)
    # ===== Precomputed teacher latent loading =====
    parser.add_argument("--teacher_latent_dir", type=str, default=None,
                        help="Directory that stores precomputed teacher latents (files named latent_{sample_id:08d}.pt). If not set, defaults to {save_model_path or ./checkpoints}/teacher_latents.")
    # DeepSpeed config path (optional). If provided, Trainer will enable DeepSpeed with this config.
    # ===== PPL analysis =====
    parser.add_argument("--no_question_image", action='store_true', default=False)

    return parser.parse_args()

def seed_everything(seed: int = 42):
    """
    Set seed for reproducibility across random, numpy, torch, and environment.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_jsonl_dataset(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
        data = data[:]
    return Dataset.from_list(data)

def load_json_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def place_input_image(text, image_pad="<|vision_start|><|image_pad|><|vision_end|>", image_placeholder="<image>", sep_token="<|im_start|>assistant") -> str:

    assert sep_token in text

    t1, t2 = text.split(sep_token)

    if image_placeholder in t1:
        t1 = t1.replace(image_pad, '')
        t1 = t1.replace(image_placeholder, image_pad)

    return t1 + sep_token + t2

def place_output_image(text, image_pad="<|vision_start|><|image_pad|><|vision_end|>", latent_placeholder="<output_image>", sep_token="<|im_start|>assistant") -> str:

    if latent_placeholder in text:
        text = text.replace(image_pad+'<think>', '<think>')
        text = text.replace(latent_placeholder, image_pad)

    return text

def place_output_image_avt(text, image_pad="<|vision_start|><|image_pad|><|vision_end|>", latent_placeholder="<abs_vis_token></abs_vis_token>", sep_token="<|im_start|>assistant") -> str:
    text = text.split(sep_token)
    res_text = text[0]
    assistant_texts = text[1:]
    for text in assistant_texts:
        if latent_placeholder in text:
            text = text.replace(image_pad, "")
            text = text.replace(latent_placeholder, image_pad)
        res_text += sep_token + text
    return res_text


def remove_user_images(examples):
    new_examples = []
    for example in examples:
        # `example` is a list of turn dicts
        new_example = []
        for turn in example:
            # Create a shallow copy of the turn so we don't modify the original
            new_turn = dict(turn)
            if turn.get("role") == "user":
                # Filter out image-type content
                new_turn["content"] = [
                    item for item in turn.get("content", [])
                    if item.get("type") != "image"
                ]
            # Add the updated turn to this new example
            new_example.append(new_turn)
        new_examples.append(new_example)

    return new_examples

def remove_assistant_images(examples):
    new_examples = []
    for example in examples:
        # `example` is a list of turn dicts
        new_example = []
        for turn in example:
            # Create a shallow copy of the turn so we don't modify the original
            new_turn = dict(turn)
            if turn.get("role") == "assistant":
                # Filter out image-type content
                new_turn["content"] = [
                    item for item in turn.get("content", [])
                    if item.get("type") != "image"
                ]
            # Add the updated turn to this new example
            new_example.append(new_turn)
        new_examples.append(new_example)

    return new_examples

def replace_visual_spectial_tokens(texts):

    update_texts = []
    for i, text in enumerate(texts):
        prev, after = text.split("<|im_start|>assistant")
        update_texts.append(prev + "<|im_start|>assistant" + after.replace("<|vision_start|><|image_pad|><|vision_end|>", "<|latent_start|><|image_pad|><|latent_end|>"))
        
    return update_texts

def replace_visual_spectial_tokens_avt(texts, latent_size, latent_pad_str="<abs_vis_token_pad>"):
    update_texts = []
    latent_pad_strs = latent_pad_str*latent_size
    for i, text in enumerate(texts):
        turns = text.split("<|im_start|>assistant")
        upd_text = turns[0]
        for turn in turns[1:]:
            upd_text += "<|im_start|>assistant" + turn.replace("<|vision_start|><|image_pad|><|vision_end|>", f"<abs_vis_token>{latent_pad_strs}</abs_vis_token>")
        update_texts.append(upd_text)
    return update_texts

def add_abs_vis_token_after_helper_img(texts, latent_size, latent_pad_str="<abs_vis_token_pad>"):
    update_texts = []
    latent_pad_strs = latent_pad_str*latent_size
    for i, text in enumerate(texts):
        turns = text.split("<|im_start|>assistant")
        upd_text = turns[0]
        for turn in turns[1:]:
            upd_text += "<|im_start|>assistant" + turn.replace("<|vision_start|><|image_pad|><|vision_end|>", f"<|vision_start|><|image_pad|><|vision_end|><abs_vis_token>{latent_pad_strs}</abs_vis_token>")
        update_texts.append(upd_text)
    return update_texts

def replace_subsequent_image_parts_2d(
    input_ids: torch.Tensor,
    start_token: int,
    end_token: int,
    replacement_token: int,
    replacement_length: int,
    pad_token: int = 0
) -> torch.Tensor:
    """
    Applies `replace_subsequent_image_parts_1d` to each row of the 2D tensor [batch_size, seq_len],
    then pads all resulting rows to the same max length, returning a single 2D tensor
    of shape [batch_size, new_max_len].
    """
    batch_size, seq_len = input_ids.shape
    
    # Process each row individually, store the result in a list
    replaced_sequences = []
    max_len = 0
    
    for i in range(batch_size):
        seq_1d = input_ids[i]
        new_seq_1d = replace_subsequent_image_parts_1d(
            seq_1d,
            start_token=start_token,
            end_token=end_token,
            replacement_token=replacement_token,
            replacement_length=replacement_length
        )
        replaced_sequences.append(new_seq_1d)
        max_len = max(max_len, new_seq_1d.size(0))
    
    # Now pad all replaced sequences to 'max_len'
    # We'll create a new tensor on the same device, same dtype
    new_input_ids = input_ids.new_full((batch_size, max_len), fill_value=pad_token)
    
    for i, seq_1d in enumerate(replaced_sequences):
        length = seq_1d.size(0)
        new_input_ids[i, :length] = seq_1d
    
    return new_input_ids

def replace_subsequent_image_parts_1d(
    seq: torch.Tensor,
    start_token: int,
    end_token: int,
    replacement_token: int,
    replacement_length: Union[int, List[int]]
) -> torch.Tensor:
    """
    Process a single 1D sequence, replacing everything from start_token..end_token
    with replacement_length copies of replacement_token. 
    """
    # Find positions of start and end tokens
    start_positions = (seq == start_token).nonzero().squeeze(-1)
    end_positions   = (seq == end_token).nonzero().squeeze(-1)

    new_seq_pieces = []
    prev_end = 0
    
    if isinstance(replacement_length, List):
        assert len(start_positions) == len(end_positions) == len(replacement_length), "Inconsistent numbers of latent start/end positions and images."

    for i, (s_pos, e_pos) in enumerate(zip(start_positions, end_positions)):
        # Add everything before this image part as-is, BUT include start_token itself
        new_seq_pieces.append(seq[prev_end : s_pos + 1])
        
        # Replace the entire chunk [s_pos+1 .. e_pos) with N copies of replacement_token
        if isinstance(replacement_length, List):
            replacement_length_i = replacement_length[i]
        else:
            replacement_length_i = replacement_length
        replacement_span = torch.tensor(
            [replacement_token] * replacement_length_i, 
            dtype=seq.dtype, 
            device=seq.device
        )
        new_seq_pieces.append(replacement_span)

        # Move past the end_token
        prev_end = e_pos
    
    # Add whatever remains after the last image part
    if prev_end < len(seq):
        new_seq_pieces.append(seq[prev_end:])
    
    # Concatenate into a single 1D tensor
    new_seq = torch.cat(new_seq_pieces, dim=0)
    return new_seq

def process_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    start_token: int,
    end_token: int,
    replacement_token: int,
    min_latent_size: int,
    min_latent_compress_factor: int = 10,
    max_latent_compress_factor: int = 20,
    pad_token: int = 0,
    batch_assistant_img_token_lens: List[int] = None
):

    batch_size, seq_len = input_ids.shape

    # We'll accumulate the processed (variable-length) sequences here.
    processed_sequences = []

    if batch_assistant_img_token_lens is not None:
        batch_compress_ratio = [random.randint(min_latent_compress_factor, max_latent_compress_factor) for _ in range(batch_size)]
        batch_compressed_img_token_lens = []
        for b in range(batch_size):
            compressed_lengths = []
            for img_len in batch_assistant_img_token_lens[b]:
                compressed_length = img_len // batch_compress_ratio[b] if img_len // batch_compress_ratio[b] > min_latent_size else min_latent_size
                compressed_lengths.append(compressed_length)
            batch_compressed_img_token_lens.append(compressed_lengths)

    for b in range(batch_size):
        # Extract the unpadded tokens using the attention mask (real tokens only)
        real_tokens = input_ids[b][attention_mask[b] == 1]

        # Perform image-part replacement on this 1D sequence
        updated_seq = replace_subsequent_image_parts_1d(
            real_tokens,
            start_token=start_token,
            end_token=end_token,
            replacement_token=replacement_token,
            replacement_length=replacement_length if batch_assistant_img_token_lens is None else batch_compressed_img_token_lens[b]
        )

        processed_sequences.append(updated_seq)

    # Now we have a list of 1D tensors of different lengths.
    # We'll re-pad them so they can be stacked into a [batch_size, new_seq_len] tensor.
    new_max_len = max(seq.size(0) for seq in processed_sequences)

    # Create new tensors for input_ids and attention_mask.
    # We'll fill input_ids with the specified pad_token.
    new_input_ids = input_ids.new_full((batch_size, new_max_len), fill_value=int(pad_token))
    # For attention_mask, padded positions are 0 by definition
    new_attention_mask = input_ids.new_zeros((batch_size, new_max_len))

    # Copy each processed sequence back into these padded tensors.
    for b in range(batch_size):
        seq_len_b = processed_sequences[b].size(0)
        new_input_ids[b, :seq_len_b] = processed_sequences[b]
        new_attention_mask[b, :seq_len_b] = 1

    return new_input_ids, new_attention_mask



def replace_assistant_image_pad_with_latent_pad(
    input_ids: torch.Tensor,
    start_token_pattern: torch.Tensor,
    image_pad_token: int,
    latent_pad_token: int
):
    """
    Replace image_pad_token with latent_pad_token only after the first occurrence 
    of start_token_pattern in each row.
    """
    batch_size, seq_len = input_ids.shape
    
    for i in range(batch_size):
        row = input_ids[i]
        # Find first occurrence of the start_token_pattern
        start_idx = find_subsequence(row, start_token_pattern)
        
        if start_idx != -1:
            # Replace image_pad_token with latent_pad_token only after the pattern
            pattern_end = start_idx + start_token_pattern.size(0)
            mask = (row[pattern_end:] == image_pad_token)
            row[pattern_end:][mask] = latent_pad_token
            
        input_ids[i] = row
    
    return input_ids
    
 

def find_subsequence(row: torch.Tensor, pattern: Union[torch.Tensor, List[torch.Tensor]], start: int=0) -> int:

    seq_len = row.size(0)
    # Naive scan over all possible start positions
    if isinstance(pattern, torch.Tensor):
        max_pat_len = pattern.size(0)
    elif isinstance(pattern, list):
        max_pat_len = max(pat.size(0) for pat in pattern)

    for start_idx in range(start, seq_len - max_pat_len + 1):
        # Compare row[start_idx : start_idx + pat_len] to pattern
        if isinstance(pattern, torch.Tensor):
            pat_len = pattern.size(0)
            if torch.all(row[start_idx : start_idx + pat_len] == pattern):
                return start_idx
        elif isinstance(pattern, list):
            for pat in pattern:
                if isinstance(pat, torch.Tensor):
                    pat_len = pat.size(0)
                    if torch.all(row[start_idx : start_idx + pat_len] == pat):
                        return start_idx

    return -1


def find_ids_poss(input_ids: torch.Tensor, answer_start_token_pattern: torch.Tensor, ids_tensor_or_list: Union[torch.Tensor,List[torch.Tensor]]) -> List[List[int]]:
    batch_poss = []
    for i in range(input_ids.shape[0]):
        manipulation_result_poss = []
        start_idx = find_subsequence(input_ids[i], answer_start_token_pattern, 0)
        while start_idx != -1:
            start_idx = find_subsequence(input_ids[i], ids_tensor_or_list, start_idx+1)
            if start_idx != -1:
                manipulation_result_poss.append(start_idx)
        manipulation_result_poss = manipulation_result_poss[:] # remove the first '\\boxed{', which is from the direct answer without avt
        batch_poss.append(manipulation_result_poss)
    return batch_poss

        
def generate_labels_after_multi_token_start(
    input_ids: torch.Tensor,
    start_sequence: torch.Tensor,
    ignore_ids: List[int] = None
) -> torch.Tensor:
    """
    For each row in `input_ids`, find the *first* occurrence of `start_sequence`
    (a 1D tensor of multiple token IDs). Mask all tokens up to and including
    that entire sub-sequence (set them to -100), and also mask any padding tokens
    anywhere in the row. The remainder (tokens *after* the sub-sequence) are kept.

    Args:
      input_ids: 2D tensor [batch_size, seq_len].
      start_sequence: 1D tensor of shape [k], the multi-token "start" pattern.
      pad_token_id: which ID is used as padding (default=0).
    
    Returns:
      labels: a new 2D tensor [batch_size, seq_len], where tokens before (and
              including) the sub-sequence are -100, as well as any pad tokens,
              and tokens after the sub-sequence are kept as in `input_ids`.
    """
    batch_size, seq_len = input_ids.shape
    
    # Clone so we can modify in-place
    labels = input_ids.clone()
    
    for b in range(batch_size):
        row = labels[b]
        # Find first occurrence of the entire sub-sequence
        start_idx = find_subsequence(row, start_sequence)
        
        if start_idx == -1:
            # Sub-sequence not found -> mask everything
            logging.warning(f"Couldn't find the <|im_start|>assistant, all labels are -100")
            row[:] = -100
        else:
            # The sub-sequence length
            sub_len = start_sequence.size(0)
            end_of_subseq = start_idx + sub_len  # the position *after* the sub-sequence
            
            # Mask everything up to (and including) the sub-sequence
            row[:end_of_subseq] = -100
        
        for id in ignore_ids:
            # Mask specified tokens (<|endoftext|>, <|vision_start|>, <|image_pad|>, <|vision_end|>)
            row[row == id] = -100


    
    return labels


def generate_labels_after_latent_tokens(
    input_ids: torch.Tensor,
    start_sequence: torch.Tensor,
    pad_token_idx: int = 0,
) -> torch.Tensor:
    """
    For each row in `input_ids`, find the *first* occurrence of `start_sequence`
    (a 1D tensor of multiple token IDs). Mask all tokens up to and including
    that entire sub-sequence (set them to -100), and also mask any padding tokens
    anywhere in the row. The remainder (tokens *after* the sub-sequence) are kept.

    Args:
      input_ids: 2D tensor [batch_size, seq_len].
      start_sequence: 1D tensor of shape [k], the multi-token "start" pattern.
      pad_token_id: which ID is used as padding (default=0).
    
    Returns:
      labels: a new 2D tensor [batch_size, seq_len], where tokens before (and
              including) the sub-sequence are -100, as well as any pad tokens,
              and tokens after the sub-sequence are kept as in `input_ids`.
    """
    batch_size, seq_len = input_ids.shape
    
    # Clone so we can modify in-place
    labels = input_ids.clone()
    
    for b in range(batch_size):
        row = labels[b]
        # Find first occurrence of the entire sub-sequence
        start_idx = find_subsequence(row, start_sequence)
        
        if start_idx == -1:
            # Sub-sequence not found -> mask everything
            row[:] = -100
        else:
            row[:start_idx] = -100
        
        # Mask pad tokens
        row[row == pad_token_idx] = -100
    
    return labels

def mask_image_output_tokens(
    input_ids: torch.Tensor,
    image_start_token: int,
    image_token: int
) -> torch.Tensor:
    """
    Creates a mask of the same shape as `input_ids`, with 1's wherever we want to
    'mask out' <image_token> after the first <image_start_token> has appeared,
    and 0's everywhere else.

    Args:
      input_ids: shape [batch_size, seq_len]
      image_start_token: the token ID that marks the start of an image chunk
      image_token: the token ID for image tokens

    Returns:
      A mask (torch.Tensor of the same shape) containing 0/1:
        - 1 = this position should be masked
        - 0 = this position is kept
    """
    batch_size, seq_len = input_ids.shape
    mask = torch.zeros_like(input_ids)

    for i in range(batch_size):
        seq = input_ids[i]
        # Find first occurrence of image_start_token
        first_start_pos = -1
        for j in range(seq_len):
            if seq[j] == image_start_token:
                first_start_pos = j
                break
        
        if first_start_pos == -1:
            continue
        
        # For every position after the first <image_start_token>,
        # if the token is <image_token>, set mask = 1
        for k in range(first_start_pos + 1, seq_len):
            if seq[k] == image_token:
                mask[i, k] = 1

    return mask



def resize_by_token_budget(images,
                           global_max_pixels=480*3*28*28,
                           per_img_max_pixels=800*28*28,
                           divisor=28):
    """等比缩放，保证一条样本内所有图像像素和 ≤ global_max_pixels"""
    # 1) 统计原总像素
    
    total = sum(img.width * img.height for img in images)
    if total <= global_max_pixels:
        return images, None   # 大多数样本会直接返回

    # 2) 统一缩放系数
    ratio = math.sqrt(global_max_pixels / total)

    processed = []
    new_sizes = []
    for img in images:
        w, h = int(img.width * ratio), int(img.height * ratio)
        # 保证能被 28 整除（Qwen 的 patch 大小）
        w = max(divisor, (w // divisor) * divisor)
        h = max(divisor, (h // divisor) * divisor)

        # 3) 仍然超过单张上限时再单独缩放
        if w * h > per_img_max_pixels:
            r = math.sqrt(per_img_max_pixels / (w * h))
            w = max(divisor, int(w * r) // divisor * divisor)
            h = max(divisor, int(h * r) // divisor * divisor)

        processed.append(img.resize((w, h), Image.BICUBIC))
        new_sizes.append((w, h))
    return processed, new_sizes


def resize_by_token_budget_sample_wise(images_per_sample,
                                       global_max_pixels=480*3*28*28,
                                       per_img_max_pixels=800*28*28,
                                       divisor=28):
    """逐样本等比缩放，每个样本单独满足像素预算。

    参数:
      images_per_sample: List[List[PIL.Image]] 按样本分组的图像列表。
      global_max_pixels: 单个样本内所有图片像素和的上限（默认与批量版一致）。
      per_img_max_pixels: 单张图片像素上限。
      divisor: 宽高需能被该值整除（Qwen 使用 28）。

    返回:
      processed_per_sample: List[List[PIL.Image]] 与输入结构一致的已缩放图像。
      new_sizes_per_sample: List[List[Tuple[int,int]]] 对应的新尺寸；若样本无需缩放则给出按同样逻辑得到的尺寸（与原尺寸一致）。
    """
    processed_per_sample = []
    sizes_per_sample = []

    for imgs in images_per_sample:
        if len(imgs) == 0:
            processed_per_sample.append([])
            sizes_per_sample.append([])
            continue

        total = sum(img.width * img.height for img in imgs)
        # 不需要缩放
        if total <= global_max_pixels:
            processed = []
            new_sizes = []
            for img in imgs:
                # 仍需确保尺寸可被 divisor 整除，否则下游 patch 数可能不一致
                w = max(divisor, (img.width // divisor) * divisor)
                h = max(divisor, (img.height // divisor) * divisor)
                if w != img.width or h != img.height:
                    processed.append(img.resize((w, h), Image.BICUBIC))
                else:
                    processed.append(img)
                new_sizes.append((w, h))
            processed_per_sample.append(processed)
            sizes_per_sample.append(new_sizes)
            continue

        # 需要按样本统一缩放系数
        ratio = math.sqrt(global_max_pixels / total)
        processed = []
        new_sizes = []
        for img in imgs:
            w, h = int(img.width * ratio), int(img.height * ratio)
            w = max(divisor, (w // divisor) * divisor)
            h = max(divisor, (h // divisor) * divisor)

            # 仍然超过单张上限时再单独缩放
            if w * h > per_img_max_pixels:
                r = math.sqrt(per_img_max_pixels / (w * h))
                w = max(divisor, int(w * r) // divisor * divisor)
                h = max(divisor, int(h * r) // divisor * divisor)

            processed.append(img.resize((w, h), Image.BICUBIC))
            new_sizes.append((w, h))

        processed_per_sample.append(processed)
        sizes_per_sample.append(new_sizes)

    return processed_per_sample, sizes_per_sample



if __name__=="__main__":
    
    pass

# ================= SFT representation analysis helpers =================
class SFTRepAnalyzer:
    """Track cosine similarity between baseline and current hidden states for selected samples & token positions.

    Usage lifecycle:
      analyzer = SFTRepAnalyzer(save_dir, categories, save_baseline)
      subset_ids = analyzer.select_subset(total_size, ratio, max_samples, seed)
      (Before training) for each tracked sample run base model with output_hidden_states=True and call build_baseline(sample_id, hidden_states)
      (During training) call update(sample_id, hidden_states, pos_dict, epoch, global_step)
      (End epoch) call finalize_epoch(epoch)
    """
    def __init__(self,
                 save_dir: str,
                 categories: List[str],
                 save_baseline: bool = True,
                 dataset_names: str = "",
                 exp_name: str = ""):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.categories = categories
        self.save_baseline_flag = save_baseline
        self.dataset_names = dataset_names
        self.subset_ids: List[int] = []
        self.baseline: dict[int, torch.Tensor] = {}  # id -> (num_layers, seq, hidden)
        self.layer_count: int = 0
        self.per_epoch_records: dict[int, dict] = {}
        self.exp_name = exp_name
        self.exp_save_folder = os.path.join(self.save_dir, self.exp_name)
        os.makedirs(self.exp_save_folder, exist_ok=True)

    def set_subset(self, subset_ids):
        self.subset_ids = subset_ids

    def select_subset(self, total_size: int, ratio: float, max_samples: int, seed: int):
        import math, random, json
        rng = random.Random(seed)
        k = min(max_samples, max(1, int(math.ceil(total_size * ratio))))
        self.subset_ids = sorted(rng.sample(range(total_size), k))
        with open(os.path.join(self.exp_save_folder, 'subset_ids.json'), 'w') as f:
            json.dump(self.subset_ids, f)
        return self.subset_ids

    def is_tracked(self, sample_id: int) -> bool:
        return sample_id in self.subset_ids

    def build_baseline(self, sample_id: int, hidden_states: List[torch.Tensor], device=None):
        if sample_id not in self.subset_ids:
            return
        if hidden_states is None or len(hidden_states) == 0:
            logging.warning(f"[SFT Analysis] Empty hidden_states when building baseline for sample {sample_id}; skipping.")
            return
        try:
            stacked = torch.stack([h[0].detach().cpu() for h in hidden_states], dim=0)
            if device is not None:
                stacked = stacked.to(device)
            # (L,S,H)
        except Exception as e:
            logging.error(f"[SFT Analysis] Failed stacking baseline hidden states for sample {sample_id}: {e}")
            return
        self.layer_count = stacked.shape[0]
        if self.save_baseline_flag:
            torch.save(stacked, os.path.join(self.exp_save_folder, f'baseline_{sample_id}.pt'))
        self.baseline[sample_id] = stacked

    @staticmethod
    def _gather_positions(hidden_layers: torch.Tensor, positions: List[int]) -> torch.Tensor:
        if len(positions)==0:
            return torch.empty(hidden_layers.size(0), 0, hidden_layers.size(-1))
        pos_tensor = torch.tensor(positions, dtype=torch.long)
        return hidden_layers[:, pos_tensor, :]

    def update(self, sample_id: int, hidden_states: List[torch.Tensor], pos_dict: dict, epoch: int, global_step: int):
        if sample_id not in self.baseline:
            return
        current = torch.stack([h[0].detach().cpu() for h in hidden_states], dim=0)
        base = self.baseline[sample_id]
        record = { 'epoch': epoch, 'step': global_step, 'sample_id': sample_id }
        for cat in self.categories:
            poss = pos_dict.get(cat, [])
            if len(poss)==0:
                continue
            cur_sel = self._gather_positions(current, poss)
            base_sel = self._gather_positions(base, poss)
            if cur_sel.numel()==0:
                continue
            cur_norm = cur_sel / (cur_sel.norm(dim=-1, keepdim=True) + 1e-6)
            base_norm = base_sel / (base_sel.norm(dim=-1, keepdim=True) + 1e-6)
            cos = (cur_norm.to(base_norm.device) * base_norm).sum(dim=-1)  # (L,P)
            layer_mean = cos.mean(dim=-1)  # (L)
            overall_mean = cos.mean().item()
            record[f'{cat}_layer_mean'] = layer_mean.tolist()
            record[f'{cat}_overall_mean'] = overall_mean
        self.per_epoch_records.setdefault(epoch, {'samples': []})['samples'].append(record)

    def finalize_epoch(self, epoch: int):
        import json
        if epoch not in self.per_epoch_records:
            return
        ep_data = self.per_epoch_records[epoch]
        samples = ep_data['samples']
        summary = {}
        for cat in self.categories:
            layer_accumulator = []
            count = 0
            for rec in samples:
                key = f'{cat}_layer_mean'
                if key in rec:
                    if not layer_accumulator:
                        layer_accumulator = [0.0]*len(rec[key])
                    for i,v in enumerate(rec[key]):
                        layer_accumulator[i]+=v
                    count+=1
            if count>0:
                summary[f'{cat}_layer_mean_avg'] = [v/count for v in layer_accumulator]
        summary['num_samples_with_cat'] = {cat: sum(1 for rec in samples if f'{cat}_layer_mean' in rec) for cat in self.categories}
        out = {'epoch': epoch, 'summary': summary, 'samples': samples}
        out_path = os.path.join(self.exp_save_folder, f'epoch_{epoch}_rep_analysis{self.exp_name}.json')
        with open(out_path, 'w') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        logging.info(f"[SFT Analysis] Saved epoch {epoch} rep analysis to {out_path}; samples={len(samples)}")

    def save_state(self):
        torch.save({'subset_ids': self.subset_ids}, os.path.join(self.exp_save_folder, 'state.pt'))


def find_segments_1d(ids, token_ids):
    """
    ids: 1D LongTensor, shape [L]
    token_ids: dict with keys:
        'v_start', 'v_end', 'img_pad',
        'abs_start', 'abs_end', 'abs_pad',
        'obs_start', 'obs_end'
    Returns: list of tuples for each S_i:
        (I_idx: LongTensor, A_idx: LongTensor, O_idx: LongTensor)
        O_idx may be empty if no <observation>...</observation> in T_i
    """
    L = ids.numel()
    I_segments, A_segments, O_segments = [], [], []

    # Helper to collect indices between two tags (exclusive) that match 'wanted_id' (or all if None)
    def between(start_pos, end_pos, wanted_id=None):
        s = start_pos + 1
        e = end_pos
        if s >= e: 
            return torch.empty(0, dtype=torch.long, device=ids.device)
        if wanted_id is None:
            idx = torch.arange(s, e, device=ids.device)
        else:
            mask = (ids[s:e] == wanted_id)
            idx = torch.nonzero(mask, as_tuple=False).squeeze(-1) + s
        return idx

    # 1) Parse all I_i by pairing <|vision_start|> ... <|vision_end|>
    v_starts = torch.nonzero(ids == token_ids['v_start'], as_tuple=False).squeeze(-1)
    v_ends   = torch.nonzero(ids == token_ids['v_end'],   as_tuple=False).squeeze(-1)

    # Greedy pairing in order
    v_ptr, e_ptr = 0, 0
    Vs, Ve = [], []
    while v_ptr < v_starts.numel() and e_ptr < v_ends.numel():
        if v_starts[v_ptr] < v_ends[e_ptr]:
            Vs.append(v_starts[v_ptr].item()); Ve.append(v_ends[e_ptr].item())
            v_ptr += 1; e_ptr += 1
        else:
            e_ptr += 1
    Vs = Vs[1:] # Remove the question image
    Ve = Ve[1:] # Remove the question image

    # 2) Parse all A_i by pairing <abs_vis_token> ... </abs_vis_token>
    a_starts = torch.nonzero(ids == token_ids['abs_start'], as_tuple=False).squeeze(-1)
    a_ends   = torch.nonzero(ids == token_ids['abs_end'],   as_tuple=False).squeeze(-1)
    As, Ae = [], []
    a_ptr, b_ptr = 0, 0
    while a_ptr < a_starts.numel() and b_ptr < a_ends.numel():
        if a_starts[a_ptr] < a_ends[b_ptr]:
            As.append(a_starts[a_ptr].item()); Ae.append(a_ends[b_ptr].item())
            a_ptr += 1; b_ptr += 1
        else:
            b_ptr += 1

    # 3) For each (I_i, A_i) in order, find O_i within T_i
    S = []
    for i in range(min(len(Vs), len(As))):
        vs, ve = Vs[i], Ve[i]
        as_, ae = As[i], Ae[i]

        I_idx = between(vs, ve, wanted_id=token_ids['img_pad'])
        A_idx = between(as_, ae, wanted_id=token_ids['abs_pad'])

        # T_i is from ae to next vision_start (or end of sequence)
        t_end = Vs[i+1] if i + 1 < len(Vs) else L
        # Find all <observation>...</observation> fully inside T_i
        obs_starts = torch.nonzero((ids == token_ids['obs_start']) & (torch.arange(L, device=ids.device) >= ae) & (torch.arange(L, device=ids.device) < t_end), as_tuple=False).squeeze(-1)
        obs_ends   = torch.nonzero((ids == token_ids['obs_end'])   & (torch.arange(L, device=ids.device) >  ae) & (torch.arange(L, device=ids.device) <= t_end), as_tuple=False).squeeze(-1)

        # Pair obs tags in order
        O_all = []
        p, q = 0, 0
        while p < obs_starts.numel() and q < obs_ends.numel():
            if obs_starts[p] < obs_ends[q]:
                # tokens between the two tags (exclusive) belong to O_i
                O_idx = between(obs_starts[p].item(), obs_ends[q].item(), wanted_id=None)
                if O_idx.numel():
                    O_all.append(O_idx)
                p += 1; q += 1
            else:
                q += 1

        O_idx = torch.cat(O_all, dim=0) if len(O_all) else torch.empty(0, dtype=torch.long, device=ids.device)
        S.append((I_idx, A_idx, O_idx))

    return S


def build_additive_bias(input_ids, pad_mask, token_ids, large_neg=-1e5, mask_latent: bool = False, observation_tokens_only_see_image_tokens: bool = False, observation_tokens_only_see_latent_tokens: bool = False):
    """
    input_ids: LongTensor [B, L]
    pad_mask:  LongTensor/BoolTensor [B, L], 1/True for real tokens
    token_ids: dict of special token ids (see above)
    large_neg: float used as "negative infinity" added to logits

    Returns:
      attn_bias: FloatTensor [B, 1, L, L] with 0 for allowed and large_neg for blocked
                 This bias ALREADY includes causal mask and padding mask.
    """
    input_ids = input_ids.cpu()
    pad_mask = pad_mask.cpu()
    
    B, L = input_ids.shape
    device = input_ids.device
    dtype_bias = torch.float32  # keep in fp32 for numerical safety; model will cast internally

    # Base: causal visibility (lower-triangular including diagonal)
    causal = torch.tril(torch.ones((L, L), dtype=torch.bool, device=device))

    # Start from causal AND valid tokens (both query & key must be valid)
    valid = pad_mask.bool()
    allowed = causal.unsqueeze(0).clone()  # [1, L, L]
    allowed = allowed.repeat(B, 1, 1)      # [B, L, L]
    for b in range(B):
        allowed[b] &= valid[b].unsqueeze(0)  # mask keys
        allowed[b] &= valid[b].unsqueeze(1)  # mask queries

    # Apply per-segment constraints
    for b in range(B):
        segs = find_segments_1d(input_ids[b], token_ids)
        if not segs:
            continue

        Lb = input_ids.shape[1]
        all_idx = torch.arange(Lb, device=device)

        for (I_idx, A_idx, O_idx) in segs:
            if A_idx.numel():
                allowed[b][A_idx, :] = False

                # ① A 段能看见紧左侧的 I 段所有位置
                if I_idx.numel():
                    # 形状广播: (|A|,1) × (|I|,) -> (|A|, |I|)
                    allowed[b][A_idx.unsqueeze(1), I_idx] = True

                # ② A 段“前缀自可见”（当前位置及以前的全部 latent 位置，包括自己）
                n = A_idx.numel()
                ar = torch.arange(n, device=A_idx.device)
                # 下三角(含对角) : row i 看到 col j 当且仅当 j <= i
                tri = ar.unsqueeze(1) >= ar.unsqueeze(0)          # (n, n) bool

                # 将局部 (n×n) 三角可见图写回全局 [L×L]
                rows = A_idx.unsqueeze(1).expand(n, n)            # (n, n)
                cols = A_idx.unsqueeze(0).expand(n, n)            # (n, n)
                allowed[b][rows, cols] = tri

                # 2) Only A_i can see I_i (mask I_i keys for all non-A queries)
                if I_idx.numel():
                    # Build row index set of "not in A_i"
                    not_A = torch.ones(Lb, dtype=torch.bool, device=device)
                    not_A[A_idx] = False
                    not_A_idx = torch.nonzero(not_A, as_tuple=False).squeeze(-1)  # [R]
                    if not_A_idx.numel():
                        # Block (rows in not_A_idx, cols in I_idx)
                        allowed[b][not_A_idx[:, None], I_idx] = False

                # Optional: make A_i invisible to all subsequent tokens (as keys)
                if mask_latent:
                    # rows r are considered "subsequent" if any a in A_idx satisfies a < r
                    r_idx = torch.arange(Lb, device=device)
                    rows_to_block = (r_idx.unsqueeze(0) > A_idx.unsqueeze(1)).any(dim=0)  # [L]
                    if rows_to_block.any():
                        allowed[b][rows_to_block.nonzero(as_tuple=False).squeeze(-1)[:, None], A_idx] = False

            if O_idx.numel() and A_idx.numel():
                '''assert (observation_tokens_only_see_latent_tokens or observation_tokens_only_see_image_tokens) and not (observation_tokens_only_see_latent_tokens and observation_tokens_only_see_image_tokens), "At most one of observation_tokens_only_see_latent_tokens and observation_tokens_only_see_image_tokens can be True"'''
                if observation_tokens_only_see_image_tokens:
                    # 3) O_i only sees I_i
                    allowed[b][O_idx, :] = False
                    allowed[b][O_idx.unsqueeze(1), I_idx] = True
                if observation_tokens_only_see_latent_tokens:
                    # 3) O_i only sees A_i
                    allowed[b][O_idx, :] = False
                    # If mask_latent is enabled, O_i cannot see A_i either; otherwise allow.
                    if not mask_latent:
                        allowed[b][O_idx.unsqueeze(1), A_idx] = True

                # ② O seg can see itself
                n = O_idx.numel()
                ar = torch.arange(n, device=O_idx.device)
                # 下三角(含对角) : row i 看到 col j 当且仅当 j <= i
                tri = ar.unsqueeze(1) >= ar.unsqueeze(0)          # (n, n) bool

                # 将局部 (n×n) 三角可见图写回全局 [L×L]
                rows = O_idx.unsqueeze(1).expand(n, n)            # (n, n)
                cols = O_idx.unsqueeze(0).expand(n, n)            # (n, n)
                allowed[b][rows, cols] = tri

            if I_idx.numel():
                # Optional safety: restrict I_i queries to themselves only (identity)
                allowed[b][I_idx, :] = False
                allowed[b][I_idx, I_idx] = True

    return allowed.unsqueeze(1)
    # Convert to additive bias: 0 for allowed, large_neg for masked
    #attn_bias = torch.zeros((B, 1, L, L), dtype=dtype_bias, device=device)
    #mask_4d = (~allowed).unsqueeze(1)           # [B, 1, L, L]
    #attn_bias = attn_bias.masked_fill(mask_4d, large_neg)
    #return (attn_bias >= 0) #attn_bias

def find_segments_1d_wo_helper_images(ids, token_ids):
    """
    ids: 1D LongTensor, shape [L]
    token_ids: dict with keys:
        'v_start', 'v_end', 'img_pad',
        'abs_start', 'abs_end', 'abs_pad',
        'obs_start', 'obs_end'
    Returns: list of tuples for each S_i:
        (I_idx: LongTensor, A_idx: LongTensor, O_idx: LongTensor)
        O_idx may be empty if no <observation>...</observation> in T_i
    """
    L = ids.numel()
    I_segments, A_segments, O_segments = [], [], []

    # Helper to collect indices between two tags (exclusive) that match 'wanted_id' (or all if None)
    def between(start_pos, end_pos, wanted_id=None):
        s = start_pos + 1
        e = end_pos
        if s >= e: 
            return torch.empty(0, dtype=torch.long, device=ids.device)
        if wanted_id is None:
            idx = torch.arange(s, e, device=ids.device)
        else:
            mask = (ids[s:e] == wanted_id)
            idx = torch.nonzero(mask, as_tuple=False).squeeze(-1) + s
        return idx


    # 2) Parse all A_i by pairing <abs_vis_token> ... </abs_vis_token>
    a_starts = torch.nonzero(ids == token_ids['abs_start'], as_tuple=False).squeeze(-1)
    a_ends   = torch.nonzero(ids == token_ids['abs_end'],   as_tuple=False).squeeze(-1)
    As, Ae = [], []
    a_ptr, b_ptr = 0, 0
    while a_ptr < a_starts.numel() and b_ptr < a_ends.numel():
        if a_starts[a_ptr] < a_ends[b_ptr]:
            As.append(a_starts[a_ptr].item()); Ae.append(a_ends[b_ptr].item())
            a_ptr += 1; b_ptr += 1
        else:
            b_ptr += 1

    # 3) For each (I_i, A_i) in order, find O_i within T_i
    S = []
    for i in range(len(As)):
        as_, ae = As[i], Ae[i]

        A_idx = between(as_, ae, wanted_id=token_ids['abs_pad'])

        # T_i is from ae to next latent start (or end of sequence)
        t_end = As[i+1] if i + 1 < len(As) else L
        # Find all <observation>...</observation> fully inside T_i
        obs_starts = torch.nonzero((ids == token_ids['obs_start']) & (torch.arange(L, device=ids.device) >= ae) & (torch.arange(L, device=ids.device) < t_end), as_tuple=False).squeeze(-1)
        obs_ends   = torch.nonzero((ids == token_ids['obs_end'])   & (torch.arange(L, device=ids.device) >  ae) & (torch.arange(L, device=ids.device) <= t_end), as_tuple=False).squeeze(-1)

        # Pair obs tags in order
        O_all = []
        p, q = 0, 0
        while p < obs_starts.numel() and q < obs_ends.numel():
            if obs_starts[p] < obs_ends[q]:
                # tokens between the two tags (exclusive) belong to O_i
                O_idx = between(obs_starts[p].item(), obs_ends[q].item(), wanted_id=None)
                if O_idx.numel():
                    O_all.append(O_idx)
                p += 1; q += 1
            else:
                q += 1

        O_idx = torch.cat(O_all, dim=0) if len(O_all) else torch.empty(0, dtype=torch.long, device=ids.device)
        S.append((A_idx, O_idx))

    return S

def build_additive_bias_wo_helper_images(input_ids, pad_mask, token_ids, large_neg=-1e5, mask_latent: bool = False, observation_tokens_only_see_latent_tokens: bool=False):
    """
    input_ids: LongTensor [B, L]
    pad_mask:  LongTensor/BoolTensor [B, L], 1/True for real tokens
    token_ids: dict of special token ids (see above)
    large_neg: float used as "negative infinity" added to logits

    Returns:
      attn_bias: FloatTensor [B, 1, L, L] with 0 for allowed and large_neg for blocked
                 This bias ALREADY includes causal mask and padding mask.
    """
    input_ids = input_ids.cpu()
    pad_mask = pad_mask.cpu()
    
    B, L = input_ids.shape
    device = input_ids.device
    dtype_bias = torch.float32  # keep in fp32 for numerical safety; model will cast internally

    # Base: causal visibility (lower-triangular including diagonal)
    causal = torch.tril(torch.ones((L, L), dtype=torch.bool, device=device))

    # Start from causal AND valid tokens (both query & key must be valid)
    valid = pad_mask.bool()
    allowed = causal.unsqueeze(0).clone()  # [1, L, L]
    allowed = allowed.repeat(B, 1, 1)      # [B, L, L]
    for b in range(B):
        allowed[b] &= valid[b].unsqueeze(0)  # mask keys
        allowed[b] &= valid[b].unsqueeze(1)  # mask queries

    # Apply per-segment constraints
    for b in range(B):
        segs = find_segments_1d_wo_helper_images(input_ids[b], token_ids)
        if not segs:
            continue

        Lb = input_ids.shape[1]
        all_idx = torch.arange(Lb, device=device)

        for (A_idx, O_idx) in segs:
            if A_idx.numel():
                # Optional: make A_i invisible to all subsequent tokens (as keys)
                if mask_latent:
                    # rows r are considered "subsequent" if any a in A_idx satisfies a < r
                    r_idx = torch.arange(Lb, device=device)
                    rows_to_block = (r_idx.unsqueeze(0) >= A_idx.unsqueeze(1)).any(dim=0)  # [L]
                    if rows_to_block.any():
                        allowed[b][rows_to_block.nonzero(as_tuple=False).squeeze(-1)[:, None], A_idx] = False

            if O_idx.numel() and A_idx.numel() and observation_tokens_only_see_latent_tokens:
                # 3) O_i only sees A_i
                allowed[b][O_idx, :] = False
                # If mask_latent is enabled, O_i cannot see A_i either; otherwise allow.
                if not mask_latent:
                    allowed[b][O_idx.unsqueeze(1), A_idx] = True
    return allowed.unsqueeze(1)
    # Convert to additive bias: 0 for allowed, large_neg for masked
    #attn_bias = torch.zeros((B, 1, L, L), dtype=dtype_bias, device=device)
    #mask_4d = (~allowed).unsqueeze(1)           # [B, 1, L, L]
    #attn_bias = attn_bias.masked_fill(mask_4d, large_neg)
    #return (attn_bias >= 0) #attn_bias

