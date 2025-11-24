import os as _early_os
# Also import standard os for later usages in this file
import os
import datetime as _dt
# Disable parallelism in HuggingFace tokenizers to avoid fork-related warnings/deadlocks
_early_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Enable faster failure and better logs for NCCL collectives
_early_os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
# Remove deprecated var in this process to avoid warnings if present
if "NCCL_ASYNC_ERROR_HANDLING" in _early_os.environ:
    _early_os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
_early_os.environ.setdefault("NCCL_DEBUG", "WARN")  # change to INFO for deeper debugging
_early_os.environ.setdefault("TORCH_NCCL_TRACE_BUFFER_SIZE", "1048576")  # enable flight recorder
import shutil
from functools import partial
import torch
from new.monet_qwen_model import apply_qwen2_5_monet
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig, AutoTokenizer, AutoProcessor
from PIL import Image
import logging
from tqdm import tqdm
from trl import SFTTrainer, SFTConfig
from qwen_vl_utils import process_vision_info
import torch.distributed as dist
from src.utils import *
from src.task import *
from src.trainer import *
import random
import wandb
import re
from glob import glob

args=get_args()
assert args.save_model_path != "./checkpoints/", "You must specify the save path of the latent embeddings"
config = Qwen2_5_VLConfig.from_pretrained(args.load_model_path)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.load_model_path,
    config=config,
    torch_dtype=torch.bfloat16,
)
processor = AutoProcessor.from_pretrained(args.load_model_path, use_fast=False)

preprocess_function = task_preporcess_config[args.task]
all_train_dataset = []
for data_path in args.data_path:
    if data_path.endswith('.jsonl'):
        train_dataset = load_jsonl_dataset(data_path)
    elif data_path.endswith('.json'):
        train_dataset = load_json_dataset(data_path)
    all_train_dataset.extend(train_dataset[:])
if args.shuffle_train:
    random.seed(42)
    random.shuffle(all_train_dataset)

train_dataset = []
cur_max = -1
for i, sample in tqdm(enumerate(all_train_dataset[:]), desc="Collecting training data and length check...", total=len(all_train_dataset)):
    processed = preprocess_function(sample, dataset_root=args.dataset_root)
    if processed is not None:
        train_dataset.append(processed)


# ================= Prepare tokenizer special ids and misc =================
processor.tokenizer.add_tokens("<abs_vis_token_pad>", special_tokens=True)
processor.tokenizer.add_tokens("<abs_vis_token>", special_tokens=True)
processor.tokenizer.add_tokens("</abs_vis_token>", special_tokens=True)
processor.tokenizer.add_tokens("<observation>", special_tokens=True)
processor.tokenizer.add_tokens("</observation>", special_tokens=True)

latent_start_idx = processor.tokenizer("<abs_vis_token>", return_tensors="pt")["input_ids"][0]
latent_end_idx = processor.tokenizer("</abs_vis_token>", return_tensors="pt")["input_ids"][0]
latent_pad_idx = processor.tokenizer("<abs_vis_token_pad>", return_tensors="pt")["input_ids"][0]
observation_start_idx = processor.tokenizer("<observation>", return_tensors="pt")["input_ids"][0]
observation_end_idx = processor.tokenizer("</observation>", return_tensors="pt")["input_ids"][0]
end_pad_token_idx = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0]
answer_start_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]
img_start_idx = processor.tokenizer("<|vision_start|>", return_tensors="pt")["input_ids"][0]
img_end_idx = processor.tokenizer("<|vision_end|>", return_tensors="pt")["input_ids"][0]
img_pad_idx = processor.tokenizer("<|image_pad|>", return_tensors="pt")["input_ids"][0]

SPECIAL_id = {
    "v_start": img_start_idx,
    "v_end": img_end_idx,
    "img_pad": img_pad_idx,
    "abs_start": latent_start_idx,
    "abs_end": latent_end_idx,
    "abs_pad": latent_pad_idx,
    "obs_start": observation_start_idx,
    "obs_end": observation_end_idx,
}

# Resize embeddings to include newly added tokens if needed
try:
    new_vocab_size = len(processor.tokenizer)
    model.resize_token_embeddings(new_vocab_size)
    model.config.vocab_size = new_vocab_size
except Exception:
    pass

# Configure latent ids on model for downstream logic
model.config.latent_token_id = int(latent_pad_idx)
model.config.latent_start_id = int(latent_start_idx)
model.config.latent_end_id = int(latent_end_idx)
model.config.answer_start_pattern = answer_start_pattern.tolist()

# Freeze visual to match training behavior and eval-only run
for p in model.visual.parameters():
    p.requires_grad = False

model.eval()
try:
    model.gradient_checkpointing_disable()
except Exception:
    pass

def collate_fn_precompute_teacher_latents(examples):
    batch = {}
    metadata = [ex['metadata'] for ex in examples]

    examples = [ex['data'] for ex in examples]
    batch_assistant_img_cnts = [sum(1 for step in examples[i][2]['content'] if step["type"] == "image") for i in range(len(examples))]
    texts = [processor.apply_chat_template(ex, tokenize=False) for ex in examples]

    # replace `<abs_vis_token></abs_vis_token>`` with `<|vision_start|><|image_pad|><|vision_end|>`` for each `<|im_start|>assistant`` content
    texts = [replace_latent_placeholder_with_img_pad(text) for text in texts]
    
    # add `<abs_vis_token><abs_vis_token_pad>...</abs_vis_token>` after each `<|vision_start|><|image_pad|><|vision_end|>` for each `<|im_start|>assistant` content
    texts = add_latent_pad_after_auxiliary_img(texts, args.latent_size, "<abs_vis_token_pad>")
    
    image_inputs, _ = process_vision_info(examples)
    if args.image_resize == "global":
        image_inputs, new_sizes = resize_by_token_budget(image_inputs, global_max_pixels=1000*28*28, per_img_max_pixels=500*28*28)
    elif args.image_resize == "clear_question_img":
        image_inputs, new_sizes = resize_diff(image_inputs)

    total_image_pads = 0    
    for txt in texts:
        total_image_pads += txt.count("<|vision_start|><|image_pad|>")
    assert total_image_pads == len(image_inputs)
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    if not args.not_use_4d:
        attn_mask_4d, batch['segs'] = build_4d_attn(
            input_ids=batch["input_ids"],
            pad_mask=batch["attention_mask"],
            token_ids=SPECIAL_id,
            not_mask_image=args.not_mask_image,
            mask_latent=args.mask_latent,
            observation_tokens_cannot_see_question_image=args.observation_tokens_cannot_see_question_image,
            observation_tokens_only_see_question_and_latent=args.observation_tokens_only_see_question_and_latent,
            latent_can_see_all_previous=args.latent_can_see_all_previous,
            return_type='bool',
            mask_question_image=args.mask_question_image
        )
        batch["attention_mask_4d"] = {"full_attention": attn_mask_4d }

    batch["metadata"] = metadata
    return batch

def _scan_existing_latents(save_dir: str) -> set[str]:
    """Scan existing latent_*.pt files and return a set of metadata_info strings.
    Expected filename pattern: latent_{metadata_info}.pt
    """
    if not os.path.isdir(save_dir):
        return set()
    done = set()
    for p in glob(os.path.join(save_dir, "latent_*.pt")):
        fname = os.path.basename(p)
        m = re.match(r"^latent_(.+)\.pt$", fname)
        if m:
            done.add(m.group(1))
    return done


def _expected_metadata_info(metadata: dict, args) -> str:
    """Build the expected metadata_info string for a sample metadata dict."""
    dataset_name = metadata["dataset_name"]
    sample_id = metadata["sample_id"]
    # Keep exactly the same prefix rule as in saving code
    if getattr(args, "output_latent_embeds", False):
        prefix = "last_layer"
    else:
        prefix = "all_layers"
    return f"{prefix}_{dataset_name}_{sample_id}"


def _filter_indices_by_resume(train_dataset: list, args) -> list[int]:
    """When --resume is on, drop indices that are already computed in save dir."""
    save_dir = args.save_model_path
    done = _scan_existing_latents(save_dir)
    # Build a deterministic list of indices to compute
    keep = []
    for idx, ex in enumerate(train_dataset):
        md = ex["metadata"]
        info = _expected_metadata_info(md, args)
        if info not in done:
            keep.append(idx)
    return keep



def _device() -> torch.device:
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
        try:
            torch.cuda.set_device(local_rank)
        except Exception:
            pass
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def main():
    # Initialize distributed if requested
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_dist = world_size > 1
    if is_dist and not (dist.is_available() and dist.is_initialized()):
        # Use a generous timeout to avoid false positives on large models
        dist.init_process_group(backend="nccl", timeout=_dt.timedelta(minutes=30))
    try:
        rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))

        device = _device()
        model.to(device)

        # Save dir for latents
        out_dir = args.save_model_path
        os.makedirs(out_dir, exist_ok=True)

        # Iterate data and precompute
        bs = max(1, int(getattr(args, 'bsz', 1)))
        total = len(train_dataset)
        
        # ===== Resume support: drop finished samples =====
        if getattr(args, "resume", False):
            # Each rank computes the same filtered index list to avoid desync
            indices_to_process = _filter_indices_by_resume(train_dataset, args)
            total = len(indices_to_process)
            if rank == 0:
                logging.info(f"[resume] filtered unfinished samples: {total} remain (out of {len(train_dataset)}).")
        else:
            indices_to_process = list(range(total))

        # Cross-rank consistency check on the filtered total
        if is_dist:
            try:
                t = torch.tensor([total], device=device)
                gathered = [torch.zeros_like(t) for _ in range(world_size)]
                dist.all_gather(gathered, t)
                totals = [int(x.item()) for x in gathered]
                if len(set(totals)) != 1:
                    if rank == 0:
                        logging.error(f"[precompute] filtered total mismatch across ranks: {totals}. Exiting.")
                    return
            except Exception as e:
                if rank == 0:
                    logging.warning(f"[precompute] filtered total all_gather check failed: {e}")

        if rank == 0:
            logging.info(f"[precompute] total_to_process={total}, batch_size={bs}, saving to {out_dir}; world_size={world_size}")

        # Shard by filtered indices
        if is_dist:
            per = (total + world_size - 1) // world_size
            start_idx = rank * per
            end_idx = min(total, (rank + 1) * per)
            shard = indices_to_process[start_idx:end_idx]
        else:
            shard = indices_to_process

        if len(shard) == 0:
            logging.info(f"[precompute] rank {rank}: nothing to do (already finished).")
            return

        # Avoid early barriers that can deadlock if any rank errors; not required for independent precompute

        with torch.inference_mode():
            rng = range(0, len(shard), bs)
            pbar = tqdm(rng, desc=f"[rank {rank}] precompute", disable=False)
            for i in pbar:
                cur_ids = shard[i:i+bs]

                examples = [train_dataset[j] for j in cur_ids]
                batch = collate_fn_precompute_teacher_latents(examples)
                inputs = {
                    'latent_mode': True,
                    'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device),
                    'pixel_values': batch['pixel_values'].to(device),
                    'image_grid_thw': batch['image_grid_thw'].to(device),
                    'labels': None,
                    'loss_type': [],
                }
                if args.output_latent_embeds: # output latent embeddings only for last layer
                    inputs['output_latent_embeds'] = True
                if args.output_hidden_states: # output hidden states for all layers (can also output hidden states of all layers for latents)
                    inputs['output_hidden_states'] = True

                outputs = model(**inputs, return_dict=True)

                if args.output_latent_embeds:
                    teacher_reps = outputs.latent_embeds
                elif args.output_hidden_states:
                    teacher_reps = outputs.hidden_states
                # Save per global sample index to avoid collisions
                B = len(teacher_reps)
                for b in range(B):
                    metadata = batch['metadata'][b]
                    dataset_name = metadata['dataset_name']
                    sample_id = metadata['sample_id']
                    if args.output_latent_embeds:
                        metadata_info = f"last_layer_{dataset_name}_{sample_id}"
                    elif args.output_hidden_states:
                        metadata_info = f"all_layers_{dataset_name}_{sample_id}"
                    save_path = os.path.join(out_dir, f"latent_{metadata_info}.pt")
                    torch.save({'metadata_info': metadata_info, 'latent': teacher_reps[b].detach().cpu()}, save_path)


        # Avoid a final barrier; log completion per-rank to prevent deadlocks if any rank terminated early
        logging.info(f"[precompute] rank {rank} done. Latents saved under: {out_dir}")
    finally:
        if is_dist and dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception as e:
                logging.warning(f"[precompute] destroy_process_group failed: {e}")


if __name__ == "__main__":
    main()