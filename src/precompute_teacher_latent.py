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
from new.avt_qwen_model import apply_qwen2_5_avt
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
    if 'avt' in args.stage:
        processed = preprocess_function(sample, dataset_root=args.dataset_root)
    else:
        processed = preprocess_function(sample)
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

def collate_fn_avt_v2_stage2(examples, alignment="boxed_start"):
    batch = {}
    batch['metadata'] = [ex['metadata'] for ex in examples]
    examples = [ex['data'] for ex in examples]
    batch_assistant_img_cnts = [sum(1 for step in examples[i][2]['content'] if step["type"] == "image") for i in range(len(examples))]
    texts = [processor.apply_chat_template(ex, tokenize=False) for ex in examples]
    # replace <abs_vis_token></abs_vis_token> with <|vision_start|><|image_pad|><|vision_end|> for each <|im_start|>assistant content
    texts = [place_output_image_avt(text) for text in texts]

    ################################################
    # teacher
    ################################################
    image_inputs, _ = process_vision_info(examples)
    image_inputs, new_sizes = resize_by_token_budget(image_inputs)
    teacher_texts = add_abs_vis_token_after_helper_img(texts, args.latent_size, "<abs_vis_token_pad>")
    teacher_batch = processor(text=teacher_texts, images=image_inputs, return_tensors="pt", padding=True)
    total_image_pads = 0
    for txt in texts:
        total_image_pads += txt.count("<|image_pad|>")
    assert total_image_pads == len(image_inputs)
    batch['teacher_pixel_values'] = teacher_batch['pixel_values']
    batch['teacher_image_grid_thw'] = teacher_batch['image_grid_thw']
    batch['teacher_input_ids'] = teacher_batch['input_ids']
    batch['teacher_attention_mask'] = teacher_batch['attention_mask']
    
    ################################################
    # student
    ################################################
    # replace <|vision_start|><|image_pad|><|vision_end|> with <abs_vis_token><abs_vis_token_pad>...</abs_vis_token> for each <|im_start|>assistant content
    student_texts = replace_visual_spectial_tokens_avt(texts, args.latent_size, "<abs_vis_token_pad>")
    user_examples = remove_assistant_images(examples)
    user_image_inputs, _ = process_vision_info(user_examples)
    resize_ptr = 0
    if new_sizes is not None:
        for i, img in enumerate(user_image_inputs):
            img = img.resize(new_sizes[resize_ptr], Image.BICUBIC)
            user_image_inputs[i] = img
            resize_ptr += batch_assistant_img_cnts[i] + 1 # user_image_inputs only contain question images of each batch sample, so we need to skip the helper images in the new_sizes by adding batch_assistant_img_cnts[i]
    student_batch = processor(text=student_texts, images=user_image_inputs, return_tensors="pt", padding=True)
    total_image_pads = 0
    for txt in student_texts:
        total_image_pads += txt.count("<|image_pad|>")
    assert total_image_pads == len(user_image_inputs)
    batch['student_pixel_values'] = student_batch['pixel_values']
    batch['student_image_grid_thw'] = student_batch['image_grid_thw']
    batch["student_input_ids"] = student_batch["input_ids"]
    batch["student_attention_mask"] = student_batch["attention_mask"]

    batch["student_alignment_poss"] = find_ids_poss(batch["student_input_ids"], answer_start_pattern, latent_pad_idx)
    batch["teacher_alignment_poss"] = find_ids_poss(batch["teacher_input_ids"], answer_start_pattern, latent_pad_idx)

    latent_start_poss = find_ids_poss(batch["student_input_ids"], answer_start_pattern, latent_start_idx)
    latent_end_poss = find_ids_poss(batch["student_input_ids"], answer_start_pattern, latent_end_idx)
    batch["ce_emphasize_poss"] = []
    for start_poss, end_poss in zip(latent_start_poss, latent_end_poss):
        poss_of_a_sample = []
        if len(start_poss) > 0 and len(end_poss) > 0:
            assert len(start_poss) == len(end_poss), f"start_poss: {start_poss}, end_poss: {end_poss}"
            poss_of_a_sample.extend(start_poss)
            poss_of_a_sample.extend(end_poss)
        batch["ce_emphasize_poss"].append(poss_of_a_sample)


    # mask tokens of '<|im_start|>assistant', '<|endoftext|>', and '<abs_vis_token_pad>' 
    batch["student_labels"] = generate_labels_after_multi_token_start(batch["student_input_ids"], answer_start_pattern, ignore_ids=[end_pad_token_idx, latent_pad_idx])


    return batch


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
        if is_dist:
            # Cross-rank consistency check for dataset length; mismatch is a common source of collective hangs
            try:
                t = torch.tensor([total], device=device)
                gathered = [torch.zeros_like(t) for _ in range(world_size)]
                dist.all_gather(gathered, t)
                totals = [int(x.item()) for x in gathered]
                if len(set(totals)) != 1:
                    if rank == 0:
                        logging.error(f"[precompute] dataset length mismatch across ranks: {totals}. "
                                      f"This can lead to deadlocks. Exiting.")
                    return
            except Exception as e:
                if rank == 0:
                    logging.warning(f"[precompute] total all_gather check failed: {e}")
        if rank == 0:
            logging.info(f"[precompute] total samples={total}, batch_size={bs}, saving to {out_dir}; world_size={world_size}")

        # Build index shards per rank
        indices = list(range(total))
        if is_dist:
            per = (total + world_size - 1) // world_size
            start_idx = rank * per
            end_idx = min(total, (rank + 1) * per)
            shard = indices[start_idx:end_idx]
        else:
            shard = indices

        # Avoid early barriers that can deadlock if any rank errors; not required for independent precompute

        with torch.inference_mode():
            rng = range(0, len(shard), bs)
            pbar = tqdm(rng, desc=f"[rank {rank}] precompute", disable=(rank != 0))
            for i in pbar:
                cur_ids = shard[i:i+bs]
                try:
                    examples = [train_dataset[j] for j in cur_ids]
                    batch = collate_fn_avt_v2_stage2(examples)
                    inputs = {
                        'stage': 'avt_v2_stage2',               # produce latent_embeds as trainer_v2_stage2 expects
                        'latent_mode': True,
                        'input_ids': batch['teacher_input_ids'].to(device),
                        'attention_mask': batch['teacher_attention_mask'].to(device),
                        'pixel_values': batch['teacher_pixel_values'].to(device),
                        'image_grid_thw': batch['teacher_image_grid_thw'].to(device),
                        'labels': None,
                        'alignment_poss': batch['teacher_alignment_poss'],
                        'loss_type': [],
                        'output_latent_embeds': True,
                    }
                    outputs = model(**inputs, return_dict=True)

                    latents = outputs.latent_embeds
                    # Save per global sample index to avoid collisions
                    B = len(latents)
                    for b in tqdm(range(B), desc="Saving latents", total=B):
                        metadata = batch['metadata'][b]
                        dataset_name = metadata['dataset_name']
                        sample_id = metadata['sample_id']
                        metadata_info = f"{dataset_name}_{sample_id}"
                        save_path = os.path.join(out_dir, f"latent_{metadata_info}.pt")
                        torch.save({'metadata_info': metadata_info, 'latent': latents[b].detach().cpu()}, save_path)
                except Exception as e:
                    logging.exception(f"[rank {rank}] Failed at batch start={i}, ids={cur_ids}: {e}")
                    # Continue processing other batches instead of crashing and hanging other ranks
                    continue

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