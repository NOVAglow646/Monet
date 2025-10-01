import os
# Disable parallelism in HuggingFace tokenizers to avoid fork-related warnings/deadlocks
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
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
from time import time
seed_everything(seed=42)
args=get_args()

# Optional: enable anomaly detection when debugging in-place grad issues
if os.environ.get("TORCH_ANOMALY", "0") == "1":
    try:
        torch.autograd.set_detect_anomaly(True)
        logging.info("Enabled torch.autograd anomaly detection (TORCH_ANOMALY=1)")
    except Exception:
        pass

# DDP-friendly logging: only rank0 writes file
_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
_handlers = [logging.StreamHandler()]
if _rank == 0 and getattr(args, 'log_file', None):
    _handlers.insert(0, logging.FileHandler(args.log_file, mode='a', encoding='utf-8'))
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=_handlers,
)

logging.info('=='*20)
logging.info(args)
logging.info('=='*20)

# Load the model and processor
cache_dir = '/home/dids/shiyang/datasets/cache_dir'
os.environ['HF_HOME'] = cache_dir

patch=14 # processor.image_processor.patch_size
# Use slow processor to avoid fast-processor info spam and behavioral drift
processor = AutoProcessor.from_pretrained(args.load_model_path, use_fast=False)

if _rank == 0:
    # Rewrite deprecated preprocessor.json into video_preprocessor.json by re-saving once
    try:
        processor.save_pretrained(args.load_model_path)
        if args.wandb_name is not None:
            wandb.init(project='Latent-Think',entity="Latent-Think",name=args.wandb_name,config={"ce_emphasize_factor":args.ce_emphasize_factor,"sft_analysis_ratio":args.sft_analysis_ratio})
    except Exception as _e:
        logging.debug(f"Processor save_pretrained skip: {_e}")

processor.tokenizer.add_tokens("<abs_vis_token_pad>", special_tokens=True)
processor.tokenizer.add_tokens("<abs_vis_token>", special_tokens=True)
processor.tokenizer.add_tokens("</abs_vis_token>", special_tokens=True)
processor.tokenizer.add_tokens("<observation>", special_tokens=True)
processor.tokenizer.add_tokens("</observation>", special_tokens=True)

config = Qwen2_5_VLConfig.from_pretrained(args.load_model_path)

config.stage = args.stage
# Avoid `use_cache=True` with gradient checkpointing warnings; training doesn't need cache
config.use_cache = False
# Some Qwen configs carry an unrecognized `loss_type=None` which triggers a warning; set explicitly
try:
    setattr(config, 'loss_type', 'ForCausalLMLoss')
except Exception:
    pass

if args.stage in ['stage1', 'avt_stage1'] or (args.stage == 'avt_sft' and args.sft_analysis_enable):
    config.output_hidden_states = True

# Prefer Trainer-managed device placement (DDP/Accelerate). Avoid device_map="auto" here.
# Enable TF32 for faster matmul on Ampere+ if available.
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.load_model_path,
    config=config,
    torch_dtype=torch.bfloat16,
)

# Prefer non-reentrant checkpointing to avoid requires_grad warnings (when enabled)
# Enable gradient checkpointing only on the LM/backbone (not on the frozen visual tower)
'''try:
    # Always start from a clean state
    model.gradient_checkpointing_disable()

    gc_kwargs = {"use_reentrant": False}

    # Only enable GC for stages that need it; keep it OFF for avt_v2_stage1/2
    if args.stage in ["avt_sft"]:
        enabled = False
        # Qwen2.5-VL commonly exposes the backbone under one of these attributes
        for attr in ["language_model", "transformer", "model"]:
            sub = getattr(model, attr, None)
            if sub is not None and hasattr(sub, "gradient_checkpointing_enable"):
                sub.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
                enabled = True
                break

        if not enabled:
            logging.warning("Could not locate LM backbone to enable gradient checkpointing; leaving it disabled.")

        # Make sure embeddings create grads for checkpointed layers
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        # HF requires use_cache=False when GC is on (you already set it above)
        model.config.use_cache = False
    else:
        logging.info(f"Gradient checkpointing disabled for stage: {args.stage}")
except Exception as _e:
    logging.debug(f"Selective gradient checkpointing setup skipped: {_e}")'''

if args.stage in ['stage1', 'avt_sft', 'avt_stage1']: 
    new_vocab_size = len(processor.tokenizer)
    model.resize_token_embeddings(new_vocab_size)
    model.config.vocab_size = new_vocab_size

tokenizer = processor.tokenizer


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
    "ans_start": answer_start_pattern
}

model.config.latent_token_id = int(latent_pad_idx)
model.config.latent_start_id = int(latent_start_idx)
model.config.latent_end_id = int(latent_end_idx)
model.config.answer_start_pattern = answer_start_pattern.tolist()

for param in model.visual.parameters():
    param.requires_grad = False


# If training avt_v4 with --no_ce, freeze token input embeddings and lm_head so only alignment-related parts learn
if args.stage == 'avt_v4' and getattr(args, 'no_ce', False):
    try:
        _emb = model.get_input_embeddings()
        _emb_params = sum(p.numel() for p in _emb.parameters()) if _emb is not None else 0
        for p in _emb.parameters():
            p.requires_grad = False
        _lm_params = sum(p.numel() for p in model.lm_head.parameters()) if hasattr(model, 'lm_head') else 0
        for p in model.lm_head.parameters():
            p.requires_grad = False
        logging.info(f"[freeze] avt_v4 + no_ce=True -> froze input_embeddings ({_emb_params} params) and lm_head ({_lm_params} params)")
    except Exception as _e:
        logging.warning(f"[freeze] Failed to freeze input embeddings / lm_head under avt_v4+no_ce: {_e}")


def collate_fn_stage1(examples):
    batch = {}
    batch['metadata'] = [ex['metadata'] for ex in examples]
    examples = [ex['data'] for ex in examples]

    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]

    texts = [place_input_image(text) for text in texts]
    texts = [place_output_image(text) for text in texts]
    texts = replace_visual_spectial_tokens(texts)

    image_inputs, _ = process_vision_info(examples)

    user_examples = remove_assistant_images(examples)
    user_text = [processor.apply_chat_template(example, tokenize=False) for example in user_examples]
    user_text = replace_visual_spectial_tokens(user_text)
    user_image_inputs, _ = process_vision_info(user_examples)
    user_batch = processor(text=user_text, images=user_image_inputs, return_tensors="pt", padding=True)

    assistant_examples = remove_user_images(examples)
    assistant_text = [processor.apply_chat_template(example, tokenize=False) for example in assistant_examples]
    assistant_text = replace_visual_spectial_tokens(assistant_text)
    assistant_image_inputs, _ = process_vision_info(assistant_examples)
    assistant_batch = processor(text=assistant_text, images=assistant_image_inputs, return_tensors="pt", padding=True)

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    batch['pixel_values'] = user_batch['pixel_values']
    batch['image_grid_thw'] = user_batch['image_grid_thw']

    batch['pixel_values_latent'] = assistant_batch['pixel_values']
    batch['image_grid_thw_latent'] = assistant_batch['image_grid_thw']

    latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0]
    latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0]
    latent_end_idx = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0]

    pad_token_idx = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0]

    new_input_ids, new_attention_mask = process_batch(batch["input_ids"], batch["attention_mask"], 
                                                      latent_start_idx, latent_end_idx, latent_token_idx, args.latent_size, pad_token_idx)

    batch["input_ids"] = new_input_ids
    batch["attention_mask"] = new_attention_mask

    answer_start_token_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]

    labels = generate_labels_after_multi_token_start(batch["input_ids"], answer_start_token_pattern, pad_token_idx, latent_token_idx)
    batch["labels"] = labels

    image_out_mask = mask_image_output_tokens(batch["input_ids"], latent_start_idx, latent_token_idx)
    batch["image_out_mask"] = image_out_mask

    return batch

def collate_fn_stage2(examples):
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    
    texts = [place_input_image(text) for text in texts]
    texts = [place_output_image(text) for text in texts]
    texts = replace_visual_spectial_tokens(texts)
    
    image_inputs, _ = process_vision_info(examples)

    user_examples = remove_assistant_images(examples)
    user_text = [processor.apply_chat_template(example, tokenize=False) for example in user_examples]
    user_text = replace_visual_spectial_tokens(user_text)
    user_image_inputs, _ = process_vision_info(user_examples)
    user_batch = processor(text=user_text, images=user_image_inputs, return_tensors="pt", padding=True)

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    
    batch['pixel_values'] = user_batch['pixel_values']
    batch['image_grid_thw'] = user_batch['image_grid_thw']

    latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0]
    latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0]
    latent_end_idx = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0]

    pad_token_idx = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0]

    new_input_ids, new_attention_mask = process_batch(batch["input_ids"], batch["attention_mask"], 
                                                      latent_start_idx, latent_end_idx, latent_token_idx, args.latent_size, pad_token_idx)

    batch["input_ids"] = new_input_ids
    batch["attention_mask"] = new_attention_mask

    answer_start_token_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]

    labels = generate_labels_after_multi_token_start(batch["input_ids"], answer_start_token_pattern, pad_token_idx, latent_token_idx)
    batch["labels"] = labels
    
    return batch

def collate_fn_avt_stage1(examples, alignment="boxed_start"):
    examples = examples['data']
    batch_assistant_img_cnts = [sum(1 for step in example[2]['content'] if step["type"] == "image") for example in examples]
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]

    # replace `<abs_vis_token></abs_vis_token>`` with `<|vision_start|><|image_pad|><|vision_end|>`` for each `<|im_start|>assistant`` content
    texts = [place_output_image_avt(text) for text in texts]
    
    # replace `<|vision_start|><|image_pad|><|vision_end|>`` with `<abs_vis_token><|image_pad|></abs_vis_token>`` for each `<|im_start|>assistant` content
    student_texts = replace_visual_spectial_tokens_avt(texts)

    image_inputs, _ = process_vision_info(examples)
    image_inputs, new_sizes = resize_by_token_budget(image_inputs)

    user_examples = remove_assistant_images(examples)
    user_text = [processor.apply_chat_template(example, tokenize=False) for example in user_examples]
    user_text = replace_visual_spectial_tokens_avt(user_text)
    user_image_inputs, _ = process_vision_info(user_examples)
    resize_ptr = 0
    if new_sizes is not None:
        for i, img in enumerate(user_image_inputs):
            img = img.resize(new_sizes[resize_ptr], Image.BICUBIC)
            user_image_inputs[i] = img
            resize_ptr += batch_assistant_img_cnts[i] + 1 # user_image_inputs only contain question images of each batch sample, so we need to skip the helper images in the new_sizes by adding batch_assistant_img_cnts[i]
    user_batch = processor(text=user_text, images=user_image_inputs, return_tensors="pt", padding=True)

    assistant_examples = remove_user_images(examples)
    assistant_text = [processor.apply_chat_template(example, tokenize=False) for example in assistant_examples]
    assistant_text = replace_visual_spectial_tokens_avt(assistant_text)
    assistant_image_inputs, _ = process_vision_info(assistant_examples)
    resize_ptr = 1
    batch_id_ptr = 0
    next_milestone = batch_assistant_img_cnts[batch_id_ptr] + 1
    if new_sizes is not None:
        for i, img in enumerate(assistant_image_inputs):
            img = img.resize(new_sizes[resize_ptr], Image.BICUBIC)
            assistant_image_inputs[i] = img
            if resize_ptr + 1 == next_milestone:
                resize_ptr += 2
                batch_id_ptr += 1
                if batch_id_ptr >= args.bsz:
                    break
                next_milestone += batch_assistant_img_cnts[batch_id_ptr] + 1
            else:
                resize_ptr += 1
                
    assistant_batch = processor(text=assistant_text, images=assistant_image_inputs, return_tensors="pt", padding=True)

    total_image_pads = 0
    for txt in student_texts:
        total_image_pads += txt.count("<|image_pad|>")
    assert total_image_pads == len(image_inputs)
    student_batch = processor(text=student_texts, images=image_inputs, return_tensors="pt", padding=True)
    teacher_batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    batch = {}

    batch['user_pixel_values'] = user_batch['pixel_values']
    batch['user_image_grid_thw'] = user_batch['image_grid_thw']

    batch['user_assistant_pixel_values'] = student_batch['pixel_values']
    batch['user_assistant_image_grid_thw'] = student_batch['image_grid_thw']

    batch_assistant_img_token_lens_merged = [(t[1]*t[2]).item()//4 for t in assistant_batch['image_grid_thw']]
    batch_assistant_img_token_lens = []
    start = 0
    for assistant_img_cnts in batch_assistant_img_cnts:
        batch_assistant_img_token_lens.append(batch_assistant_img_token_lens_merged[start:start+assistant_img_cnts])
        start += assistant_img_cnts

    # <|latent_start|><|image_pad|><|latent_end|> -> <|latent_start|><|latent_pad|>...<|latent_end|>; pad the sequences to the same length
    batch["student_input_ids"], batch["student_attention_mask"] = process_batch(student_batch["input_ids"], student_batch["attention_mask"], 
                                                      latent_start_idx, latent_end_idx, latent_pad_idx, args.min_latent_size, 
                                                      args.min_latent_compress_factor, args.max_latent_compress_factor,
                                                      end_pad_token_idx, batch_assistant_img_token_lens)
    
    #if (batch["student_input_ids"]==151655).sum()==1472:
    #    pass
        
    batch["teacher_input_ids"] = teacher_batch["input_ids"] #replace_assistant_image_pad_with_latent_pad(teacher_batch["input_ids"], answer_start_pattern, img_pad_idx, latent_pad_idx)
    batch["teacher_attention_mask"] = teacher_batch["attention_mask"]

    if alignment == "observation_end":
        alignment_pattern = observation_end_idx
        batch["student_alignment_poss"] = find_ids_poss(batch["student_input_ids"], answer_start_pattern, alignment_pattern)
        batch["teacher_alignment_poss"] = find_ids_poss(batch["teacher_input_ids"], answer_start_pattern, alignment_pattern)
    elif alignment == "boxed_start":
        alignment_pattern = [processor.tokenizer("\\boxed{", return_tensors="pt")["input_ids"][0], processor.tokenizer(" \\boxed{", return_tensors="pt")["input_ids"][0]]
        batch["student_alignment_poss"] = find_ids_poss(batch["student_input_ids"], answer_start_pattern, alignment_pattern)
        batch["teacher_alignment_poss"] = find_ids_poss(batch["teacher_input_ids"], answer_start_pattern, alignment_pattern)
    elif alignment == "observation_all":
        student_observation_start_poss = find_ids_poss(batch["student_input_ids"], answer_start_pattern, observation_start_idx)
        student_observation_end_poss = find_ids_poss(batch["student_input_ids"], answer_start_pattern, observation_end_idx)
        batch["student_alignment_poss"] = []
        assert len(student_observation_start_poss) == len(student_observation_end_poss)
        for start_poss, end_poss in zip(student_observation_start_poss, student_observation_end_poss):
            poss_of_a_sample = []
            if len(start_poss) > 0 and len(end_poss) > 0:
                assert len(start_poss) == len(end_poss), f"start_poss: {start_poss}, end_poss: {end_poss}"
                for start, end in zip(start_poss, end_poss):
                    poss_of_a_sample.extend(list(range(start, end + 1)))
            batch["student_alignment_poss"].append(poss_of_a_sample)
        teacher_observation_start_poss = find_ids_poss(batch["teacher_input_ids"], answer_start_pattern, observation_start_idx)
        teacher_observation_end_poss = find_ids_poss(batch["teacher_input_ids"], answer_start_pattern, observation_end_idx)
        batch["teacher_alignment_poss"] = []
        assert len(teacher_observation_start_poss) == len(teacher_observation_end_poss)
        for start_poss, end_poss in zip(teacher_observation_start_poss, teacher_observation_end_poss):
            poss_of_a_sample = []
            if len(start_poss) > 0 and len(end_poss) > 0:
                assert len(start_poss) == len(end_poss), f"start_poss: {start_poss}, end_poss: {end_poss}"
                for start, end in zip(start_poss, end_poss):
                    poss_of_a_sample.extend(list(range(start, end + 1)))
            batch["teacher_alignment_poss"].append(poss_of_a_sample)

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

    # We needn't compute the ce loss for the teacher 
    #batch["teacher_labels"] = generate_labels_after_multi_token_start(batch["teacher_input_ids"], answer_start_pattern, end_pad_token_idx, img_pad_idx)

    # return a mask where tokens of <|latent_pad|> are 1, else 0
    batch["student_image_out_mask"] = mask_image_output_tokens(batch["student_input_ids"], latent_start_idx, latent_pad_idx)

    # return a mask where tokens of <|image_pad|> are 1, else 0
    batch["teacher_image_out_mask"] = mask_image_output_tokens(batch["teacher_input_ids"], latent_start_idx, img_pad_idx)

    return batch

def collate_fn_avt_sft(examples):
    # examples: list of {conversation: [...], sample_id: int}
    batch = {}
    batch['metadata'] = [ex['metadata'] for ex in examples]
    examples = [ex['data'] for ex in examples]
    texts = [processor.apply_chat_template(ex, tokenize=False) for ex in examples]

    # replace <abs_vis_token></abs_vis_token> with <|vision_start|><|image_pad|><|vision_end|> for each <|im_start|>assistant content
    texts = [place_output_image_avt(text) for text in texts]
    
    ################################################
    # teacher
    ################################################
    image_inputs, _ = process_vision_info(examples)
    if args.image_resize == "global":
        image_inputs, new_sizes = resize_by_token_budget(image_inputs)
    elif args.image_resize == "clear_question_img":
        image_inputs, new_sizes = resize_diff(image_inputs) # resize_by_token_budget(image_inputs)
    teacher_texts = texts
    teacher_batch = processor(text=teacher_texts, images=image_inputs, return_tensors="pt", padding=True)
    total_image_pads = 0
    for txt in texts:
        total_image_pads += txt.count("<|image_pad|>")
    assert total_image_pads == len(image_inputs)
    batch['teacher_pixel_values'] = teacher_batch['pixel_values']
    batch['teacher_image_grid_thw'] = teacher_batch['image_grid_thw']
    batch['teacher_input_ids'] = teacher_batch['input_ids']
    batch['teacher_attention_mask'] = teacher_batch['attention_mask']

    observation_start_poss = find_ids_poss(batch["teacher_input_ids"], answer_start_pattern, observation_start_idx)
    observation_end_poss = find_ids_poss(batch["teacher_input_ids"], answer_start_pattern, observation_end_idx)
    batch["teacher_observation_poss"] = []
    assert len(observation_start_poss) == len(observation_end_poss)
    for start_poss, end_poss in zip(observation_start_poss, observation_end_poss):
        poss_of_a_sample = []
        if len(start_poss) > 0 and len(end_poss) > 0:
            assert len(start_poss) == len(end_poss), f"start_poss: {start_poss}, end_poss: {end_poss}"
            for start, end in zip(start_poss, end_poss):
                poss_of_a_sample.extend(list(range(start, end)))
        batch["teacher_observation_poss"].append(poss_of_a_sample)

    batch["teacher_labels"] = generate_labels_after_multi_token_start(batch["teacher_input_ids"], answer_start_pattern, ignore_ids=[end_pad_token_idx, img_pad_idx, img_start_idx, img_end_idx, observation_start_idx, observation_end_idx])
    return batch

def collate_fn_avt_v2_stage1(examples):
    if _rank == 0:
        start_time = time()
    batch = {}
    batch['metadata'] = [ex['metadata'] for ex in examples]
    examples = [ex['data'] for ex in examples]
    batch_assistant_img_cnts = [sum(1 for step in examples[i][2]['content'] if step["type"] == "image") for i in range(len(examples))]
    texts = [processor.apply_chat_template(ex, tokenize=False) for ex in examples]

    # replace `<abs_vis_token></abs_vis_token>`` with `<|vision_start|><|image_pad|><|vision_end|>`` for each `<|im_start|>assistant`` content
    texts = [place_output_image_avt(text) for text in texts]
    
    # add `<abs_vis_token><abs_vis_token_pad>...</abs_vis_token>` after each `<|vision_start|><|image_pad|><|vision_end|>` for each `<|im_start|>assistant` content
    texts = add_abs_vis_token_after_helper_img(texts, args.latent_size, "<abs_vis_token_pad>")
    
    image_inputs, _ = process_vision_info(examples)
    if args.image_resize == "global":
        image_inputs, new_sizes = resize_by_token_budget(image_inputs)
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

    
    observation_start_poss = find_ids_poss(batch["input_ids"], answer_start_pattern, observation_start_idx)
    observation_end_poss = find_ids_poss(batch["input_ids"], answer_start_pattern, observation_end_idx)
    batch["observation_poss"] = []
    assert len(observation_start_poss) == len(observation_end_poss)
    for start_poss, end_poss in zip(observation_start_poss, observation_end_poss):
        poss_of_a_sample = []
        if len(start_poss) > 0 and len(end_poss) > 0:
            assert len(start_poss) == len(end_poss), f"start_poss: {start_poss}, end_poss: {end_poss}"
            for start, end in zip(start_poss, end_poss):
                poss_of_a_sample.extend(list(range(start+1, end)))
        batch["observation_poss"].append(poss_of_a_sample)

    if args.only_predict_obs:
        batch["labels"] = generate_labels_after_multi_token_start_only_allow(batch["input_ids"], answer_start_pattern, allowed_poss=batch["observation_poss"])
    else:
        batch["labels"] = generate_labels_after_multi_token_start(batch["input_ids"], answer_start_pattern, ignore_ids=[end_pad_token_idx, 
        latent_pad_idx, latent_end_idx, img_pad_idx, img_start_idx, img_end_idx, observation_start_idx, observation_end_idx])

    return batch

def collate_fn_avt_v2_stage2(examples, alignment="boxed_start"):
    # Support wrapped examples providing sample_id
    batch = {}
    batch['metadata'] = [ex['metadata'] for ex in examples]
    examples = [ex['data'] for ex in examples]
    batch_assistant_img_cnts = [sum(1 for step in examples[i][2]['content'] if step["type"] == "image") for i in range(len(examples))]
    texts = [processor.apply_chat_template(ex, tokenize=False) for ex in examples]

    # replace <abs_vis_token></abs_vis_token> with <|vision_start|><|image_pad|><|vision_end|> for each <|im_start|>assistant content
    texts = [place_output_image_avt(text) for text in texts]
    image_inputs, _ = process_vision_info(examples)
    image_inputs, new_sizes = resize_by_token_budget(image_inputs, global_max_pixels=2000*28*28, per_img_max_pixels=2000*28*28,)
    
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

    if args.mask_latent:
        attn_mask_4d = build_4d_attn_wo_helper_images(
            input_ids=batch["student_input_ids"],
            pad_mask=batch["student_attention_mask"],
            token_ids=SPECIAL_id,
            mask_latent=getattr(args, 'mask_latent', False),
        )
        batch["student_attention_mask_4d"] = {"full_attention": attn_mask_4d }

    batch["student_alignment_poss"] = find_ids_poss(batch["student_input_ids"], answer_start_pattern, latent_pad_idx)

    observation_start_poss = find_ids_poss(batch["student_input_ids"], answer_start_pattern, observation_start_idx)
    observation_end_poss = find_ids_poss(batch["student_input_ids"], answer_start_pattern, observation_end_idx)
    batch["observation_poss"] = []
    assert len(observation_start_poss) == len(observation_end_poss)
    for start_poss, end_poss in zip(observation_start_poss, observation_end_poss):
        poss_of_a_sample = []
        if len(start_poss) > 0 and len(end_poss) > 0:
            assert len(start_poss) == len(end_poss), f"start_poss: {start_poss}, end_poss: {end_poss}"
            for start, end in zip(start_poss, end_poss):
                poss_of_a_sample.extend(list(range(start+1, end)))
        batch["observation_poss"].append(poss_of_a_sample)

    # mask tokens of '<|im_start|>assistant', '<|endoftext|>', and '<abs_vis_token_pad>' 
    batch["student_labels"] = generate_labels_after_multi_token_start(batch["student_input_ids"], answer_start_pattern, ignore_ids=[img_pad_idx, img_start_idx, img_end_idx, end_pad_token_idx, latent_pad_idx, latent_end_idx, observation_start_idx, observation_end_idx])

    return batch

def collate_fn_avt_v3(examples, alignment="boxed_start"):
    # Support wrapped examples providing sample_id
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
    if args.image_resize == "global":
        image_inputs, new_sizes = resize_by_token_budget(image_inputs)
    elif args.image_resize == "clear_question_img":
        image_inputs, new_sizes = resize_diff(image_inputs) # resize_by_token_budget(image_inputs)
    teacher_texts = texts
    teacher_batch = processor(text=teacher_texts, images=image_inputs, return_tensors="pt", padding=True)
    total_image_pads = 0
    for txt in texts:
        total_image_pads += txt.count("<|image_pad|>")
    assert total_image_pads == len(image_inputs)
    batch['teacher_pixel_values'] = teacher_batch['pixel_values']
    batch['teacher_image_grid_thw'] = teacher_batch['image_grid_thw']
    batch['teacher_input_ids'] = teacher_batch['input_ids']
    batch['teacher_attention_mask'] = teacher_batch['attention_mask']
    
    batch['teacher_segs'] = []
    for b in range(teacher_batch['input_ids'].shape[0]):
        q_img_idx, segs = find_helper_img_segs(teacher_batch['input_ids'][b], SPECIAL_id)
        batch['teacher_segs'].append(segs)
    ################################################
    # student
    ################################################
    # replace <|vision_start|><|image_pad|><|vision_end|> with <abs_vis_token><abs_vis_token_pad>...</abs_vis_token> for each <|im_start|>assistant content
    student_texts = replace_visual_spectial_tokens_avt(texts, args.latent_size, "<abs_vis_token_pad>")
    user_examples = remove_assistant_images(examples)
    user_image_inputs, _ = process_vision_info(user_examples)
    resize_ptr = 0
    if args.image_resize == "global":
        if new_sizes is not None:
            for i, img in enumerate(user_image_inputs):
                img = img.resize(new_sizes[resize_ptr], Image.BICUBIC)
                user_image_inputs[i] = img
                resize_ptr += batch_assistant_img_cnts[i] + 1
    elif args.image_resize == "clear_question_img":
        for i, img in enumerate(user_image_inputs):
            if new_sizes[resize_ptr] is not None:
                img = img.resize(new_sizes[resize_ptr], Image.BICUBIC)
            user_image_inputs[i] = img
            resize_ptr += batch_assistant_img_cnts[i] + 1 # user_image_inputs only contain question images of each batch sample, so we need to skip the helper images in the new_sizes by adding batch_assistant_img_cnts[i]
    else:
        raise NotImplementedError
        

    student_batch = processor(text=student_texts, images=user_image_inputs, return_tensors="pt", padding=True)
    total_image_pads = 0
    for txt in student_texts:
        total_image_pads += txt.count("<|image_pad|>")
    assert total_image_pads == len(user_image_inputs)
    batch['student_pixel_values'] = student_batch['pixel_values']
    batch['student_image_grid_thw'] = student_batch['image_grid_thw']
    batch["student_input_ids"] = student_batch["input_ids"]
    batch["student_attention_mask"] = student_batch["attention_mask"]

    if args.mask_latent:
        attn_mask_4d = build_4d_attn_wo_helper_images(
            input_ids=batch["student_input_ids"],
            pad_mask=batch["student_attention_mask"],
            token_ids=SPECIAL_id,
            mask_latent=getattr(args, 'mask_latent', False),
        )
        batch["student_attention_mask_4d"] = {"full_attention": attn_mask_4d }

    batch["student_alignment_poss"] = find_ids_poss(batch["student_input_ids"], answer_start_pattern, latent_pad_idx)
    batch["teacher_alignment_poss"] = find_ids_poss(batch["teacher_input_ids"], answer_start_pattern, latent_pad_idx)

    '''latent_start_poss = find_ids_poss(batch["student_input_ids"], answer_start_pattern, latent_start_idx)
    latent_end_poss = find_ids_poss(batch["student_input_ids"], answer_start_pattern, latent_end_idx)
    batch["ce_emphasize_poss"] = []
    for start_poss, end_poss in zip(latent_start_poss, latent_end_poss):
        poss_of_a_sample = []
        if len(start_poss) > 0 and len(end_poss) > 0:
            assert len(start_poss) == len(end_poss), f"start_poss: {start_poss}, end_poss: {end_poss}"
            poss_of_a_sample.extend(start_poss)
            poss_of_a_sample.extend(end_poss)
        batch["ce_emphasize_poss"].append(poss_of_a_sample)'''


    observation_start_poss = find_ids_poss(batch["student_input_ids"], answer_start_pattern, observation_start_idx)
    observation_end_poss = find_ids_poss(batch["student_input_ids"], answer_start_pattern, observation_end_idx)
    batch["observation_poss"] = []
    assert len(observation_start_poss) == len(observation_end_poss)
    for start_poss, end_poss in zip(observation_start_poss, observation_end_poss):
        poss_of_a_sample = []
        if len(start_poss) > 0 and len(end_poss) > 0:
            assert len(start_poss) == len(end_poss), f"start_poss: {start_poss}, end_poss: {end_poss}"
            for start, end in zip(start_poss, end_poss):
                poss_of_a_sample.extend(list(range(start+1, end)))
        batch["observation_poss"].append(poss_of_a_sample)

    # mask tokens of '<|im_start|>assistant', '<|endoftext|>', and '<abs_vis_token_pad>' 
    batch["student_labels"] = generate_labels_after_multi_token_start(batch["student_input_ids"], answer_start_pattern, ignore_ids=[img_pad_idx, img_start_idx, img_end_idx, end_pad_token_idx, latent_pad_idx, observation_start_idx, observation_end_idx, latent_end_idx])

    return batch

def collate_fn_avt_v4(examples, alignment="boxed_start"):
    # Support wrapped examples providing sample_id
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
    new_sizes = None
    if args.teacher_reps_dir is None:
        image_inputs, _ = process_vision_info(examples)
        if args.image_resize == "global":
            image_inputs, new_sizes = resize_by_token_budget(image_inputs,global_max_pixels=1280*28*28, per_img_max_pixels=1000*28*28)
        elif args.image_resize == "clear_question_img":
            image_inputs, new_sizes = resize_diff(image_inputs) # resize_by_token_budget(image_inputs)
        teacher_texts = texts
        teacher_batch = processor(text=teacher_texts, images=image_inputs, return_tensors="pt", padding=True)
        total_image_pads = 0
        for txt in texts:
            total_image_pads += txt.count("<|image_pad|>")
        assert total_image_pads == len(image_inputs)
        batch['teacher_pixel_values'] = teacher_batch['pixel_values']
        batch['teacher_image_grid_thw'] = teacher_batch['image_grid_thw']
        batch['teacher_input_ids'] = teacher_batch['input_ids']
        batch['teacher_attention_mask'] = teacher_batch['attention_mask']

        batch["teacher_labels"] = generate_labels_after_multi_token_start(batch["teacher_input_ids"], answer_start_pattern, ignore_ids=[img_pad_idx, img_start_idx, img_end_idx, end_pad_token_idx, observation_start_idx, observation_end_idx])

    ################################################
    # student
    ################################################
    # replace <|vision_start|><|image_pad|><|vision_end|> with <abs_vis_token><abs_vis_token_pad>...</abs_vis_token> for each <|im_start|>assistant content
    student_texts = replace_visual_spectial_tokens_avt(texts, args.latent_size, "<abs_vis_token_pad>")
    user_examples = remove_assistant_images(examples)
    user_image_inputs, _ = process_vision_info(user_examples)
    resize_ptr = 0

    if new_sizes is not None:
        if args.image_resize == "global":
            if new_sizes is not None:
                for i, img in enumerate(user_image_inputs):
                    img = img.resize(new_sizes[resize_ptr], Image.BICUBIC)
                    user_image_inputs[i] = img
                    resize_ptr += batch_assistant_img_cnts[i] + 1
        elif args.image_resize == "clear_question_img":
            for i, img in enumerate(user_image_inputs):
                if new_sizes[resize_ptr] is not None:
                    img = img.resize(new_sizes[resize_ptr], Image.BICUBIC)
                user_image_inputs[i] = img
                resize_ptr += batch_assistant_img_cnts[i] + 1 # user_image_inputs only contain question images of each batch sample, so we need to skip the helper images in the new_sizes by adding batch_assistant_img_cnts[i]
        else:
            raise NotImplementedError
    else:
        if args.image_resize == "global":
            user_image_inputs, new_sizes = resize_by_token_budget(user_image_inputs,global_max_pixels=1000*28*28, per_img_max_pixels=1000*28*28)
        elif args.image_resize == "clear_question_img":
            user_image_inputs, new_sizes = resize_diff(user_image_inputs) # resize_by_token_budget
        

    student_batch = processor(text=student_texts, images=user_image_inputs, return_tensors="pt", padding=True)
    total_image_pads = 0
    for txt in student_texts:
        total_image_pads += txt.count("<|image_pad|>")
    assert total_image_pads == len(user_image_inputs)
    batch['student_pixel_values'] = student_batch['pixel_values']
    batch['student_image_grid_thw'] = student_batch['image_grid_thw']
    batch["student_input_ids"] = student_batch["input_ids"]
    batch["student_attention_mask"] = student_batch["attention_mask"]

    if args.mask_latent:
        attn_mask_4d = build_4d_attn_wo_helper_images(
            input_ids=batch["student_input_ids"],
            pad_mask=batch["student_attention_mask"],
            token_ids=SPECIAL_id,
            mask_latent=getattr(args, 'mask_latent', False),
        )
        batch["student_attention_mask_4d"] = {"full_attention": attn_mask_4d }

    observation_start_poss = find_ids_poss(batch["teacher_input_ids"], answer_start_pattern, observation_start_idx)
    observation_end_poss = find_ids_poss(batch["teacher_input_ids"], answer_start_pattern, observation_end_idx)
    batch["teacher_observation_poss"] = []
    assert len(observation_start_poss) == len(observation_end_poss)
    for start_poss, end_poss in zip(observation_start_poss, observation_end_poss):
        poss_of_a_sample = []
        if len(start_poss) > 0 and len(end_poss) > 0:
            assert len(start_poss) == len(end_poss), f"start_poss: {start_poss}, end_poss: {end_poss}"
            for start, end in zip(start_poss, end_poss):
                poss_of_a_sample.extend(list(range(start, end)))
        batch["teacher_observation_poss"].append(poss_of_a_sample)

    observation_start_poss = find_ids_poss(batch["student_input_ids"], answer_start_pattern, observation_start_idx)
    observation_end_poss = find_ids_poss(batch["student_input_ids"], answer_start_pattern, observation_end_idx)
    batch["student_observation_poss"] = []
    assert len(observation_start_poss) == len(observation_end_poss)
    for start_poss, end_poss in zip(observation_start_poss, observation_end_poss):
        poss_of_a_sample = []
        if len(start_poss) > 0 and len(end_poss) > 0:
            assert len(start_poss) == len(end_poss), f"start_poss: {start_poss}, end_poss: {end_poss}"
            for start, end in zip(start_poss, end_poss):
                poss_of_a_sample.extend(list(range(start, end)))
        batch["student_observation_poss"].append(poss_of_a_sample)

    # mask tokens of '<|im_start|>assistant', '<|endoftext|>', and '<abs_vis_token_pad>' 
    batch["student_labels"] = generate_labels_after_multi_token_start(batch["student_input_ids"], answer_start_pattern, ignore_ids=[img_pad_idx, img_start_idx, img_end_idx, end_pad_token_idx, latent_pad_idx, observation_start_idx, observation_end_idx, latent_end_idx])

    return batch


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

#train_dataset = [d for d in [preprocess_function(sample) for sample in all_train_dataset[:]] if d is not None]


dataset_names = ""
for data_path in args.data_path:
    dataset_name = data_path.split("/")[-2]
    dataset_names += f"-{dataset_name}"

# Automatically construct save name with the hyperparameters
exp_name = f"{args.stage}-ep{args.epochs}-bsz{args.bsz}-lr{args.lr}"

if args.stage == 'avt_stage1':
    exp_name += f"-{args.min_latent_size}-{args.min_latent_compress_factor}-{args.max_latent_compress_factor}-wt{args.alignment_weight}-ce_emphasize_{args.ce_emphasize_factor}"
    exp_name = f"{args.alignment}-" + exp_name

if args.stage == "avt_sft":
    exp_name += f"-ce_emphasize_{args.ce_emphasize_factor}-warmup_{args.ce_emphasize_warmup_steps}"

if args.stage == 'avt_v2_stage1':
    exp_name += f"-latent{args.latent_size}-ce_factor{args.ce_emphasize_factor}"

if args.stage == 'avt_v2_stage2':
    exp_name += f"-latent{args.latent_size}-ce_factor{args.ce_emphasize_factor}-align_wt{args.alignment_weight}"


if args.shuffle_train:
    exp_name += "-shuffle"

save_dir = f"./checkpoints/{exp_name}"
if args.save_model_path != './checkpoints/':
    save_dir = args.save_model_path

if args.stage == 'avt_stage1':
    CustomTrainer = CustomTrainerAVTStage1
    collate_fn = partial(collate_fn_avt_stage1, alignment=args.alignment)
elif args.stage == 'avt_sft':
    CustomTrainer = CustomTrainerSFT
    collate_fn = partial(collate_fn_avt_sft)
elif args.stage == 'avt_v2_stage1':
    CustomTrainer = CustomTrainerAVT_V2_Stage1
    collate_fn = partial(collate_fn_avt_v2_stage1)
elif args.stage == 'avt_v2_stage2':
    CustomTrainer = CustomTrainerAVT_V2_Stage2
    collate_fn = partial(collate_fn_avt_v2_stage2)
elif args.stage == 'avt_v3':
    CustomTrainer = CustomTrainerAVT_V3
    collate_fn = partial(collate_fn_avt_v3)
elif args.stage == 'avt_v3_1':
    CustomTrainer = CustomTrainerAVT_V3_1
    collate_fn = partial(collate_fn_avt_v3)
elif args.stage == 'avt_v4':
    CustomTrainer = CustomTrainerAVT_V4
    collate_fn = partial(collate_fn_avt_v4)


if args.deepspeed != "":
    print(f"Note: DeepSpeed is enabled. Using the deepspeed config in {args.deepspeed} (the bsz per device and gradient_accumulation_steps will be adopted from the deepspeed config)")
is_parallel = int(os.environ.get("WORLD_SIZE", "1")) > 1
if args.stage == 'avt_v2_stage1': 
    gradient_checkpointing = False
elif args.stage == 'avt_v2_stage2':
    gradient_checkpointing = False
elif args.stage in ['avt_sft','avt_stage1','avt_v3', 'avt_v3_1','avt_v4']:
    gradient_checkpointing = True


training_args = SFTConfig(
    output_dir=save_dir,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.bsz,
    gradient_accumulation_steps=args.grad_accum_steps,
    warmup_steps=10,
    learning_rate=args.lr,
    weight_decay=0.01,
    logging_steps=1,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=10,
    optim="adamw_torch_fused",
    bf16=True,
    push_to_hub=False,
    remove_unused_columns=False,
    gradient_checkpointing=gradient_checkpointing, #False if args.stage == "avt_stage1" else True,
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
    report_to=['wandb'] if args.wandb_name is not None else [],
    logging_dir='./logs/',
    logging_strategy='steps',
    # Avoid FLOPs estimation logs (set to False through env if needed)
    disable_tqdm=False,
    # DDP related
    ddp_backend="nccl" if is_parallel else None,
    ddp_find_unused_parameters=False if is_parallel else None,
    dataloader_num_workers=4 if is_parallel else 0,
    dataloader_pin_memory=True,
    # Save only on global rank 0 when running multi-node
    save_on_each_node=False,
    # DeepSpeed config (if provided via --deepspeed)
    deepspeed=(args.deepspeed if getattr(args, 'deepspeed', '') else None),
)

# ---- Inject custom SFT analysis flags into training_args so CustomTrainerSFT can access them ----
if args.stage == 'avt_sft':
    setattr(training_args, 'ce_emphasize_factor', args.ce_emphasize_factor)
    setattr(training_args, 'teacher_reps_dir', args.teacher_reps_dir)

elif args.stage == 'avt_v2_stage1':
    setattr(training_args, 'ce_emphasize_factor', args.ce_emphasize_factor)
    setattr(training_args, 'gradient_checkpointing_kwargs', {"use_reentrant": False})
    setattr(training_args, 'attn_analysis', args.attn_analysis)
    setattr(training_args, 'use_emphasize_latent_attn_loss', args.use_emphasize_latent_attn_loss)
    setattr(training_args, 'emphasize_latent_attn_coef', args.emphasize_latent_attn_coef)
    setattr(training_args, 'emphasize_topk_layers', args.emphasize_topk_layers)
    setattr(training_args, 'attn_loss_layers', args.attn_loss_layers)
    setattr(training_args, 'use_align_vision_latent_loss_projector', args.use_align_vision_latent_loss_projector)
    setattr(training_args, 'use_align_vision_latent_loss_pooling', args.use_align_vision_latent_loss_pooling)
    setattr(training_args, 'align_vision_latent_loss_weight', args.align_vision_latent_loss_weight)
    setattr(training_args, 'latent_size', args.latent_size)
    setattr(training_args, 'emphasize_latent_weight', args.emphasize_latent_weight)
    
elif args.stage == 'avt_v2_stage2':
    setattr(training_args, 'ce_emphasize_factor', args.ce_emphasize_factor)
    setattr(training_args, 'alignment_weight', args.alignment_weight)
    setattr(training_args, 'gradient_checkpointing_kwargs', {"use_reentrant": False})
    setattr(training_args, 'alignment_layer', args.alignment_layer)
    _tld = args.teacher_latent_dir if getattr(args, 'teacher_latent_dir', None) else os.path.join(save_dir, 'teacher_latents')
    setattr(training_args, 'teacher_latent_dir', _tld)


elif args.stage in ['avt_v3', 'avt_v3_1']:
    setattr(training_args, 'ce_emphasize_factor', args.ce_emphasize_factor)
    setattr(training_args, 'alignment_weight', args.alignment_weight)
    setattr(training_args, 'gradient_checkpointing_kwargs', {"use_reentrant": False})
    setattr(training_args, 'use_align_vision_latent_loss_projector', args.use_align_vision_latent_loss_projector)
    setattr(training_args, 'use_align_vision_latent_loss_pooling', args.use_align_vision_latent_loss_pooling)
    setattr(training_args, 'align_vision_latent_loss_weight', args.align_vision_latent_loss_weight)
    setattr(training_args, 'latent_size', args.latent_size)
    setattr(training_args, 'emphasize_latent_weight', args.emphasize_latent_weight)

elif args.stage in ['avt_v4']:
    setattr(training_args, 'ce_emphasize_factor', args.ce_emphasize_factor)
    setattr(training_args, 'alignment_layer', args.alignment_layer)
    setattr(training_args, 'alignment_weight', args.alignment_weight)
    setattr(training_args, 'gradient_checkpointing_kwargs', {"use_reentrant": False})
    setattr(training_args, 'latent_size', args.latent_size)
    setattr(training_args, 'emphasize_latent_weight', args.emphasize_latent_weight)
    setattr(training_args, 'teacher_reps_dir', args.teacher_reps_dir)
    setattr(training_args, 'no_ce', args.no_ce)

# Initialize the trainer (callbacks that need trainer instance will be added after)
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
    processing_class=processor,
    exp_name=exp_name,
)

# Add representation summary callback after trainer is created
if args.stage=='avt_sft' and getattr(args, 'sft_analysis_enable', False):
    from src.trainer import RepSummaryCallback
    trainer.add_callback(RepSummaryCallback(trainer))

"""
Build baseline hidden states for SFT analysis in parallel across ranks.
Protocol:
  - rank0 samples subset_ids and writes to analyzer.save_dir/subset_ids.json, then creates .subset_ready
  - all ranks read the same subset_ids and take a disjoint shard to build baselines concurrently
  - force-save baselines to disk during this phase
  - barrier, then all ranks load full baselines from disk into memory for training-time access
"""
if args.stage == 'avt_sft' and getattr(args, 'sft_analysis_enable', False):
    analyzer = getattr(trainer, 'rep_analyzer', None)
    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0
    world_size = dist.get_world_size() if is_dist else 1
    exp_save_folder = analyzer.exp_save_folder
    if analyzer is not None:
        os.makedirs(exp_save_folder, exist_ok=True)
        subset_file = os.path.join(exp_save_folder, f'subset_ids{dataset_names}.json')
        ready_marker = os.path.join(exp_save_folder, f'.subset_ready{dataset_names}')

        # Step 1: rank0 selects subset and signals readiness
        if rank == 0:
            # fresh subset each run (overwrite prior)
            try:
                if os.path.exists(ready_marker):
                    os.remove(ready_marker)
            except Exception:
                pass
            total_size = len(train_dataset)
            subset_ids = analyzer.select_subset(
                total_size, args.sft_analysis_ratio, args.sft_analysis_max_samples, args.sft_analysis_seed
            )
            logging.info(f"[SFT Analysis] Selected {len(subset_ids)} samples for representation tracking.")
            with open(subset_file, 'w') as f:
                json.dump(subset_ids, f)
                f.flush()
                os.fsync(f.fileno())
            with open(ready_marker, 'w') as f:
                f.write('ready')


        # Step 2: all ranks wait for subset_ready and read subset_ids
        if is_dist and rank != 0:
            import time
            if not os.path.exists(ready_marker):
                logging.info("[SFT Analysis] Waiting for subset ids from rank0...")
            while not os.path.exists(ready_marker):
                time.sleep(0.5)
            with open(subset_file, 'r') as f:
                subset_ids = json.load(f)
            analyzer.set_subset(subset_ids)


        # read subset ids
        import json as _json
        with open(subset_file, 'r') as f:
            subset_ids = _json.load(f)

        # Step 3: partition subset for this rank
        def _partition(lst, r, w):
            n = len(lst)
            per = (n + w - 1) // w
            start = r * per
            end = min(n, start + per)
            return lst[start:end]

        shard_ids = _partition(subset_ids, rank, world_size)
        logging.info(f"[SFT Analysis][rank {rank}/{world_size}] shard size={len(shard_ids)}")

        # Ensure baselines are persisted to disk so every rank can load later
        analyzer.save_baseline_flag = True

        # Prepare model/device and temporarily disable gradient checkpointing
        mdl = trainer.model
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))

        acc = getattr(trainer, "accelerator", None)
        if acc is not None:
            device = acc.device
        else:
            device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
            if device.type == "cuda":
                torch.cuda.set_device(local_rank)
        print(f"[rank {rank}] using device: {device}")
        mdl.to(device)

        _re_enable_gc = False
        try:
            if getattr(mdl, 'is_gradient_checkpointing', False):
                mdl.gradient_checkpointing_disable()
                _re_enable_gc = True
        except Exception:
            pass

        # Step 4: run baseline forward on this rank's shard
        mdl.eval()

        # clean up the saved reps from previous SFT experiments
        rep_save_path = exp_save_folder
        if os.path.isdir(rep_save_path) and rank == 0:
            shutil.rmtree(rep_save_path)
            os.makedirs(rep_save_path, exist_ok=True)

        if is_dist:
            dist.barrier()
            logging.info(f"[SFT Analysis][rank {rank}] passed barrier, start generating baselines")

        with torch.inference_mode():
            bs = min(2, args.bsz)
            for i in tqdm(range(0, len(shard_ids), bs)):
                cur_ids = shard_ids[i:i+bs]
                examples = [{'conversation': train_dataset[j]['conversation'], 'sample_id': j} for j in cur_ids]
                batch_b = collate_fn(examples)
                inputs_model = {
                    'input_ids': batch_b['teacher_input_ids'].to(device),
                    'attention_mask': batch_b['teacher_attention_mask'].to(device),
                    'pixel_values': batch_b['user_assistant_pixel_values'].to(device),
                    'image_grid_thw': batch_b['user_assistant_image_grid_thw'].to(device),
                    'output_hidden_states': True
                }
                outputs = mdl(**inputs_model)
                hidden_states = outputs.hidden_states
                for bi, sid in enumerate(cur_ids):
                    analyzer.build_baseline(int(sid), [h[[bi]] for h in hidden_states])

        mdl.train()
        if _re_enable_gc:
            try:
                mdl.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                if hasattr(mdl, 'enable_input_require_grads'):
                    mdl.enable_input_require_grads()
            except Exception:
                pass

        # Step 5: synchronize all ranks to ensure all shard baselines are saved
        if is_dist:
            logging.info(f"[SFT Analysis][rank {rank}] entering barrier after baseline save")
            dist.barrier()
            logging.info(f"[SFT Analysis][rank {rank}] passed barrier, start loading baselines")

        # Step 6: load all baselines from disk to local memory for training-time updates
        if os.path.isdir(rep_save_path):
            import glob
            paths = glob.glob(os.path.join(rep_save_path, 'baseline_*.pt'))
            loaded = 0
            for p in tqdm(paths, desc=f"[rank {rank}] loading all baseline reps", total=len(paths)):
                try:
                    sid = int(os.path.basename(p).split('_')[-1].split('.pt')[0])
                except Exception:
                    continue
                if sid in analyzer.baseline:
                    continue
                try:
                    tensor = torch.load(p, map_location='cpu')
                    analyzer.baseline[sid] = tensor
                    loaded += 1
                except Exception as e:
                    logging.warning(f"[SFT Analysis] Failed loading baseline from {p}: {e}")
            logging.info(f"[SFT Analysis][rank {rank}] loaded {loaded} baselines from disk.")
        else:
            logging.warning(f"[SFT Analysis] baseline dir not found: {rep_save_path}")


trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
trainer.save_model(training_args.output_dir)

