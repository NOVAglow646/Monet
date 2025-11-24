import os
# Disable parallelism in HuggingFace tokenizers to avoid fork-related warnings/deadlocks
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
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
from time import time
import pdb
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
processor = AutoProcessor.from_pretrained(args.load_model_path, use_fast=True, trust_remote_code=True)

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


def collate_fn_sft_stage1(examples):
    # examples: list of {conversation: [...], sample_id: int}
    batch = {}
    batch['metadata'] = [ex['metadata'] for ex in examples]
    examples = [ex['data'] for ex in examples]
    texts = [processor.apply_chat_template(ex, tokenize=False) for ex in examples]

    # replace <abs_vis_token></abs_vis_token> with <|vision_start|><|image_pad|><|vision_end|> for each <|im_start|>assistant content
    texts = [replace_latent_placeholder_with_img_pad(text) for text in texts]
    #pdb.set_trace()
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

def collate_fn_sft_stage2(examples):
    if _rank == 0:
        start_time = time()
    batch = {}
    metadata = [ex['metadata'] for ex in examples]
    examples = [ex['data'] for ex in examples]

    texts = [processor.apply_chat_template(ex, tokenize=False) for ex in examples]

    # replace `<abs_vis_token></abs_vis_token>`` with `<|vision_start|><|image_pad|><|vision_end|>`` for each `<|im_start|>assistant`` content
    texts = [replace_latent_placeholder_with_img_pad(text) for text in texts]
    
    # add `<abs_vis_token><abs_vis_token_pad>...</abs_vis_token>` after each `<|vision_start|><|image_pad|><|vision_end|>` for each `<|im_start|>assistant` content
    texts = add_latent_pad_after_auxiliary_img(texts, args.latent_size, "<abs_vis_token_pad>")
    
    image_inputs, _ = process_vision_info(examples)
    if args.image_resize == "global":
        image_inputs, new_sizes = resize_by_token_budget(image_inputs, global_max_pixels=args.sft_stage2_global_img_tokens*28*28, per_img_max_pixels=args.sft_stage2_per_img_tokens*28*28)
    elif args.image_resize == "clear_question_img":
        image_inputs, new_sizes = resize_diff(image_inputs)

    total_image_pads = 0    
    for txt in texts:
        total_image_pads += txt.count("<|vision_start|><|image_pad|>")
    assert total_image_pads == len(image_inputs)
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    batch['metadata'] = metadata
    if not args.not_use_4d:
        attn_mask_4d, _ = build_4d_attn(
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

    if args.sft_stage2_align_poss == 'latent_end':
        batch["latent_end_poss"] = find_ids_poss(batch["input_ids"], answer_start_pattern, latent_end_idx)
    
    observation_start_poss = find_ids_poss(batch["input_ids"], answer_start_pattern, observation_start_idx)
    observation_end_poss = find_ids_poss(batch["input_ids"], answer_start_pattern, observation_end_idx)
    batch["observation_poss"] = []
    assert len(observation_start_poss) == len(observation_end_poss)
    for start_poss, end_poss in zip(observation_start_poss, observation_end_poss):
        poss_of_a_sample = []
        if len(start_poss) > 0 and len(end_poss) > 0:
            assert len(start_poss) == len(end_poss), f"start_poss: {start_poss}, end_poss: {end_poss}"
            for start, end in zip(start_poss, end_poss):
                poss_of_a_sample.extend(list(range(start, end)))
        batch["observation_poss"].append(poss_of_a_sample)

    if args.only_predict_obs:
        batch["labels"] = generate_labels_after_multi_token_start_only_allow(batch["input_ids"], answer_start_pattern, allowed_poss=batch["observation_poss"])
    else:
        batch["labels"] = generate_labels_after_multi_token_start(batch["input_ids"], answer_start_pattern, ignore_ids=[end_pad_token_idx, 
        latent_pad_idx, latent_end_idx, img_pad_idx, img_start_idx, img_end_idx, observation_start_idx, observation_end_idx])

    return batch

def collate_fn_sft_stage3(examples, alignment="boxed_start"):
    # Support wrapped examples providing sample_id
    batch = {}
    batch['metadata'] = [ex['metadata'] for ex in examples]
    examples = [ex['data'] for ex in examples]
    batch_user_img_cnts = [sum(1 for step in examples[i][1]['content'] if step["type"] == "image") for i in range(len(examples))]
    batch_assistant_img_cnts = [sum(1 for step in examples[i][2]['content'] if step["type"] == "image") for i in range(len(examples))]
    texts = [processor.apply_chat_template(ex, tokenize=False) for ex in examples]

    # replace <abs_vis_token></abs_vis_token> with <|vision_start|><|image_pad|><|vision_end|> for each <|im_start|>assistant content
    texts = [replace_latent_placeholder_with_img_pad(text) for text in texts]
    image_inputs, _ = process_vision_info(examples)
    image_inputs, new_sizes = resize_by_token_budget(image_inputs, global_max_pixels=args.sft_stage3_img_tokens*28*28, per_img_max_pixels=args.sft_stage3_img_tokens*28*28,)
    
    ################################################
    # student
    ################################################
    # replace <|vision_start|><|image_pad|><|vision_end|> with <abs_vis_token><abs_vis_token_pad>...</abs_vis_token> for each <|im_start|>assistant content
    student_texts = replace_img_pad_with_latent_pad(texts, args.latent_size, "<abs_vis_token_pad>")
    user_examples = remove_auxiliary_images(examples)
    user_image_inputs, _ = process_vision_info(user_examples)
    resize_ptr = 0
    b_ptr = 0
    usr_img_cnt_accum = 0
    if new_sizes is not None:
        for i, img in enumerate(user_image_inputs):
            img = img.resize(new_sizes[resize_ptr], Image.BICUBIC)
            user_image_inputs[i] = img
            resize_ptr += 1
            usr_img_cnt_accum += 1
            if usr_img_cnt_accum == batch_user_img_cnts[b_ptr]:
                resize_ptr += batch_assistant_img_cnts[b_ptr] # user_image_inputs only contain question images of each batch sample, so we need to skip the helper images in the new_sizes by adding batch_assistant_img_cnts[i]
                b_ptr += 1
                usr_img_cnt_accum = 0
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
    processed = preprocess_function(sample, dataset_root=args.dataset_root, allow_no_observation=args.allow_no_observation)
    if processed is not None:
        train_dataset.append(processed)

#train_dataset = [d for d in [preprocess_function(sample) for sample in all_train_dataset[:]] if d is not None]


dataset_names = ""
for data_path in args.data_path:
    dataset_name = data_path.split("/")[-2]
    dataset_names += f"-{dataset_name}"

# Automatically construct save name with the hyperparameters
exp_name = f"{args.stage}-ep{args.epochs}-bsz{args.bsz}-lr{args.lr}"

if args.stage == "sft_stage1":
    exp_name += f"-ce_emphasize_{args.ce_emphasize_factor}-warmup_{args.ce_emphasize_warmup_steps}"

if args.shuffle_train:
    exp_name += "-shuffle"

save_dir = f"./checkpoints/{exp_name}"
if args.save_model_path != './checkpoints/':
    save_dir = args.save_model_path

if args.stage == 'sft_stage1':
    CustomTrainer = CustomTrainerSFT_STAGE1
    collate_fn = partial(collate_fn_sft_stage1)
elif args.stage == 'sft_stage2':
    CustomTrainer = CustomTrainerSFT_STAGE2
    collate_fn = partial(collate_fn_sft_stage2)
elif args.stage == 'sft_stage3':
    CustomTrainer = CustomTrainerSFT_STAGE3
    collate_fn = partial(collate_fn_sft_stage3)

if args.deepspeed != "":
    print(f"Note: DeepSpeed is enabled. Using the deepspeed config in {args.deepspeed} (the bsz per device and gradient_accumulation_steps will be adopted from the deepspeed config)")
is_parallel = int(os.environ.get("WORLD_SIZE", "1")) > 1
gradient_checkpointing = True


training_args = SFTConfig(
    output_dir=save_dir,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.bsz,
    gradient_accumulation_steps=args.grad_accum_steps,
    warmup_steps=10,
    learning_rate=args.lr,
    weight_decay=0.01,
    logging_steps=args.log_freq,
    save_strategy="steps",
    save_steps=args.save_freq,
    save_total_limit=10,
    optim="adamw_torch_fused",
    bf16=True,
    push_to_hub=False,
    remove_unused_columns=False,
    gradient_checkpointing=gradient_checkpointing,
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
if args.stage == 'sft_stage1':
    setattr(training_args, 'ce_emphasize_factor', args.ce_emphasize_factor)
    setattr(training_args, 'teacher_reps_dir', args.teacher_reps_dir)
elif args.stage in ['sft_stage2','sft_stage3']:
    setattr(training_args, 'ce_emphasize_factor', args.ce_emphasize_factor)
    setattr(training_args, 'alignment_layer', args.alignment_layer)
    setattr(training_args, 'alignment_weight', args.alignment_weight)
    setattr(training_args, 'gradient_checkpointing_kwargs', {"use_reentrant": False})
    setattr(training_args, 'latent_size', args.latent_size)
    setattr(training_args, 'emphasize_latent_weight', args.emphasize_latent_weight)
    setattr(training_args, 'teacher_reps_dir', args.teacher_reps_dir)
    setattr(training_args, 'teacher_latent_dir', args.teacher_latent_dir)
    setattr(training_args, 'image_resize', args.image_resize)
    setattr(training_args, 'sft_stage2_align_poss', args.sft_stage2_align_poss)

# Initialize the trainer (callbacks that need trainer instance will be added after)
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
    processing_class=processor,
    exp_name=exp_name,
)


trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
trainer.save_model(training_args.output_dir)

