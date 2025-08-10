import os as _early_os
# Disable parallelism in HuggingFace tokenizers to avoid fork-related warnings/deadlocks
_early_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from functools import partial
import torch
from new.avt_qwen_model import apply_qwen2_5_avt
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig, AutoTokenizer, AutoProcessor
from PIL import Image
import logging
from tqdm import tqdm
from trl import SFTTrainer, SFTConfig
from qwen_vl_utils import process_vision_info

from src.utils import *
from src.task import *
from src.trainer import CustomTrainerStage1, CustomTrainerStage2
from src.trainer import CustomTrainerAVTStage1, CustomTrainerSFT
import random

seed_everything(seed=42)
args=get_args()

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
processor = AutoProcessor.from_pretrained(args.model)

if args.stage in ['stage1', 'stage2']:
    processor.tokenizer.add_tokens("<|latent_pad|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|latent_start|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|latent_end|>", special_tokens=True)
elif args.stage in ['avt_stage1', 'avt_sft']:
    processor.tokenizer.add_tokens("<abs_vis_token_pad>", special_tokens=True)
    processor.tokenizer.add_tokens("<abs_vis_token>", special_tokens=True)
    processor.tokenizer.add_tokens("</abs_vis_token>", special_tokens=True)
    processor.tokenizer.add_tokens("<observation>", special_tokens=True)
    processor.tokenizer.add_tokens("</observation>", special_tokens=True)

config = Qwen2_5_VLConfig.from_pretrained(args.load_model_path)
config.compress_strategy = args.compress_strategy
config.latent_size = args.latent_size
config.stage = args.stage

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

if args.stage in ['stage1']: model.resize_token_embeddings(len(processor.tokenizer))

if args.stage in ['stage1', 'stage2']:
    latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0]
    latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0]
    latent_end_idx = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0]
elif args.stage in ['avt_stage1', 'avt_sft']:
    latent_token_idx = processor.tokenizer("<abs_vis_token_pad>", return_tensors="pt")["input_ids"][0]
    latent_start_idx = processor.tokenizer("<abs_vis_token>", return_tensors="pt")["input_ids"][0]
    latent_end_idx = processor.tokenizer("</abs_vis_token>", return_tensors="pt")["input_ids"][0]
    observation_start_idx = processor.tokenizer("<observation>", return_tensors="pt")["input_ids"][0]
    observation_end_idx = processor.tokenizer("</observation>", return_tensors="pt")["input_ids"][0]

model.config.latent_token_id = int(latent_token_idx)
model.config.latent_start_id = int(latent_start_idx)
model.config.latent_end_id = int(latent_end_idx)

for param in model.visual.parameters():
    param.requires_grad = False



def collate_fn_stage1(examples):
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]

    texts = [place_input_image(text) for text in texts]
    texts = [place_output_image(text) for text in texts]
    # replace <|vision_start|><|image_pad|><|vision_end|> with <abs_vis_token><|image_pad|></abs_vis_token> after each <|im_start|>assistant
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

    end_pad_token_idx = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0]

    # <|latent_start|><|image_pad|><|latent_end|> -> <|latent_start|><|latent_pad|>...<|latent_end|>; pad the sequences to the same length
    new_input_ids, new_attention_mask = process_batch(batch["input_ids"], batch["attention_mask"], 
                                                      latent_start_idx, latent_end_idx, latent_token_idx, args.latent_size, end_pad_token_idx)

    batch["input_ids"] = new_input_ids
    batch["attention_mask"] = new_attention_mask

    answer_start_token_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]

    # mask tokens of '<|im_start|>assistant', '<|endoftext|>', and '<|latent_pad|>' 
    labels = generate_labels_after_multi_token_start(batch["input_ids"], answer_start_token_pattern, end_pad_token_idx, latent_token_idx)
    batch["labels"] = labels

    # return a mask where tokens of <|latent_pad|> are 1, else 0
    image_out_mask = mask_image_output_tokens(batch["input_ids"], latent_start_idx, latent_token_idx)
    batch["image_out_mask"] = image_out_mask

    return batch

def collate_fn_stage2(examples):
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    
    texts = [place_input_image(text) for text in texts]
    texts = [place_output_image(text) for text in texts]
    # replace <|vision_start|><|image_pad|><|vision_end|> with <abs_vis_token><|image_pad|></abs_vis_token> after each <|im_start|>assistant
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

    end_pad_token_idx = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0]

    new_input_ids, new_attention_mask = process_batch(batch["input_ids"], batch["attention_mask"], 
                                                      latent_start_idx, latent_end_idx, latent_token_idx, args.latent_size, end_pad_token_idx)

    batch["input_ids"] = new_input_ids
    batch["attention_mask"] = new_attention_mask

    answer_start_token_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]

    labels = generate_labels_after_multi_token_start(batch["input_ids"], answer_start_token_pattern, end_pad_token_idx, latent_token_idx)
    batch["labels"] = labels
    
    return batch

def collate_fn_avt_stage1(examples, alignment="boxed_start"):
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

    latent_token_idx = processor.tokenizer("<abs_vis_token_pad>", return_tensors="pt")["input_ids"][0]
    latent_start_idx = processor.tokenizer("<abs_vis_token>", return_tensors="pt")["input_ids"][0]
    latent_end_idx = processor.tokenizer("</abs_vis_token>", return_tensors="pt")["input_ids"][0]
    img_pad_token_idx = processor.tokenizer("<|image_pad|>", return_tensors="pt")["input_ids"][0]
    end_pad_token_idx = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0]


    answer_start_token_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]

    # <|latent_start|><|image_pad|><|latent_end|> -> <|latent_start|><|latent_pad|>...<|latent_end|>; pad the sequences to the same length
    batch["student_input_ids"], batch["student_attention_mask"] = process_batch(student_batch["input_ids"], student_batch["attention_mask"], 
                                                      latent_start_idx, latent_end_idx, latent_token_idx, args.min_latent_size, 
                                                      args.min_latent_compress_factor, args.max_latent_compress_factor,
                                                      end_pad_token_idx, batch_assistant_img_token_lens)
    
    #if (batch["student_input_ids"]==151655).sum()==1472:
    #    pass
        
    batch["teacher_input_ids"] = teacher_batch["input_ids"] #replace_assistant_image_pad_with_latent_pad(teacher_batch["input_ids"], answer_start_token_pattern, img_pad_token_idx, latent_token_idx)
    batch["teacher_attention_mask"] = teacher_batch["attention_mask"]

    if alignment == "observation_end":
        alignment_pattern = observation_end_idx
        batch["student_alignment_poss"] = find_ids_poss(batch["student_input_ids"], answer_start_token_pattern, alignment_pattern)
        batch["teacher_alignment_poss"] = find_ids_poss(batch["teacher_input_ids"], answer_start_token_pattern, alignment_pattern)
    elif alignment == "boxed_start":
        alignment_pattern = [processor.tokenizer("\\boxed{", return_tensors="pt")["input_ids"][0], processor.tokenizer(" \\boxed{", return_tensors="pt")["input_ids"][0]]
        batch["student_alignment_poss"] = find_ids_poss(batch["student_input_ids"], answer_start_token_pattern, alignment_pattern)
        batch["teacher_alignment_poss"] = find_ids_poss(batch["teacher_input_ids"], answer_start_token_pattern, alignment_pattern)
    elif alignment == "observation_all":
        student_observation_start_poss = find_ids_poss(batch["student_input_ids"], answer_start_token_pattern, observation_start_idx)
        student_observation_end_poss = find_ids_poss(batch["student_input_ids"], answer_start_token_pattern, observation_end_idx)
        batch["student_alignment_poss"] = []
        assert len(student_observation_start_poss) == len(student_observation_end_poss)
        for start_poss, end_poss in zip(student_observation_start_poss, student_observation_end_poss):
            poss_of_a_sample = []
            if len(start_poss) > 0 and len(end_poss) > 0:
                assert len(start_poss) == len(end_poss), f"start_poss: {start_poss}, end_poss: {end_poss}"
                for start, end in zip(start_poss, end_poss):
                    poss_of_a_sample.extend(list(range(start, end + 1)))
            batch["student_alignment_poss"].append(poss_of_a_sample)
        teacher_observation_start_poss = find_ids_poss(batch["teacher_input_ids"], answer_start_token_pattern, observation_start_idx)
        teacher_observation_end_poss = find_ids_poss(batch["teacher_input_ids"], answer_start_token_pattern, observation_end_idx)
        batch["teacher_alignment_poss"] = []
        assert len(teacher_observation_start_poss) == len(teacher_observation_end_poss)
        for start_poss, end_poss in zip(teacher_observation_start_poss, teacher_observation_end_poss):
            poss_of_a_sample = []
            if len(start_poss) > 0 and len(end_poss) > 0:
                assert len(start_poss) == len(end_poss), f"start_poss: {start_poss}, end_poss: {end_poss}"
                for start, end in zip(start_poss, end_poss):
                    poss_of_a_sample.extend(list(range(start, end + 1)))
            batch["teacher_alignment_poss"].append(poss_of_a_sample)

    # mask tokens of '<|im_start|>assistant', '<|endoftext|>', and '<|latent_pad|>' 
    batch["student_labels"] = generate_labels_after_multi_token_start(batch["student_input_ids"], answer_start_token_pattern, end_pad_token_idx, latent_token_idx)
    batch["teacher_labels"] = generate_labels_after_multi_token_start(batch["teacher_input_ids"], answer_start_token_pattern, end_pad_token_idx, img_pad_token_idx)

    # return a mask where tokens of <|latent_pad|> are 1, else 0
    batch["student_image_out_mask"] = mask_image_output_tokens(batch["student_input_ids"], latent_start_idx, latent_token_idx)

    # return a mask where tokens of <|image_pad|> are 1, else 0
    batch["teacher_image_out_mask"] = mask_image_output_tokens(batch["teacher_input_ids"], latent_start_idx, img_pad_token_idx)

    return batch




def collate_fn_avt_sft(examples):
    # examples: list of {conversation: [...], sample_id: int}
    sample_ids = [ex['sample_id'] for ex in examples]
    conversations = [ex['conversation'] for ex in examples]
    texts = [processor.apply_chat_template(conv, tokenize=False) for conv in conversations]
    texts = [place_output_image_avt(text) for text in texts]
    # Per-sample resize to avoid batch-coupled image token counts
    all_images = []
    for conv in conversations:
        imgs_i, _ = process_vision_info([conv])
        processed_i, _ = resize_by_token_budget_sample_wise([imgs_i])
        all_images.extend(processed_i[0])
    # Optional sanity check: number of <|image_pad|> should match image count
    total_image_pads = sum(t.count("<|image_pad|>") for t in texts)
    if total_image_pads != len(all_images):
        logging.warning(f"[avt_sft] image placeholders ({total_image_pads}) != images ({len(all_images)})")
    teacher_batch = processor(text=texts, images=all_images, return_tensors="pt", padding=True)
    batch = {}
    batch['sample_id'] = torch.tensor(sample_ids, dtype=torch.long)
    batch['user_assistant_pixel_values'] = teacher_batch['pixel_values']
    batch['user_assistant_image_grid_thw'] = teacher_batch['image_grid_thw']
    img_pad_token_idx = processor.tokenizer("<|image_pad|>", return_tensors="pt")["input_ids"][0]
    end_pad_token_idx = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0]
    answer_start_token_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]
    batch["teacher_input_ids"] = teacher_batch["input_ids"]
    batch["teacher_attention_mask"] = teacher_batch["attention_mask"]
    alignment_pattern = [processor.tokenizer("\\boxed{", return_tensors="pt")["input_ids"][0], processor.tokenizer(" \\boxed{", return_tensors="pt")["input_ids"][0]]
    batch["boxed_start_poss"] = find_ids_poss(batch["teacher_input_ids"], answer_start_token_pattern, alignment_pattern)
    teacher_observation_start_poss = find_ids_poss(batch["teacher_input_ids"], answer_start_token_pattern, observation_start_idx)
    teacher_observation_end_poss = find_ids_poss(batch["teacher_input_ids"], answer_start_token_pattern, observation_end_idx)
    batch["observation_poss"] = []
    assert len(teacher_observation_start_poss) == len(teacher_observation_end_poss)
    for start_poss, end_poss in zip(teacher_observation_start_poss, teacher_observation_end_poss):
        poss_of_a_sample = []
        if len(start_poss) > 0 and len(end_poss) > 0:
            assert len(start_poss) == len(end_poss), f"start_poss: {start_poss}, end_poss: {end_poss}"
            for start, end in zip(start_poss, end_poss):
                poss_of_a_sample.extend(list(range(start, end + 1)))
        batch["observation_poss"].append(poss_of_a_sample)
    batch["teacher_labels"] = generate_labels_after_multi_token_start(batch["teacher_input_ids"], answer_start_token_pattern, end_pad_token_idx, img_pad_token_idx)
    # Build non_observation_poss: positions where label != -100 and not in observation_poss
    non_obs_poss = []
    teacher_labels = batch["teacher_labels"]
    for b in range(teacher_labels.size(0)):
        obs_set = set(batch["observation_poss"][b]) if b < len(batch["observation_poss"]) else set()
        valid_indices = [int(i) for i in torch.nonzero(teacher_labels[b] != -100, as_tuple=False).flatten().tolist() if i not in obs_set]
        non_obs_poss.append(valid_indices)
    batch["non_observation_poss"] = non_obs_poss
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
# Check if the preprocess function is abstract_visual_token_single_input_images_preprocess_function
if "avt" in args.stage:
    preprocess_function = partial(preprocess_function, dataset_root=args.dataset_root)
train_dataset = [d for d in [preprocess_function(sample) for sample in all_train_dataset[:]] if d is not None]

# ---- Conversation wrapper for avt_sft stage (adds sample_id) ----
if args.stage == 'avt_sft':
    from torch.utils.data import Dataset as _TorchDataset
    class ConversationDataset(_TorchDataset):
        def __init__(self, conversations):
            self.conversations = conversations
        def __len__(self):
            return len(self.conversations)
        def __getitem__(self, idx):
            return { 'conversation': self.conversations[idx], 'sample_id': idx }
    wrapped_dataset = ConversationDataset(train_dataset)
else:
    wrapped_dataset = None


exp_name = f"ep{args.epochs}-bsz{args.bsz}-lr{1e-5}-{args.min_latent_size}-{args.min_latent_compress_factor}-{args.max_latent_compress_factor}"
if args.stage in ["avt_stage1"]:
    exp_name = f"{args.alignment}-" + exp_name
exp_name = args.stage+'-'+exp_name
if args.shuffle_train:
    exp_name += "-shuffle"
dataset_names = ""
for data_path in args.data_path:
    dataset_name = data_path.split("/")[-2]
    dataset_names += f"-{dataset_name}"
dataset_name += dataset_names
save_dir = f"./checkpoints/{exp_name}"

if args.stage in ['stage1']:
    CustomTrainer = CustomTrainerStage1
    collate_fn = collate_fn_stage1
elif args.stage in ['stage2']:
    CustomTrainer = CustomTrainerStage2
    collate_fn = collate_fn_stage2
elif args.stage in ['avt_stage1']:
    CustomTrainer = CustomTrainerAVTStage1
    collate_fn = partial(collate_fn_avt_stage1, alignment=args.alignment)
elif args.stage == 'avt_sft':
    CustomTrainer = CustomTrainerSFT
    collate_fn = partial(collate_fn_avt_sft)
# 
training_args = SFTConfig(
    output_dir=save_dir,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.bsz,
    gradient_accumulation_steps=args.grad_accum_steps,
    warmup_steps=10,
    learning_rate=1e-5,
    weight_decay=0.01,
    logging_steps=20,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    optim="adamw_torch_fused",
    bf16=True,
    push_to_hub=False,
    remove_unused_columns=False,
    gradient_checkpointing=False if args.stage == "avt_stage1" else True,
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
    report_to=[],
    logging_dir='./logs/',
    logging_strategy='steps',
    # DDP related
    ddp_backend="nccl",
    ddp_find_unused_parameters=False,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
)

# ---- Inject custom SFT analysis flags into training_args so CustomTrainerSFT can access them ----
if args.stage == 'avt_sft':
    setattr(training_args, 'sft_analysis_enable', args.sft_analysis_enable)
    setattr(training_args, 'sft_analysis_save_dir', args.sft_analysis_save_dir)
    setattr(training_args, 'sft_analysis_categories', args.sft_analysis_categories)
    setattr(training_args, 'sft_analysis_save_baseline', args.sft_analysis_save_baseline)
    setattr(training_args, 'dataset_names', dataset_names)

# Initialize the trainer (callbacks that need trainer instance will be added after)
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=wrapped_dataset if args.stage=='avt_sft' else train_dataset,
    data_collator=collate_fn,
    tokenizer=processor.tokenizer,
    exp_name=exp_name,
)

# Add representation summary callback after trainer is created
if args.stage=='avt_sft' and getattr(args, 'sft_analysis_enable', False):
    from src.trainer import RepSummaryCallback
    trainer.add_callback(RepSummaryCallback(trainer))

# Build baseline hidden states for SFT analysis subset before training (rank0 only)
if args.stage=='avt_sft' and getattr(args, 'sft_analysis_enable', False) and _rank == 0:
    analyzer = getattr(trainer, 'rep_analyzer', None)
    if analyzer is not None:
        total_size = len(wrapped_dataset)
        subset_ids = analyzer.select_subset(total_size, args.sft_analysis_ratio, args.sft_analysis_max_samples, args.sft_analysis_seed)
        logging.info(f"[SFT Analysis] Selected {len(subset_ids)} samples for representation tracking.")
        mdl = trainer.model
        mdl.eval()
        with torch.no_grad():
            bs = min(2, args.bsz)
            for i in tqdm(range(0, len(subset_ids), bs)):
                cur_ids = subset_ids[i:i+bs]
                # Reconstruct mini-batch examples identical to training dataset items
                examples = [{ 'conversation': wrapped_dataset[j]['conversation'], 'sample_id': j } for j in cur_ids]
                batch_b = collate_fn(examples)  # uses collate_fn_avt_sft ensuring identical preprocessing & resizing
                # Forward pass mirroring training (teacher / latent_mode False path)
                device = getattr(mdl, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                inputs_model = {
                    'input_ids': batch_b['teacher_input_ids'].to(device),
                    'attention_mask': batch_b['teacher_attention_mask'].to(device),
                    'pixel_values': batch_b['user_assistant_pixel_values'].to(device),
                    'image_grid_thw': batch_b['user_assistant_image_grid_thw'].to(device),
                    'output_hidden_states': True
                }
                outputs = mdl(**inputs_model)
                hidden_states = outputs.hidden_states  # list[L] each (B,S,H)
                for bi, sid in enumerate(cur_ids):
                    analyzer.build_baseline(int(sid), [h[[bi]] for h in hidden_states])
        mdl.train()

trainer.train()
trainer.save_model(training_args.output_dir)

