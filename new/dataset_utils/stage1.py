#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 1: 数据预处理 + policy_mllm 首轮推理 + 样本筛选
=====================================================

目标：
- 读取原始多模态数据集 (PixelReasoner / CoM / CoF)。
- 规范化 question + choices；抽取 ground-truth 文本答案。
- 生成 & 落盘主图及 helper 图像（裁剪 / 标注框 / 线段等）。
- 使用 *policy_mllm* 对 “主图 + 问题” 批量推理；判定是否答对。
- **仅保留首轮答错的样本**（用户要求 policy 正确样本不保留）。
- 输出 `stage1_policy_out.jsonl`，供 Stage 2 使用。

运行示例：

```bash
python stage1_prep_policy.py \
  --dataset-name CoF \
  --policy-model-path /data1/qxwang/checkpoints/Qwen2.5-VL-7B-Instruct \
  --devices 1,2,3,9 \
```

输出：
  ./created_dataset/filtered_data/<dataset_name>/stage1_policy_out.jsonl
  ./created_dataset/filtered_data/<dataset_name>/images/<sid>_<step>.jpg

说明：
- 本阶段只加载 *policy_mllm*（Qwen2.5-VL-7B）。
- 不再做 observation 对齐（insert_alignment_tokens），留到 Stage 3。
- helpers 列表中保留每一步的文本 + 对应 helper 图像路径 (如有)。
- sid 为递增整型（从 --start-id 起）。
"""

import os
import sys
import re
import json
import argparse
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import random
import datasets

# ------------------------------
# AAA 本地工具 (按用户环境)
# ------------------------------
# 如果这些 import 失败，会 fallback 到简化实现。

from AAA_vllm_toolkit.load_and_gen_vllm import (
    vllm_mllm_init,
    vllm_llm_init,
    vllm_mllm_process_batch_from_messages,
    count_qwen_vl_tokens,
    vllm_kill_model,
)
from AAA_vllm_toolkit.extract_and_check import (
    extract_boxed_answer,
    batch_judge,
    quick_batch_judge,
    llm_batch_judge,
    extract_html_answer,
    llm_batch_extract,
    data_spec_batch_judge
)
try:
    from .judge_pipeline import sequential_judge_predictions
except Exception:  # when executed as a script without package context
    from judge_pipeline import sequential_judge_predictions

from transformers import AutoProcessor, AutoTokenizer  # trust_remote_code needed
import multiprocessing as mp

# =============================
# 数据集路径映射 (按用户环境)
# =============================
DEFAULT_DATASETS = {
    "PixelReasoner": {
        "dataset_path": "/home/dids/shiyang/datasets/PixelReasoner-SFT-Data/processed_data.json",
        "dataset_images_root": "/home/dids/shiyang/datasets/PixelReasoner-SFT-Data",
    },
    "CoM_w_MathVista": {
        "dataset_path": "/home/dids/shiyang/datasets/CoMDataset/com_math_processed.jsonl",
        "dataset_images_root": "/home/dids/shiyang/datasets/CoMDataset",
    },
    "CoM_wo_MathVista": {
        "dataset_path": "/home/dids/shiyang/datasets/CoMDataset/com_math_processed.jsonl",
        "dataset_images_root": "/home/dids/shiyang/datasets/CoMDataset",
    },
    "CoF": {
        "dataset_path": "/home/dids/shiyang/datasets/CoF-SFT-Data-5.4k/cof_sft_data.json",
        "dataset_images_root": "/home/dids/shiyang/datasets/CoF-SFT-Data-5.4k",
    },
    "vigorl_spatial_reasoning": {
        "dataset_path": "/home/dids/shiyang/datasets/vigorl_datasets/spatial_reasoning/MCTS_72b_reasoning_chains_train.json",
        "dataset_images_root": "/home/dids/shiyang/datasets/vigorl_datasets",
    },
    "ReFocus": {
        "dataset_path": "/home/dids/shiyang/datasets/ReFocus_Data/train_raw_data/chartqa_vcot/train.jsonl",
        "dataset_images_root": "/home/dids/shiyang/datasets/ReFocus_Data",
    },
    "Visual_CoT_v7w": {
        "dataset_path": "/home/dids/shiyang/datasets/Visual-CoT/metadata/visual7w_cot_train.jsonl",
        "dataset_images_root": "/home/dids/shiyang/datasets/Visual-CoT",
    },
    "Visual_CoT_gqa": {
        "dataset_path": "/home/dids/shiyang/datasets/Visual-CoT/viscot_363k.json",
        "dataset_images_root": "/home/dids/shiyang/datasets/Visual-CoT/cot_images_tar_split",
    },
    "Visual_CoT": {
        "dataset_path": "/home/dids/shiyang/datasets/Visual-CoT/viscot_363k.json",
        "dataset_images_root": "/home/dids/shiyang/datasets/Visual-CoT/cot_images_tar_split",
    },
    "VTS": {
        "dataset_path": "/home/dids/shiyang/datasets/VTS/vts_train.jsonl",
        "dataset_images_root": "/home/dids/shiyang/datasets/VTS",
    },
    "Zebra_CoT_visual_search": {
        "dataset_path": "/home/dids/shiyang/datasets/Zebra-CoT/2D Visual Reasoning - Visual Search",
        "dataset_images_root": "",
    },
    "Zebra_CoT_geometry": {
        "dataset_path": "/home/dids/shiyang/datasets/Zebra-CoT/Scientific Reasoning - Geometry",
        "dataset_images_root": "",
    },
    "Zebra_CoT_physics": {
        "dataset_path": "/home/dids/shiyang/datasets/Zebra-CoT/Scientific Reasoning - Physics",
        "dataset_images_root": "",
    },
    "Zebra_CoT_maze": {
        "dataset_path": "/home/dids/shiyang/datasets/Zebra-CoT/Visual Logic & Strategic Games - Maze",
        "dataset_images_root": "",
    },
    "Zebra_CoT_count": {
        "dataset_path": "/home/dids/shiyang/datasets/Zebra-CoT/3D Visual Reasoning - Multi-Hop Objects Counting",
        "dataset_images_root": "",
    },
    "VTS_1": {
        "dataset_path": "/home/dids/shiyang/datasets/VTS_SFT_Data/VTS_SFT/VTS_SFT_315k.json",
        "dataset_images_root": "/home/dids/shiyang/datasets/VTS_SFT_Data/VTS_SFT",
    },
    "VTS_2": {
        "dataset_path": "/home/dids/shiyang/datasets/VTS_SFT_Data/VTS_SFT/VTS_SFT_315k.json",
        "dataset_images_root": "/home/dids/shiyang/datasets/VTS_SFT_Data/VTS_SFT",
    },
    "VTS_3": {
        "dataset_path": "/home/dids/shiyang/datasets/VTS_SFT_Data/VTS_SFT/VTS_SFT_315k.json",
        "dataset_images_root": "/home/dids/shiyang/datasets/VTS_SFT_Data/VTS_SFT",
    },
    "VTS_4": {
        "dataset_path": "/home/dids/shiyang/datasets/VTS_SFT_Data/VTS_SFT/VTS_SFT_315k.json",
        "dataset_images_root": "/home/dids/shiyang/datasets/VTS_SFT_Data/VTS_SFT",
    }
}


def _ensure_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode != "RGB" else img


def draw_bboxes(
    img: Image.Image,
    bboxes: List[List[int]],
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    line_width: int = 3,
    font_path: Optional[str] = None,
    font_size: int = 16,
) -> Image.Image:
    img = _ensure_rgb(img.copy())
    draw = ImageDraw.Draw(img)
    default_palette = ["red", "lime", "blue", "yellow", "cyan", "magenta", "orange"]
    if colors is None:
        colors = [default_palette[i % len(default_palette)] for i in range(len(bboxes))]
    font = None
    if labels:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
    for i, bbox in enumerate(bboxes):
        if len(bbox) != 4:
            continue
        xmin, ymin, xmax, ymax = bbox
        color = colors[i]
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=line_width)
        if labels and i < len(labels):
            text = labels[i]
            w = draw.textlength(text, font=font)
            h = font.getbbox(text)[3] - font.getbbox(text)[1] if font else 10
            draw.rectangle([xmin, ymin - h - 4, xmin + w + 6, ymin], fill=color)
            draw.text((xmin + 3, ymin - h - 2), text, fill="black", font=font)
    return img


def valid_bbox(bbox: Union[List[int], Tuple[int, int, int, int]]) -> bool:
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        if (
            not isinstance(bbox[0], int)
            or not isinstance(bbox[1], int)
            or not isinstance(bbox[2], int)
            or not isinstance(bbox[3], int)
        ):
            return False
        xmin, ymin, xmax, ymax = bbox
        return xmin < xmax and ymin < ymax
    return False


def draw_line(
    img: Image.Image,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: str = "red",
    width: int = 3,
) -> Image.Image:
    img = _ensure_rgb(img.copy())
    draw = ImageDraw.Draw(img)
    draw.line([pt1, pt2], fill=color, width=width)
    return img


def valid_line_pt(pt: Tuple[int, int, int, int]) -> bool:
    if (
        not isinstance(pt[0], int)
        or not isinstance(pt[1], int)
        or not isinstance(pt[2], int)
        or not isinstance(pt[3], int)
    ):
        return False
    return pt[2] > pt[0] and pt[3] > pt[1] and pt[0] >= 0 and pt[1] >= 0


def valid_img_size(img: Optional[Image.Image]) -> bool:
    if img is None:
        return False
    w, h = img.size
    return not (w == 0 or h == 0)


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def choice2str(gt_choices: Optional[List[str]], letter: str) -> str:
    if gt_choices is None:
        return letter
    if not letter:
        return letter
    letter = letter.strip().upper()
    if letter and letter in LETTERS:
        idx = LETTERS.index(letter)
        if idx < len(gt_choices):
            return gt_choices[idx]
    return letter


def add_boxed_instruction(question: str, allow_None=True, mode="normal") -> str:
    if mode == "multi_choice":
        boxed_instruction = "Put the letter of your choice within \\boxed{}."
    elif mode == "normal":
        boxed_instruction = "Put your final answer within \\boxed{}."
    elif mode == "single_word":
        boxed_instruction = (
            "Given the answer in a single word and put it within \\boxed{}."
        )

    if allow_None:
        return (
            question
            + f"\n{boxed_instruction}"
            + " If you cannot see relevant visual information to infer the answer from the image, just output \\boxed{None} and don't guess the answer based on your knowledge."
        )
    else:
        return question + f"\n{boxed_instruction}"


# ============================================================
# 各数据集解析器 —— 返回标准结构
# ============================================================
# 标准结构：
#   {
#     'qid': qid (int/str),
#     'question': str,
#     'gt_choices': [str] or None,
#     'gt_answer_text': str or None,
#     'main_image': PIL.Image,
#     'helpers': [ { 'text': str, 'image': PIL.Image or None, 'type': str, 'step_idx': int } ... ]
#   }
# 如果解析失败，返回 None。
# ============================================================


def parse_pixelreasoner(
    sample: Dict[str, Any], dataset_images_root: Path
) -> Optional[Dict[str, Any]]:
    # ---------------- question / choices ----------------
    raw_q = sample["question"]

    def _get_pure_question(text: str) -> str:
        s = text.find("Question:")
        e = text.find("\nThink in the mind first,")
        if s == -1 or e == -1:
            return text.strip()
        return text[s + len("Question:") : e].strip()

    def _extract_choices(text: str) -> Optional[List[str]]:
        s = text.find("\nchoices:\n")
        e = text.find("\n\nGuidelines")
        if s == -1 or e == -1:
            return None
        choices_str = text[s + len("\nchoices:\n") : e]
        lines = []
        for line in choices_str.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                parts = line.split(":", 1)
                lines.append(parts[1].strip())
            else:
                lines.append(line)
        return lines or None

    def _remove_choices(q: str) -> str:
        s = q.find("\nchoices:\n")
        if s == -1:
            return q.strip()
        return q[:s].strip()

    gt_choices = _extract_choices(raw_q)
    question = add_boxed_instruction(_remove_choices(raw_q))

    # ---------------- main image ----------------
    img_path = dataset_images_root / sample["image"]
    if not img_path.is_file():
        print(f"[PixelReasoner] missing image: {img_path}")
        return None
    main_img = Image.open(img_path).convert("RGB")
    W, H = main_img.size

    # ---------------- steps -> helpers, GT ----------------
    steps = sample.get("response_steps", [])
    helpers = []
    gt_letter = None
    for i, step in enumerate(steps):
        resp_str = step.get("response_str", "")
        # Keep raw text
        step_text = resp_str
        # collect GT if boxed
        _gt = extract_boxed_answer(resp_str)
        if _gt is not None:
            gt_letter = _gt
        # manipulation -> crop
        mani = step.get("manipulation")
        if mani and mani.get("type") == "crop" and "<abs_vis_token>" in resp_str:
            bbox_norm = mani.get("parameters", None)
            if not bbox_norm or len(bbox_norm) != 4:
                continue
            x0 = int(bbox_norm[0] * W)
            y0 = int(bbox_norm[1] * H)
            x1 = int(bbox_norm[2] * W)
            y1 = int(bbox_norm[3] * H)
            helper_img = main_img.crop((x0, y0, x1, y1))
            if not valid_img_size(helper_img):
                return None
            helpers.append(
                {
                    "step_idx": i,
                    "text": step_text,
                    "image": helper_img,
                    "type": "crop",
                }
            )
        else:
            # text-only step? 仅在需要还原 COT 时保留
            helpers.append(
                {
                    "step_idx": i,
                    "text": step_text,
                    "image": None,
                    "type": "text",
                }
            )

    gt_answer_text = None
    if gt_letter is not None:
        gt_answer_text = choice2str(gt_choices, gt_letter)

    return {
        "qid": sample.get("qid"),
        "question": question,
        "gt_choices": gt_choices,
        "gt_answer_text": gt_answer_text,
        "main_image": main_img,
        "helpers": helpers,
    }


def parse_com(
    sample: Dict[str, Any], dataset_images_root: Path, w_mathvista: bool = True
) -> Optional[Dict[str, Any]]:
    raw_q = sample["question"]

    def _remove_choices(question: str) -> str:
        s = question.find("\nChoices:")
        if s == -1:
            return question.strip()
        return question[:s].strip()

    def _extract_choices(question: str) -> Optional[List[str]]:
        s = question.find("\nChoices:\n")
        if s == -1:
            return None
        choices_str = question[s + len("\nChoices:\n") :].strip()
        # pattern picks (A) ... (B) ... etc
        pattern = r"\([A-H]\)\s*(.*?)\s*(?=\([A-H]\)|$)"
        matches = re.findall(pattern, choices_str, re.DOTALL)
        if not matches:
            return None
        return [m.strip() for m in matches]

    gt_choices = _extract_choices(raw_q)
    question = add_boxed_instruction(_remove_choices(raw_q))
    if not w_mathvista:
        if "MathVista" in sample["image"]:
            return None
    img_path = dataset_images_root / sample["image"]
    if not img_path.is_file():
        print(f"[CoM] missing image: {img_path}")
        return None
    main_img = Image.open(img_path).convert("RGB")

    gt_raw = sample.get("answer")
    gt_answer_text = None
    if gt_raw is not None:
        gt_answer_text = (
            choice2str(gt_choices, str(gt_raw)) if gt_choices else str(gt_raw)
        )

    steps = sample.get("response_steps", [])
    helpers = []
    for i, step in enumerate(steps):
        resp_str = step.get("response_str", "")
        mani = step.get("manipulation")
        helper_img = None
        typ = "text"
        if mani:
            mtype = mani.get("type")
            if mtype == "crop_and_zoomin":
                bbox = mani.get("parameters")
                if bbox and len(bbox) == 4 and valid_bbox(bbox):
                    helper_img = main_img.crop(tuple(bbox))
                    typ = "crop"
                else:
                    return None
            elif mtype == "grounding":
                bboxes = mani.get("parameters")
                if bboxes is not None:
                    if isinstance(bboxes, dict):  # skip weird
                        return None
                    else:
                        if not isinstance(bboxes[0], list):
                            bboxes = [bboxes]
                        if not all(valid_bbox(bbox) for bbox in bboxes):
                            return None
                        helper_img = draw_bboxes(main_img, bboxes=bboxes)
                        typ = "bbox"
                else:
                    return None
            elif mtype == "line":
                pts = mani.get("parameters")
                if pts is not None and len(pts) == 4 and valid_line_pt(pts):
                    # 用户原逻辑：将线约束成水平或垂直
                    if "MathVista" not in sample["image"]:
                        if pts[2] - pts[0] < pts[3] - pts[1]:
                            pts[2] = pts[0]
                        else:
                            pts[3] = pts[1]
                    helper_img = draw_line(main_img, (pts[0], pts[1]), (pts[2], pts[3]))
                    typ = "line"
                else:
                    return None
            else:
                return None
        if helper_img is not None and not valid_img_size(helper_img):
            helper_img = None
            typ = "text"
        helpers.append(
            {
                "step_idx": i,
                "text": resp_str,
                "image": helper_img,
                "type": typ,
            }
        )

    return {
        "qid": sample.get("qid"),
        "question": question,
        "gt_choices": gt_choices,
        "gt_answer_text": gt_answer_text,
        "main_image": main_img,
        "helpers": helpers,
    }


def parse_cof(
    sample: Dict[str, Any], dataset_images_root: Path
) -> Optional[Dict[str, Any]]:
    # CoF 样本结构：{"images": [...], "messages": [...]}
    images_rel = sample.get("images", [])
    if len(images_rel) < 2:
        return None
    main_rel = images_rel[0]
    main_path = dataset_images_root / "images" / main_rel
    if not main_path.is_file():
        print(f"[CoF] missing main image: {main_path}")
        return None
    main_img = Image.open(main_path).convert("RGB")

    msgs = sample.get("messages", [])
    if len(msgs) < 2:
        return None

    def _rmv_tool_call_instruction(question: str) -> str:
        s = question.find("\nThink in the mind first,")
        if s == -1:
            return question.strip().replace("<image> ", "")
        return question[:s].strip().replace("<image> ", "")

    def _rmv_choices(question: str) -> str:
        s = question.find("\n(A)")
        if s == -1:
            return question.strip()
        return question[:s].strip()

    def _extract_choices(question: str) -> Optional[List[str]]:
        s = question.find("?\n")
        e = question.find("\nAnswer with the option")
        if s == -1 or e == -1:
            return None
        choices_str = question[s + 2 : e].strip()
        pattern = r"\([A-D]\)\s*(.*?)\s*(?=\([A-D]\)|$)"
        matches = re.findall(pattern, choices_str, re.DOTALL)
        if not matches:
            return None
        return [m.strip() for m in matches]

    def _get_bbox_and_rmv_tool(text: str):
        text = (
            text.replace("<think> ", "")
            .replace("</think>", "")
            .replace("<answer>", "\\boxed{")
            .replace("</answer>", "}")
        )
        s = text.find("<tool_call>")
        e = text.find("</tool_call>")
        if s == -1 or e == -1:
            return None, text
        tool_call_str = text[s + len("<tool_call>") : e]
        bbox = json.loads(tool_call_str)["arguments"]["bbox_2d"]
        return bbox, text[:s] + "<abs_vis_token></abs_vis_token>"

    # 第二条消息 (msgs[1]) 应是 user question
    raw_q = msgs[1]["content"]
    question_clean = _rmv_choices(_rmv_tool_call_instruction(raw_q))
    gt_choices = _extract_choices(raw_q)
    question = add_boxed_instruction(question_clean)

    # helpers: 之后的 assistant 消息里带 tool_call -> bbox -> 对应 images_rel[img_id]
    helpers = []
    img_id = 1
    step_id = 0
    gt_letter = None
    for i, step in enumerate(msgs[2:]):
        if step.get("role") == "user":
            continue
        text = step.get("content", "")

        gt = extract_html_answer(text)
        bbox, resp_wo_tool = _get_bbox_and_rmv_tool(step["content"])
        helper_img = None
        if bbox is not None:
            typ = "crop"
            helper_img = Image.open(
                dataset_images_root / "images" / images_rel[img_id]
            ).convert("RGB")
            if not valid_img_size(helper_img):
                print(f"Invalid image size for helper image, return None")
                return None
            img_id += 1

        helpers.append(
            {
                "step_idx": step_id,
                "text": resp_wo_tool,
                "image": helper_img,
                "type": typ,
            }
        )
        step_id += 1

    gt_answer_text = None
    if gt is not None:
        gt_answer_text = choice2str(gt_choices, gt)

    # 从路径提取 qid（用户原逻辑：第一个路径片段）
    try:
        qid = int(main_rel.split("/")[0])
    except Exception:  # noqa
        qid = main_rel

    return {
        "qid": qid,
        "question": question,
        "gt_choices": gt_choices,
        "gt_answer_text": gt_answer_text,
        "main_image": main_img,
        "helpers": helpers,
    }


def parse_refocus(
    sample: Dict[str, Any], dataset_images_root: Path
) -> Optional[Dict[str, Any]]:

    def _remove_action_and_code_format(step: str) -> str:
        # 移除代码块
        step = re.sub(
            r"\n\nACTION \d+:\n```.*?```",
            "<abs_vis_token></abs_vis_token>",
            step,
            flags=re.DOTALL,
        )
        return re.sub(r"\n\nACTION \d+:.*?\.\n", "", step, flags=re.DOTALL).strip()

    def _remove_thought_format(step: str) -> str:
        # 移除思考格式
        return re.sub(r"THOUGHT \d+: ", "", step, flags=re.DOTALL).strip()

    if sample["edited_image"] == "":
        return None

    question = sample["question"]
    gt_answer_text = sample["answer"]
    main_img = Image.open(dataset_images_root / "images" / sample["image"]).convert(
        "RGB"
    )
    cot = sample["thoughts"]
    helpers = []
    for i, step in enumerate(cot):
        raw_step = step
        formatted_step = _remove_thought_format(
            _remove_action_and_code_format(raw_step)
        )
        helper_img = None
        if "<abs_vis_token></abs_vis_token>" in formatted_step:
            helper_img = Image.open(
                dataset_images_root / "train_raw_data" / sample["edited_image"]
            ).convert("RGB")
        helpers.append(
            {
                "step_idx": i,
                "text": formatted_step,
                "image": helper_img,
                "type": "highlight",
            }
        )
    return {
        "qid": sample["id"],
        "question": add_boxed_instruction(question),
        "gt_choices": None,
        "gt_answer_text": gt_answer_text,
        "main_image": main_img,
        "helpers": helpers,
    }


def parse_visual_cot_w_choices(
    sample: Dict[str, Any], dataset_images_root: Path, dataset_name: str, qid: int
) -> Optional[Dict[str, Any]]:
    def _remove_image_pad(text: str) -> str:
        return text.replace("<image>\n", "")

    def _remove_instruction(text: str) -> str:
        s = text.find(" Please provide the")
        if s == -1:
            return text
        return text[:s]

    def _add_choices(question: str, choices: List[str]) -> Optional[List[str]]:
        if not choices:
            return None
        choice_letter = "A"
        for choice in choices:
            question += f"\n{choice_letter}: {choice}"
            choice_letter = chr(ord(choice_letter) + 1)
        return question

    question = sample["question"]
    choices = sample["multiple_choices"]
    answer = sample["answer"]
    if answer and answer not in choices:
        insert_pos = random.randint(0, len(choices))
        choices.insert(insert_pos, answer)
    question_with_choices = _add_choices(question, choices)
    gt_answer_text = sample["answer"]
    sub_dataset_name = dataset_name.split("_")[-1]
    main_img_path = (
        dataset_images_root
        / "images"
        / "cot_image_data"
        / sub_dataset_name
        / sample["image"]
    )
    if not os.path.exists(main_img_path):
        return None
    main_img = Image.open(main_img_path).convert("RGB")
    bbox = tuple(sample["bboxs"][0])
    if not valid_bbox(bbox):
        return None
    helper_img = main_img.crop(bbox)

    helpers = [
        {
            "step_idx": 0,
            "text": "<abs_vis_token></abs_vis_token>",
            "image": helper_img,
            "type": "crop",
        }
    ]

    return {
        "qid": qid,
        "question": add_boxed_instruction(
            question_with_choices, allow_None=False, mode="multi_choice"
        ),
        "gt_choices": choices,
        "gt_answer_text": gt_answer_text,
        "main_image": main_img,
        "helpers": helpers,
    }


def parse_visual_cot_wo_choices(
    sample: Dict[str, Any], dataset_images_root: Path, dataset_name: str, qid: int
) -> Optional[Dict[str, Any]]:
    question = sample["question"]

    gt_answer_text = sample["answer"]
    sub_dataset_name = dataset_name.split("_")[-1]
    main_img_path = (
        dataset_images_root
        / "images"
        / "cot_image_data"
        / sub_dataset_name
        / sample["image"]
    )
    if not os.path.exists(main_img_path):
        return None
    main_img = Image.open(main_img_path).convert("RGB")
    bbox = tuple(sample["bboxs"][0])
    if not valid_bbox(bbox):
        return None
    helper_img = main_img.crop(bbox)

    helpers = [
        {
            "step_idx": 0,
            "text": "<abs_vis_token></abs_vis_token>",
            "image": helper_img,
            "type": "crop",
        }
    ]

    return {
        "qid": qid,
        "question": add_boxed_instruction(
            question, allow_None=False, mode="single_word"
        ),
        "gt_choices": None,
        "gt_answer_text": gt_answer_text,
        "main_image": main_img,
        "helpers": helpers,
    }

def parse_visual_cot_all(
    sample: Dict[str, Any], dataset_images_root: Path, dataset_name: str, qid: int
) -> Optional[Dict[str, Any]]:
    def parse_bbox(bbox_str: str) -> Optional[Tuple[int, int, int, int]]:
        start_pos = bbox_str.find("[")
        end_pos = bbox_str.find("]")
        if start_pos == -1 or end_pos == -1 or end_pos <= start_pos:
            return None
        bbox_values = bbox_str[start_pos + 1 : end_pos].split(",")
        if len(bbox_values) != 4:
            return None

        bbox = tuple(int(value.strip()) for value in bbox_values)
        if valid_bbox(bbox):
            return bbox
    qid = sample["question_id"]
    conversations = sample["conversations"]
    question = conversations[0]["value"].replace("Please provide the bounding box coordinate of the region that can help you answer the question better.", "").strip()
    gt_answer_text = conversations[-1]["value"]
    images = sample["image"]
    main_img_path = (
        dataset_images_root
        / images[0]
    )
    img_ptr = 1
    step_ptr = 1
    if not os.path.exists(main_img_path):
        return None
    main_img = Image.open(main_img_path).convert("RGB")
    
    helpers = []
    for i, conv in enumerate(conversations[1:]):
        if "<image>" in conv["value"]:
            bbox = parse_bbox(images[img_ptr])
            if not valid_bbox(bbox):
                return None
            helper_img = main_img.crop(bbox)
            helpers.append(
                {
                    "step_idx": step_ptr,
                    "text": "<abs_vis_token></abs_vis_token>",
                    "image": helper_img,
                    "type": "crop",
                }
            )
            step_ptr += 1
        elif "[" in conv["value"] and "]" in conv["value"]:
            continue
        else:
            helpers.append(
                {
                    "step_idx": step_ptr,
                    "text": conv["value"],
                    "image": None,
                    "type": "text",
                }
            )
            step_ptr += 1

    return {
        "qid": qid,
        "question":question,
        "gt_choices": None,
        "gt_answer_text": gt_answer_text,
        "main_image": main_img,
        "helpers": helpers,
    }


def parse_zebra_cot(
    sample: Dict[str, Any], dataset_images_root: Path
) -> Optional[Dict[str, Any]]:
    main_img = sample["problem_image_1"]
    if main_img is None:
        return None

    def _split_cot_by_thought(text):
        pattern = r"(?=THOUGHT\s+\d+:)"
        steps = re.split(pattern, text.strip())

        steps = [
            re.sub(r"^THOUGHT\s+\d+:\s*", "", step).strip()
            for step in steps
            if step.strip()
        ]
        return steps

    def _get_img_key(step: str) -> Optional[str]:
        match = re.search(r"<image_start>\[(reasoning_image_\d+|problem_image_\d+)\]<image_end>", step)
        if match:
            return match.group(1)
        return None

    def _replace_img_pad(step: str):
        return re.sub(
            r"<image_start>\[(reasoning_image_\d+|problem_image_\d+)]<image_end>",
            "<abs_vis_token></abs_vis_token>",
            step,
        )

    def _remove_question_img_pad(txt: str) -> str:
        # (?s) = DOTALL，让 . 匹配换行
        pattern = (
            r'(?s)(?<=\.\s{1})'   # 可变长向后断言：定位在 ".␠␠" 之后
            r'.*?:\n'             # 非贪婪地吃掉直到最近 ":\n"（含它）
            r'<image_start>\[problem_image_1\]<image_end>'  # 图片标签
        )
        txt = re.sub(pattern, '', txt)  # 直接整段清空
        txt = re.sub(r'<image_start>\[problem_image_1\]<image_end>', '', txt)
        return txt
    question = _remove_question_img_pad(sample["Question"])
    
    
    cot = sample["Text Reasoning Trace"]
    steps = _split_cot_by_thought(cot)
    helpers = []
    for i, step in enumerate(steps):
        helper_img = None
        if "<image_start>" in step:
            img_key = _get_img_key(step)
            helper_img = sample[img_key]
            step = _replace_img_pad(step)

        helper = {"step_idx": i, "text": step, "image": helper_img, "type": "any"}
        helpers.append(helper)
    gt_answer_text = sample["Final Answer"].strip()

    return {
        "qid": None,
        "question": add_boxed_instruction(question, allow_None=False, mode="normal"),
        "gt_choices": None,
        "gt_answer_text": gt_answer_text,
        "main_image": main_img,
        "helpers": helpers,
    }


def parse_vts(
    sample: Dict[str, Any], dataset_images_root: Path
) -> Optional[Dict[str, Any]]:
    # remove this helper step
    "The bounding box coordinates are:"
    "Here's the depth map:"
    "However, no image was returned"
    "Here's the segmented image:"
    "Here's the cropped image:"
    "Here's the overlayed image:"
    
    # process
    "Extracted text: Pumpkins sold, Day, Number of pumpkins, Friday, 84, Saturday, Sunday, Monday\n<image>"
    
    # discard this sample
    "Similarity scores:"
    
    # w_choice
    "<image>what vehicle is it?\nChoices:\nA. Car.\nB. Bicycle.\nC. Truck.\nD. Boat.\nAnswer with the option's letter from the given choices directly."
    
    helpers = []
    messages = sample["messages"]
    images = sample["images"]
    question = messages[1]["content"].replace("<image>", "").strip()
    main_img = Image.open(
        dataset_images_root / images[0]
    ).convert("RGB")
    if "Choices" in question:
        question = add_boxed_instruction(question, allow_None=False, mode="multi_choice")
    else:
        question = add_boxed_instruction(question, allow_None=False, mode="normal")        

    img_ptr = 1
    cot_step_id = 0
    total_messages = len(messages)
    for i, msg in enumerate(messages):
        if msg["role"] == "system" or msg["role"] == "user":
            continue
        if i==1:
            continue
        if msg["role"] == "assistant":
            content_dict = json.loads(msg["content"])
            if not isinstance(content_dict, dict): # invalid helper content
                print("[VTS] invalid helper content, discard this sample")
                return None
            add_use_content_as_next_helper_step = False
            if i+1 < total_messages and messages[i+1]["role"] == "user":
                user_content = messages[i+1]["content"]
                if "Similarity scores:" in user_content:
                    print("[VTS] Similarity scores found, discard this sample")
                    return None
                if "Extracted text:" in user_content:
                    user_content = user_content.replace("<image>", "")
                    add_use_content_as_next_helper_step = True
                user_img_cnt = user_content.count("<image>")
                if user_img_cnt == 0:
                    helper_img = None 
                else:
                    user_content = messages[i+1]["content"].replace("Here's the original image: <image>", "")
                    if user_content.count("<image>") > 1: # multiple images in a helper step, discard
                        print("[VTS] multiple images in a helper step, discard")
                        return None
                    if user_img_cnt == 0:
                        helper_img = None
                    else:
                        if "Extracted text:" in user_content: # the image in the OCR step is the original image, so we don't need it as the helper image
                            helper_img = None
                        else:
                            helper_img = Image.open(
                                dataset_images_root / images[img_ptr]
                            ).convert("RGB")
                    img_ptr += user_img_cnt
            
            step_text = content_dict["thought"]
            if helper_img is not None:
                step_text += " <abs_vis_token></abs_vis_token>"
            if add_use_content_as_next_helper_step:
                step_text += f" {user_content}"
            gt_answer_text = content_dict["action"].get("final_response", "")   
            if not isinstance(gt_answer_text, str):   # invalid answer (not str)
                print("[VTS] invalid answer, discard this sample")
                return None
            if "yes" in gt_answer_text.lower() or "no" in gt_answer_text.lower(): # drop yes/no answers
                print("[VTS] yes/no answer found, discard this sample")
                return None
            if gt_answer_text != "":
                step_text += f" The final answer is: {gt_answer_text}"

            helpers.append(
                {
                    "step_idx": cot_step_id,
                    "text": step_text,
                    "image": helper_img,
                    "type": "any"        
                }
            )
            helper_img = None
            cot_step_id += 1
            
    return {
        "qid": None,
        "question": question,
        "gt_choices": None,
        "gt_answer_text": gt_answer_text,
        "main_image": main_img,
        "helpers": helpers,
    }

def build_policy_conversation(question: str, pil_img: Image.Image) -> Dict[str, Any]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": question},
            ],
        }
    ]


# ============================================================
# 保存图像到输出目录
# ============================================================


def save_images_for_sample(
    sid: int, out_img_dir: Path, main_img: Image.Image, helpers: List[Dict[str, Any]]
):
    out_img_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    # main
    main_path = out_img_dir / f"{sid}_0.jpg"
    main_img.save(main_path)
    paths.append(str(main_path))
    # helpers
    idx = 1
    for h in helpers:
        img = h.get("image")
        if img is None:
            # paths.append(None)
            continue
        p = out_img_dir / f"{sid}_{idx}.jpg"
        img.save(p)
        paths.append(str(p))
        idx += 1
    return paths  # [main_path, helper1_path_orNone, ...]


# ============================================================
# policy 模型推理 (批)
# ============================================================


def _dp_worker_run(
    device_ids: List[int],
    conv_list: List[list],
    max_model_len: Optional[int],
    policy_model_path: str,
    tp: int,
    batch_size: int = 8192,
) -> Tuple[List[Optional[str]], List[Optional[str]], List[bool]]:
    """单个进程在指定 GPU 组上运行一个 vLLM 实例并完成一个分片的推理。"""
    import os as _os
    from transformers import AutoProcessor as _AutoProcessor, AutoTokenizer as _AutoTokenizer
    from AAA_vllm_toolkit.load_and_gen_vllm import (
        vllm_mllm_init as _vllm_mllm_init,
        vllm_mllm_process_batch_from_messages as _proc_from_msgs,
        count_qwen_vl_tokens as _count_tokens,
        vllm_kill_model as _vllm_kill_model,
    )
    from AAA_vllm_toolkit.extract_and_check import extract_boxed_answer as _extract_boxed_answer

    _os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in device_ids)
    # 每个 worker 独立加载 tokenizer/processor 与模型
    _tokenizer = _AutoTokenizer.from_pretrained(policy_model_path, trust_remote_code=True)
    _processor = _AutoProcessor.from_pretrained(policy_model_path)
    _policy_mllm, _sampling_params = _vllm_mllm_init(policy_model_path, tp=tp)
    try:
        raw_outs_all: List[Optional[str]] = []
        extr_outs_all: List[Optional[str]] = []
        keep_mask_all: List[bool] = []

        for start in range(0, len(conv_list), batch_size):
            sub_convs = conv_list[start : start + batch_size]
            inputs = _proc_from_msgs(sub_convs, _processor)

            keep_mask = [True] * len(inputs)
            if max_model_len is not None and _count_tokens is not None:
                tok_lens = _count_tokens(inputs, _tokenizer, _processor)
                for i, L in enumerate(tok_lens):
                    if L > max_model_len:
                        keep_mask[i] = False

            kept_inputs = [inp for inp, k in zip(inputs, keep_mask) if k]
            if kept_inputs:
                outputs = _policy_mllm.generate(kept_inputs, sampling_params=_sampling_params, use_tqdm=True)
            else:
                outputs = []

            raw_outs, extr_outs = [], []
            j = 0
            for k in keep_mask:
                if not k:
                    raw_outs.append(None)
                    extr_outs.append(None)
                else:
                    o = outputs[j]
                    j += 1
                    text = o.outputs[0].text.strip() if o.outputs else ""
                    raw_outs.append(text)
                    extr_outs.append(_extract_boxed_answer(text))

            raw_outs_all.extend(raw_outs)
            extr_outs_all.extend(extr_outs)
            keep_mask_all.extend(keep_mask)

        return raw_outs_all, extr_outs_all, keep_mask_all
    finally:
        try:
            _vllm_kill_model(_policy_mllm)
        except Exception:
            pass


def _dp_worker_entry(
    queue,
    device_ids: List[int],
    conv_list: List[list],
    max_model_len: Optional[int],
    policy_model_path: str,
    tp: int,
    batch_size: int,
    shard_idx: int,
):
    """进程入口：运行分片并将结果回传到队列。"""
    try:
        raw, extr, keep = _dp_worker_run(
            device_ids, conv_list, max_model_len, policy_model_path, tp, batch_size
        )
        queue.put((shard_idx, raw, extr, keep))
    except Exception:
        import traceback as _tb
        _tb.print_exc()
        queue.put((shard_idx, [], [], []))


def run_policy_batch(
    policy_mllm,
    conv_list: List[list],
    sampling_params,
    processor,
    tokenizer,
    max_model_len: Optional[int],
    batch_size: int = 8192,
    dp_device_groups: Optional[List[List[int]]] = None,
    policy_model_path: Optional[str] = None,
    tp: int = 1,
) -> Tuple[List[Optional[str]], List[Optional[str]], List[bool]]:
    """支持两种模式：
    - 单引擎批推理（原逻辑）
    - 多进程多卡数据并行（dp_device_groups 不为空时）：每个进程独立引擎 + GPU 组
    """

    # DP 路径
    if dp_device_groups and len(dp_device_groups) > 1:
        if not policy_model_path:
            raise ValueError("policy_model_path must be provided for data-parallel inference")
        num_workers = len(dp_device_groups)
        n = len(conv_list)
        if n == 0:
            return [], [], []

        # contiguous split into num_workers shards
        shard_sizes = [(n + i) // num_workers for i in range(num_workers)]
        total_assigned = sum(shard_sizes)
        if total_assigned != n:
            shard_sizes[0] += (n - total_assigned)
        shards = []
        idx = 0
        for sz in shard_sizes:
            shards.append(conv_list[idx : idx + sz])
            idx += sz

        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        procs: List[mp.Process] = []
        # 启动进程（非 daemon），每个进程独立引擎
        for shard_idx, (devs, shard) in enumerate(zip(dp_device_groups, shards)):
            if not shard:
                # 直接放空结果
                queue.put((shard_idx, [], [], []))
                continue
            p = ctx.Process(
                target=_dp_worker_entry,
                args=(queue, devs, shard, max_model_len, policy_model_path, tp, batch_size, shard_idx),
            )
            p.daemon = False
            p.start()
            procs.append(p)

        # 收集结果
        received = 0
        outputs = [None] * num_workers  # type: ignore
        while received < num_workers:
            shard_idx, raw, extr, keep = queue.get()
            outputs[shard_idx] = (raw, extr, keep)
            received += 1

        for p in procs:
            p.join()

        # 拼接
        raw_all: List[Optional[str]] = []
        extr_all: List[Optional[str]] = []
        keep_all: List[bool] = []
        for out in outputs:
            if out is None:
                continue
            raw, extr, keep = out
            raw_all.extend(raw)
            extr_all.extend(extr)
            keep_all.extend(keep)
        return raw_all, extr_all, keep_all

    # 单引擎路径（原逻辑）

    if vllm_mllm_process_batch_from_messages is None:
        raise RuntimeError(
            "vllm_mllm_process_batch_from_messages 未导入，无法运行 policy 推理。"
        )

    # 结果占位
    raw_outs_all: List[Optional[str]] = []
    extr_outs_all: List[Optional[str]] = []
    keep_mask_all: List[bool] = []

    # 按 batch_size 迭代
    for start in range(0, len(conv_list), batch_size):
        sub_convs = conv_list[start : start + batch_size]

        # ---------- 1. 构建输入 ----------
        inputs = vllm_mllm_process_batch_from_messages(sub_convs, processor)

        # ---------- 2. 计算 token 长度并过滤 ----------
        keep_mask = [True] * len(inputs)
        if max_model_len is not None and count_qwen_vl_tokens is not None:
            tok_lens = count_qwen_vl_tokens(inputs, tokenizer, processor)
            for i, L in enumerate(tok_lens):
                if L > max_model_len:
                    keep_mask[i] = False

        # ---------- 3. 调用 vLLM ----------
        # 仅保留未被过滤的样本传入模型，减少无谓推理
        kept_inputs = [inp for inp, k in zip(inputs, keep_mask) if k]
        if kept_inputs:
            outputs = policy_mllm.generate(
                kept_inputs, sampling_params=sampling_params, use_tqdm=True
            )
        else:
            outputs = []

        # ---------- 4. 组装子批次结果 ----------
        raw_outs, extr_outs = [], []
        j = 0
        for k in keep_mask:
            if not k:
                raw_outs.append(None)
                extr_outs.append(None)
            else:
                o = outputs[j]
                j += 1
                text = o.outputs[0].text.strip() if o.outputs else ""
                raw_outs.append(text)
                extr_outs.append(extract_boxed_answer(text))

        # ---------- 5. 追加到总结果 ----------
        raw_outs_all.extend(raw_outs)
        extr_outs_all.extend(extr_outs)
        keep_mask_all.extend(keep_mask)

    return raw_outs_all, extr_outs_all, keep_mask_all


# ============================================================
# 主流程
# ============================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, required=True, help="数据集名称")
    parser.add_argument(
        "--policy-model-path",
        type=str,
        required=True,
        help="policy_mllm 模型路径 (如 Qwen2.5-VL-7B-Instruct)",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="0,1,2,3",
        help="逗号分隔 CUDA GPU ID，例如 0,1,2,3",
    )
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "auto"],
        help="模型精度",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="mp",
        choices=["mp", "ray"],
        help="vLLM 分布式后端；建议 mp 单节点稳定",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="./created_dataset/filtered_data/",
        help="输出根目录",
    )
    parser.add_argument(
        "--start-id", type=int, default=0, help="Stage1 样本起始 id (避免断点重复)"
    )
    parser.add_argument(
        "--limit", type=int, default=-1, help="最多处理多少条；-1 表示全部"
    )
    parser.add_argument("--judge_llm_dir", type=str, default=None)
    parser.add_argument("--judge_llm_tensor_parallel_size", type=int, default=2)
    parser.add_argument(
        "--policy_mllm_tensor_parallel_size",
        type=int,
        default=2,
        help="policy_mllm 模型的 tensor parallel size",
    )
    parser.add_argument("--judge_mode", choices=["llm", "quick", "data_spec", "api"], nargs="+")
    # API judging related
    parser.add_argument("--api_name", type=str, default=None, choices=["gemini-2.5-pro", "deepseek-chat"], help="External API judge model name")
    parser.add_argument("--api_max_workers", type=int, default=32, help="Parallel workers for API judging")
    parser.add_argument("--api_temperature", type=float, default=0.0, help="Temperature for API judge calls")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path for dataset, if not set, use the default from DEFAULT_DATASETS")
    parser.add_argument("--dataset_images_root", type=str, default=None, help="Root path for dataset images, if not set, use the default from DEFAULT_DATASETS")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    # os.environ.setdefault("VLLM_TENSOR_PARALLEL_SIZE", str(args.tensor_parallel_size))

    # 数据集路径
    ds_cfg = DEFAULT_DATASETS[args.dataset_name]
    dataset_path = Path(ds_cfg["dataset_path"])
    dataset_images_root = Path(ds_cfg["dataset_images_root"])
    if args.dataset_path is not None:
        dataset_path = Path(args.dataset_path)
    if args.dataset_images_root is not None:
        dataset_images_root = Path(args.dataset_images_root)


    # 输出路径
    out_root = Path(args.out_root) / args.dataset_name
    out_root.mkdir(parents=True, exist_ok=True)
    out_img_dir = out_root / "images"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_root / "stage1_policy_out.jsonl"

    # 加载模型 processor/tokenizer
    print("[Load tokenizer/processor]", args.policy_model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.policy_model_path, trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(args.policy_model_path)

    # 加载 policy 模型
    # 设备划分：根据 devices 与 tp 自动计算数据并行分组
    devices_list = [int(x) for x in args.devices.split(",") if x.strip() != ""]
    tp = args.policy_mllm_tensor_parallel_size
    dp_degree = len(devices_list) // tp if tp > 0 else 0
    use_dp = dp_degree > 1
    dp_groups = [devices_list[i * tp : (i + 1) * tp] for i in range(dp_degree)] if use_dp else []

    if not use_dp:
        print("[Init policy_mllm]")
        policy_mllm, sampling_params = vllm_mllm_init(
            args.policy_model_path, tp=args.policy_mllm_tensor_parallel_size
        )

    # ------------------ 加载原始数据 ------------------
    print(f"[Load dataset] {args.dataset_name} -> {dataset_path}")
    if ds_cfg["dataset_path"].endswith(".jsonl"):
        # jsonl
        samples = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        if "CoM" in args.dataset_name:
            samples = samples[0]
    elif ds_cfg["dataset_path"].endswith(".json"):
        with open(dataset_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        if "VTS" in args.dataset_name:
            total = len(samples)
            quarter_cnt = total // 4
            split_num = int(args.dataset_name.split("_")[-1])
            if split_num < 1 or split_num > 4:
                raise ValueError("split_num must be between 1 and 4")
            start_idx = (split_num - 1) * quarter_cnt
            end_idx = split_num * quarter_cnt
            samples = samples[start_idx:end_idx]
    else:
        if "Zebra_CoT" in args.dataset_name:
            data_root = ds_cfg["dataset_path"]
            samples = load_dataset(
                "parquet",  # 数据格式
                data_files={"train": f"{data_root}/train-*.parquet"},
                split="train",
                cache_dir=".cache",  # 可写可不写；默认还是 ~/.cache/huggingface
            )

    if args.limit > 0:
        if isinstance(samples, list):
            samples = samples[: args.limit]
        elif isinstance(samples, datasets.arrow_dataset.Dataset):
            # datasets.Dataset 需要用 .select() 方法
            samples = samples.select(range(args.limit))

    # ------------------ 解析所有样本 ------------------
    parsed_recs = []  # 暂存解析后的样本（含 PIL）
    convs = []  # policy 推理输入
    questions = []
    for idx, samp in enumerate(tqdm(samples, desc="Parse samples")):
        rec = None
        if args.dataset_name == "PixelReasoner":
            rec = parse_pixelreasoner(samp, dataset_images_root)
        elif "CoM" in args.dataset_name:
            rec = parse_com(
                samp,
                dataset_images_root,
                w_mathvista=("w_MathVista" in args.dataset_name),
            )
        elif args.dataset_name == "CoF":
            rec = parse_cof(samp, dataset_images_root)
        elif args.dataset_name == "ReFocus":
            rec = parse_refocus(samp, dataset_images_root)
        elif args.dataset_name == "Visual_CoT":
            rec = parse_visual_cot_all(
                samp, dataset_images_root, args.dataset_name, idx
            )
        elif args.dataset_name == "Visual_CoT_v7w":
            rec = parse_visual_cot_w_choices(
                samp, dataset_images_root, args.dataset_name, idx
            )
        elif args.dataset_name == "Visual_CoT_gqa":
            rec = parse_visual_cot_wo_choices(
                samp, dataset_images_root, args.dataset_name, idx
            )
        elif "Zebra_CoT" in args.dataset_name:
            rec = parse_zebra_cot(samp, dataset_images_root)
        elif "VTS" in args.dataset_name:
            rec = parse_vts(samp, dataset_images_root)
        if rec is None:
            continue
        rec["orig_idx"] = idx
        # conversation for policy
        conv = build_policy_conversation(rec["question"], rec["main_image"])
        parsed_recs.append(rec)
        convs.append(conv)
        questions.append(rec["question"])
    print(f"[Parse] success {len(parsed_recs)} samples (of {len(samples)})")
    if not parsed_recs:
        print("[Exit] no valid samples")
        return

    # ------------------ policy 批推理 ------------------
    print("[Policy inference] running batch...")
    if use_dp:
        # DP 模式：每个 worker 进程独立引擎
        raw_outs, extr_outs, keep_mask = run_policy_batch(
            None,
            convs,
            None,
            None,
            None,
            max_model_len=args.max_model_len,
            dp_device_groups=dp_groups,
            policy_model_path=args.policy_model_path,
            tp=tp,
        )
    else:
        raw_outs, extr_outs, keep_mask = run_policy_batch(
            policy_mllm,
            convs,
            sampling_params,
            processor,
            tokenizer,
            max_model_len=args.max_model_len,
        )
        vllm_kill_model(policy_mllm)  #
    # ------------------ 判分并写出 ------------------
    # ground truth list
    gts = [r["gt_answer_text"] for r in parsed_recs]
    choices_list = [r["gt_choices"] for r in parsed_recs]

    
    # 顺序判分：quick -> data_spec -> llm，仅对上一阶段判错的继续判
    judged = sequential_judge_predictions(
        extr_outs,
        gts,
        judge_mode=args.judge_mode,
        dataset_name=args.dataset_name,
        judge_llm_dir=args.judge_llm_dir,
        judge_llm_tensor_parallel_size=args.judge_llm_tensor_parallel_size,
        questions=questions,
        api_name=args.api_name,
        api_max_workers=args.api_max_workers,
        api_kwargs={"temperature": args.api_temperature} if args.api_temperature is not None else None,
    )
        
    sid = args.start_id
    kept = 0
    skipped_policy_correct = 0
    skipped_overlen = 0
    skipped_parse = len(samples) - len(parsed_recs)

    with open(out_jsonl, "w", encoding="utf-8") as fw:
        for rec, raw_pred, extr_pred, keep, ok in zip(
            parsed_recs, raw_outs, extr_outs, keep_mask, judged
        ):
            if not keep:
                skipped_overlen += 1
                continue
            if ok:  # policy 正确样本不保留
                skipped_policy_correct += 1
                continue
            # 保存图像
            paths = save_images_for_sample(
                sid, out_img_dir, rec["main_image"], rec["helpers"]
            )
            main_path = paths[0]
            # 回填 helper 图像路径
            img_idx = 1
            for h in rec["helpers"]:
                if h["image"] is not None:
                    h["image_path"] = paths[img_idx]
                    img_idx += 1
                else:
                    h["image_path"] = None
                # 释放 PIL 以节省内存
                h["image"] = None
            # 构造 Stage1 记录
            out_item = {
                "dataset_name": args.dataset_name,
                "stage1_id": sid,
                "orig_idx": rec.get("orig_idx"),
                "qid": rec.get("qid"),
                "question": rec["question"],
                "gt_choices": rec["gt_choices"],
                "gt_answer_text": rec["gt_answer_text"],
                "image_main": main_path,
                "helpers": rec[
                    "helpers"
                ],  # list[{'step_idx','text','image_path','type'}]
                "policy_pred_raw": raw_pred,
                "policy_pred_extracted": extr_pred,
                "policy_correct": False,
            }
            fw.write(json.dumps(out_item, ensure_ascii=False) + "\n")
            kept += 1
            sid += 1

    print("================ SUMMARY ================")
    print(f"dataset_name         : {args.dataset_name}")
    print(f"raw_samples          : {len(samples)}")
    print(f"parsed_ok            : {len(parsed_recs)}")
    print(f"skipped_parse_fail   : {skipped_parse}")
    print(f"skipped_token_overlen: {skipped_overlen}")
    print(f"skipped_policy_correct: {skipped_policy_correct}")
    print(f"stage1_kept (wrong)  : {kept}")
    print(f"output_jsonl         : {out_jsonl}")
    print("==========================================")


if __name__ == "__main__":
    main()
