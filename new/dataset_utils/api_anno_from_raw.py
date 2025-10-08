#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
conda activate mirage
cd /data1/qxwang/codes/abstract-visual-token/new
python -m dataset_utils.api_anno_from_raw \
    --dataset-name Zebra_CoT_arc_agi \
    --limit 5 \
    --api_model_name deepseek-chat

conda activate mirage
cd /data1/qxwang/codes/abstract-visual-token/new
python -m dataset_utils.api_anno_from_raw \
    --dataset-name Zebra_CoT_checker \
    --limit 5 \
    --api_model_name deepseek-chat

conda activate mirage
cd /data1/qxwang/codes/abstract-visual-token/new
python -m dataset_utils.api_anno_from_raw \
    --dataset-name Zebra_CoT_connect_four \
    --limit 5 \
    --api_model_name deepseek-chat

conda activate mirage
cd /data1/qxwang/codes/abstract-visual-token/new
python -m dataset_utils.api_anno_from_raw \
    --dataset-name Zebra_CoT_rpm \
    --limit 5 \
    --api_model_name deepseek-chat

conda activate mirage
cd /data1/qxwang/codes/abstract-visual-token/new
python -m dataset_utils.api_anno_from_raw \
    --dataset-name Zebra_CoT_tetris \
    --limit 5 \
    --api_model_name deepseek-chat

"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import random
import datasets
from dataset_utils.prompts import examples_pool_exact  
import concurrent.futures as cf
from AAA_vllm_toolkit.api import get_api_response          # type: ignore
from dataset_utils.prompts import examples_pool_exact            # type: ignore
import time
# ------------------------------
# AAA 本地工具 (按用户环境)
# ------------------------------
# 如果这些 import 失败，会 fallback 到简化实现。


from transformers import AutoProcessor, AutoTokenizer  # trust_remote_code needed

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
    },
    "Zebra_CoT_arc_agi": {
        "dataset_path": "/ytech_m2v5_hdd/workspace/kling_mm/Datasets/Zebra-CoT/Visual Logic & Strategic Games - ARC-AGI",
        "dataset_images_root": "",
    },
    "Zebra_CoT_checkers": {
        "dataset_path": "/ytech_m2v5_hdd/workspace/kling_mm/Datasets/Zebra-CoT/Visual Logic & Strategic Games - Checkers",
        "dataset_images_root": "",
    },
    "Zebra_CoT_connect_four": {
        "dataset_path": "/ytech_m2v5_hdd/workspace/kling_mm/Datasets/Zebra-CoT/Visual Logic & Strategic Games - Connect Four",
        "dataset_images_root": "",
    },
    "Zebra_CoT_rpm": {
        "dataset_path": "/ytech_m2v5_hdd/workspace/kling_mm/Datasets/Zebra-CoT/Visual Logic & Strategic Games - RPM",
        "dataset_images_root": "",
    },
    "Zebra_CoT_tetris": {
        "dataset_path": "/ytech_m2v5_hdd/workspace/kling_mm/Datasets/Zebra-CoT/Visual Logic & Strategic Games - Tetris",
        "dataset_images_root": "",
    },
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

def extract_boxed_answer(text, debug=False):
    if text is None:
        return "Invalid prediction."
    start = text.rfind(r"\boxed{")
    if start == -1:
        return "Invalid prediction."
    end = None
    stack = []
    answer = text[start:]
    for i, c in enumerate(answer):
        if c == "{":
            stack.append(i)
        elif c == "}":
            start = stack.pop()  # \boxed start{
            if len(stack) == 0:
                end = i  # \boxed end}
                break
    if end is None and debug:
        print("Extract boxed answer: brack not closing, return None", answer)
        return "Invalid prediction."
    return answer[start + 1 : end]

def extract_html_answer(text: str):
    if text is None:
        return None
    start = text.find("<answer>")
    if start == -1:
        return None
    end = text.find("</answer>")
    if end == -1:
        return None
    return text[start + len("<answer>"):end].strip()

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
        "question":add_boxed_instruction(question, allow_None=False, mode="normal"),
        "gt_choices": None,
        "gt_answer_text": gt_answer_text,
        "main_image": main_img,
        "helpers": helpers,
    }


def parse_zebra_cot(
    sample: Dict[str, Any], dataset_images_root: Path
) -> Optional[Dict[str, Any]]:
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
            r'<image_start>\[problem_image_\d+\]<image_end>'  # 图片标签
        )
        txt = re.sub(pattern, '', txt)  # 直接整段清空
        txt = re.sub(r'<image_start>\[problem_image_\d+\]<image_end>', '', txt)
        return txt
    
    print(sample)
    if not sample.get("problem_image_2", None):
        main_img = sample["problem_image_1"]
        question = _remove_question_img_pad(sample["Question"])
    else:
        i = 1
        main_img = []
        while sample.get(f"problem_image_{i}", None):
            main_img.append(sample[f"problem_image_{i}"])
            i += 1
        question = _replace_img_pad(sample["Question"])

    if not main_img:
        print("Empty main image, return None")
        return None

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
    helpers.append(
        {
            "step_idx": len(steps),
            "text": "The final answer is: \\boxed{" + gt_answer_text + "}",
            "image": None,
            "type": "any"
        }
    )
    

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
    sid: int, out_img_dir: Path, dataset_name, main_img, helpers: List[Dict[str, Any]]
):
    os.makedirs(out_img_dir / dataset_name, exist_ok=True)
    paths = []
    # main
    
    if isinstance(main_img, list):
        main_path = []
        for i, img in enumerate(main_img):
            save_p = out_img_dir / f"{sid}_0_{i}.jpg"
            p = os.path.join(dataset_name, f"{sid}_0_{i}.jpg")
            main_path.append(p)
            img.save(save_p)
    else:
        save_path = out_img_dir / f"{sid}_0.jpg"
        main_img.save(save_path)
        main_img = os.path.join(dataset_name, f"{sid}_0.jpg")
    paths.append(main_path)
    # helpers
    idx = 1
    for h in helpers:
        img = h.get("image")
        if img is None:
            # paths.append(None)
            continue
        save_p = out_img_dir / f"{sid}_{idx}.jpg"
        img.save(save_p)
        p = os.path.join(dataset_name, f"{sid}_{idx}.jpg")
        paths.append(p)
        idx += 1
    return paths  # [main_path, helper1_path_orNone, ...]


# ---------- 标签模板 ----------
STEP_START = "<STEP_{i}>"
STEP_END   = "<END_STEP_{i}>"

# ---------- I/O ----------
def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1); f.seek(0)
        if head == "[":
            return json.load(f)
        return [json.loads(l) for l in f if l.strip()]

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ---------- 构建 / 解析 ----------
_STEP_RE_TPL = r"<STEP_{i}>\s*(.*?)\s*<END_STEP_{i}>"

def build_alignment_text(sample: Dict[str, Any]) -> Tuple[str, int]:
    segs = []
    for i, h in enumerate(sample["helpers"]):
        segs.append(STEP_START.format(i=i) + h.get("text", "") + STEP_END.format(i=i))
    return "\n".join(segs), len(segs)

def parse_aligned(text: str) -> List[str]:
    out, i = [], 0
    while True:
        m = re.search(_STEP_RE_TPL.format(i=i), text, re.S)
        if not m: break
        out.append(m.group(1).strip())
        i += 1
    return out

# ---------- prompt ----------
def _prompts(inputs: List[str], ds_name: str):
    sys_part = examples_pool_exact[ds_name]["sys_prompt"] + examples_pool_exact[ds_name]["examples"]
    return sys_part, [
        "Now it's your turn. ## Input: " + t + "\n\n## Your output:" for t in inputs
    ]

def _api_chunk_call_pack(args: Tuple[str, str, List[str], float]) -> List[str]:
    """顶层函数，供进程池调用，避免局部函数无法被 pickle 的问题；内部包含 3 次重试。"""
    api_model, sys_prompt, sub_prompts, api_temperature = args
    return _api_call_with_retries(api_model, sys_prompt, sub_prompts, api_temperature)

def _api_call_with_retries(
    api_model: str,
    sys_prompt: str,
    user_prompts: List[str],
    temperature: float,
    max_retries: int = 3,
    base_backoff_sec: float = 1.0,
) -> List[str]:
    """调用外部 API（DeepSeek / Gemini）带重试。

    策略：最多重试 max_retries 次，指数退避 base_backoff_sec * 2**attempt。
    若最后一次仍失败，则抛出异常（由上层捕获或终止）。
    """
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            return get_api_response(api_model, sys_prompt, user_prompts, temperature=temperature)
        except Exception as e:
            last_err = e
            if attempt == max_retries - 1:
                break
            # 简单指数退避 + 轻微抖动
            delay = base_backoff_sec * (2 ** attempt)
            try:
                time.sleep(delay)
            except Exception:
                pass
    # 走到这里说明多次失败，抛出最后一次异常
    raise last_err if last_err else RuntimeError("Unknown API error with retries exhausted")


def add_args(parser):
    parser.add_argument("--dataset-name", type=str, required=True, help="数据集名称")
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
    parser.add_argument("--api_model_name", type=str, default="gemini-2.5-pro")
    return parser

def make_final_cot(s1_rec: Dict[str, Any], aligned_steps: List[str]):
    helpers = s1_rec["helpers"]
    cot = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a helpful assistant. You can generate abstract visual "
                        "tokens that represent a cropped image region or images with auxiliary "
                        "information like lines, bounding boxes, etc. When you decide to generate "
                        "abstract visual tokens, put them in <abs_vis_token>...</abs_vis_token>."
                    ),
                }
            ],
        }
    ]

    # user turn
    usr_content = []
    usr_content.append({"type": "text", "text": s1_rec["question"]})
    main_image = s1_rec.get("main_image", None)
    if main_image is not None:
        if isinstance(main_image, list):
            for main_img in main_image:
                usr_content.append({"type": "image", "image_file_name": main_img})
        else:
            usr_content.append({"type": "image", "image_file_name": main_image})
    
    cot.append({"role": "user", "content": usr_content})

    # assistant turn (aligned helpers)
    assist_content = []
    for txt, h in zip(aligned_steps, helpers):
        assist_content.append({"type": "text", "text": txt})
        ip = h.get("image_path")
        if ip:
            assist_content.append({"type": "image", "image_file_name": ip})
    cot.append({"role": "assistant", "content": assist_content})
    return cot



def main():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()

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
    # 根据 args.dataset_path 的结尾决定输出文件名：
    # 若以 _<i>.json 结尾，则输出为 stage1_policy_out_<i>.jsonl；否则为 stage1_policy_out.jsonl
    suffix_num = None
    if args.dataset_path is not None:
        m = re.search(r"_(\d+)\.json$", args.dataset_path)
        if m:
            suffix_num = m.group(1)
    if suffix_num is not None:
        out_json = out_root / f"raw_train_w_obs_{suffix_num}.json"
    else:
        out_json = out_root / "raw_train_w_obs.json"

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
    questions = []
    usr_prompts = []
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

        parsed_recs.append(rec)
        usr_prompts.append(build_alignment_text(rec)[0])
        questions.append(rec["question"])
    print(f"[Parse] success {len(parsed_recs)} samples (of {len(samples)})")
    
    dataset_name = args.dataset_name
    api_max_workers = args.api_max_workers
    api_model = args.api_model_name
    api_temperature = 0.1
    sys_prompt, _ = _prompts(["dummy"], dataset_name)
    #print(usr_prompts)
    _, usr_prompts = _prompts(usr_prompts, dataset_name)
    outs: List[str] = []

    if api_max_workers and api_max_workers > 1 and len(usr_prompts) > 1:
        # 均匀切分到 workers 数量（不超过样本数）
        workers = min(api_max_workers, len(usr_prompts))
        # 生成切片边界
        sizes = [(len(usr_prompts) + j) // workers for j in range(workers)]
        total = sum(sizes)
        if total != len(usr_prompts):
            sizes[0] += (len(usr_prompts) - total)
        slices = []
        s = 0
        for sz in sizes:
            slices.append(usr_prompts[s : s + sz])
            s += sz

        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            # 用顶层函数以避免 pickling 问题
            futs = [
                ex.submit(_api_chunk_call_pack, (api_model, sys_prompt, sub, api_temperature))
                for sub in slices if sub
            ]
            # 保序收集：按切片顺序扩展
            for fut in futs:
                resps = fut.result()
                outs.extend(r.strip() if isinstance(r, str) else "" for r in resps)

    
    sid = args.start_id
    finals = []
    for s1_rec, txt_al in zip(parsed_recs, outs):
        steps_al = parse_aligned(txt_al)
        # 若解析失败，回退到原 helper.text
        if len(steps_al) != len(s1_rec["helpers"]):
            steps_al = [h.get("text", "") for h in s1_rec["helpers"]]
        
        paths = save_images_for_sample(
                sid, out_img_dir, dataset_name, s1_rec["main_image"], s1_rec["helpers"]
            )
        main_path = paths[0]
        s1_rec["main_image"] = main_path
        img_idx = 1
        for h in s1_rec["helpers"]:
            if h["image"] is not None:
                h["image_path"] = paths[img_idx]
                img_idx += 1
            else:
                h["image_path"] = None
            # 释放 PIL 以节省内存
            h["image"] = None
        print(rec)
        finals.append(make_final_cot(s1_rec, steps_al))

        sid += 1

    save_json(finals, out_json)
    print(f"[Stage-3] wrote {len(finals)} records -> {out_json}")


if __name__ == "__main__":
    main()