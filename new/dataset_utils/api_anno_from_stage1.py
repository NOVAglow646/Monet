#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
api_anno_from_stage1.py

功能：
1) 读取 Stage-1 的 jsonl（如 stage1_policy_out.jsonl）。
2) 将每条样本的 helpers.text 按 <STEP_i> ... <END_STEP_i> 连接为 CoT。
3) 使用 prompts.py 的对齐系统提示，调用已实现的 API（gemini / deepseek），让模型在文本中用 <observation>...</observation> 标出由图像观察得到的内容（仅限出现在 <abs_vis_token></abs_vis_token> 之后）。
4) 参考 stage3_new.py 的 parse_aligned 对 API 返回内容进行解析，对齐回每个 step 文本；若解析失败则回退原文本。
5) 以形如 filtered_train_w_metadata.json 的结构保存输出，每条记录包含 metadata 和 data；其中 metadata 至少包含 dataset_name，data 为对话 COT（system/user/assistant 三轮）。

用法示例：
python -m dataset_utils.api_anno_from_stage1 \
  --stage1 /path/to/stage1_policy_out.jsonl \
  --out-json /path/to/filtered_train_w_metadata.json \
  --api_model_name deepseek-chat \
  --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple

from new.AAA_vllm_toolkit.api import get_api_response  # 直接复用现有 API 封装
from new.dataset_utils.prompts import (
    ALIGN_SYS_PROMPT_w_boxed,
    VTS_examples,
    examples_pool_exact,
)


# ---------------- I/O ----------------
def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            return json.load(f)
        return [json.loads(l) for l in f if l.strip()]


def save_json(data, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------- 构建 / 解析 ----------------
STEP_START = "<STEP_{i}>"
STEP_END = "<END_STEP_{i}>"
_STEP_RE_TPL = r"<STEP_{i}>\s*(.*?)\s*<END_STEP_{i}>"


def build_alignment_text(sample: Dict[str, Any]) -> Tuple[str, int]:
    instruction = "\nPut your final answer within \\boxed{}. If you cannot see relevant visual information to infer the answer from the image, just output \\boxed{None} and don't guess the answer based on your knowledge."
    question = sample['question'].replace(instruction, '')
    instruction = "\nPut your final answer within \\boxed{}."
    question = question.replace(instruction, '')
    segs: List[str] = [question]
    for i, h in enumerate(sample.get("helpers", [])):
        segs.append(STEP_START.format(i=i) + h.get("text", "") + STEP_END.format(i=i))
    return "\n".join(segs), len(segs)


def parse_aligned(text: str) -> List[str]:
    """与 stage3_new.py 中保持一致的解析逻辑。"""
    out: List[str] = []
    i = 0
    while True:
        m = re.search(_STEP_RE_TPL.format(i=i), text, re.S)
        if not m:
            break
        out.append(m.group(1).strip())
        i += 1
    return out


# ---------------- Prompt 组装 ----------------
def _format_user_prompt(text: str) -> str:
    return "Now it's your turn. ## Input: " + text + "\n\n## Your output:"


def _get_prompts_for_dataset(dataset_name: str) -> Tuple[str, str]:
    """Return (sys_prompt, examples) from examples_pool_exact by dataset name.

    Fallback to ALIGN_SYS_PROMPT_w_boxed + VTS_examples when not found or malformed.
    """
    entry = examples_pool_exact.get(dataset_name)
    if entry is None:
        return ALIGN_SYS_PROMPT_w_boxed, VTS_examples

    # Common case: dict with keys 'sys_prompt' and 'examples'
    if isinstance(entry, dict):
        sys_p = entry.get("sys_prompt", ALIGN_SYS_PROMPT_w_boxed)
        ex = entry.get("examples", "")
        # Some entries might store examples as tuple/list, join them
        if isinstance(ex, (list, tuple)):
            ex = "".join(ex)
        if not isinstance(ex, str):
            ex = str(ex)
        return sys_p, ex

    # If entry is a raw string or tuple of examples without explicit sys_prompt
    if isinstance(entry, (tuple, list)):
        examples = "".join(map(str, entry))
        return ALIGN_SYS_PROMPT_w_boxed, examples
    if isinstance(entry, str):
        return ALIGN_SYS_PROMPT_w_boxed, entry

    # Unknown type -> fallback
    return ALIGN_SYS_PROMPT_w_boxed, VTS_examples


# ---------------- COT 构建 ----------------
def make_final_cot(s1_rec: Dict[str, Any], aligned_steps: List[str]) -> List[Dict[str, Any]]:
    """构造与训练兼容的对话格式：system / user / assistant。

    - system：说明可生成抽象视觉 token；
    - user：包含主图与问题；
    - assistant：顺序给出每条对齐后的 step 文本；并按原 helpers 顺序附上可用的辅助图。
    """
    # system
    cot: List[Dict[str, Any]] = [
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

    # user
    usr_content: List[Dict[str, Any]] = []
    if s1_rec.get("image_main"):
        usr_content.append({"type": "image", "image_file_name": s1_rec["image_main"]})
    usr_content.append({"type": "text", "text": s1_rec.get("question", "")})
    cot.append({"role": "user", "content": usr_content})

    # assistant（文本+可能的中间图）
    assist_content: List[Dict[str, Any]] = []
    helpers = s1_rec.get("helpers", [])
    for txt, h in zip(aligned_steps, helpers):
        assist_content.append({"type": "text", "text": txt})
        ip = h.get("image_path")
        if ip:
            assist_content.append({"type": "image", "image_file_name": ip})
    cot.append({"role": "assistant", "content": assist_content})
    return cot


# ---------------- 主流程 ----------------
def main():
    pa = argparse.ArgumentParser("Annotate CoT from Stage-1 using API")
    pa.add_argument("--stage1", required=True, help="path to stage1_policy_out.jsonl")
    pa.add_argument("--out-json", required=True, help="path to save output json")
    pa.add_argument(
        "--api_model_name",
        required=True,
        choices=["gemini-2.5-pro", "deepseek-chat", "deepseek-reasoner"],
        help="which API to use (already implemented in AAA_vllm_toolkit/api.py)",
    )
    pa.add_argument("--batch-size", type=int, default=10000)
    pa.add_argument("--max-records", type=int, default=None)
    args = pa.parse_args()

    # 1) 读取 Stage-1
    print("[api_anno] loading Stage-1 ...", args.stage1)
    s1_recs: List[Dict[str, Any]] = load_jsonl(args.stage1)
    if args.max_records:
        s1_recs = s1_recs[: args.max_records]
    print(f"[api_anno] {len(s1_recs)} samples to annotate")

    # 2) 构造带标记的输入文本
    full_texts: List[str] = []
    for s1 in s1_recs:
        ft, _ = build_alignment_text(s1)
        full_texts.append(ft)

    # 3) 按数据集分组构建 prompts（使用 examples_pool_exact）并调用 API
    # 分组，保留原顺序索引，便于还原
    grouped: Dict[str, List[Tuple[int, str]]] = {}
    for idx, s1 in enumerate(s1_recs):
        ds = s1.get("dataset_name", "") or "__UNKNOWN__"
        grouped.setdefault(ds, []).append((idx, full_texts[idx]))

    aligned_pairs: List[Tuple[int, str]] = []  # (original_index, aligned_text)
    for ds_name, items in grouped.items():
        sys_p, examples = _get_prompts_for_dataset(ds_name)
        sys_prompt = f"{sys_p}\n{examples}" if examples else sys_p

        # 分批调用
        usr_all = [_format_user_prompt(t) for _, t in items]
        for i in range(0, len(usr_all), args.batch_size):
            chunk = usr_all[i : i + args.batch_size]
            idx_chunk = [items[j][0] for j in range(i, min(i + args.batch_size, len(items)))]
            print(
                f"[api_anno] calling API {args.api_model_name} | dataset={ds_name} | batch {i}-{i+len(chunk)-1}"
            )
            resps = get_api_response(
                args.api_model_name,
                sys_prompt=sys_prompt,
                user_prompts=chunk,
                temperature=0.3,
            )
            # 归并回原索引
            for k, r in zip(idx_chunk, resps):
                aligned_pairs.append((k, r.strip() if isinstance(r, str) else ""))

    # 按原顺序重排
    aligned_pairs.sort(key=lambda x: x[0])
    aligned_texts = [t for _, t in aligned_pairs]

    assert len(aligned_texts) == len(s1_recs), (
        f"response len mismatch: {len(aligned_texts)} vs {len(s1_recs)}"
    )

    # 5) 解析回对齐 steps，并生成最终 COT
    out_records: List[Dict[str, Any]] = []
    for rec_idx, (s1, al_txt) in enumerate(zip(s1_recs, aligned_texts)):
        steps_aligned = parse_aligned(al_txt)
        if len(steps_aligned) != len(s1.get("helpers", [])):
            # 回退到原 helpers 文本
            steps_aligned = [h.get("text", "") for h in s1.get("helpers", [])]

        cot = make_final_cot(s1, steps_aligned)

        # 保存为 metadata + data 的结构；metadata 至少含 dataset_name
        # 补充 sample_id：
        #  - 优先使用原始样本索引 orig_idx；
        #  - 否则回退到 stage1_id；
        #  - 再否则使用当前顺序索引（带前缀 fallback-）。
        sid = s1.get("orig_idx")
        if sid is None:
            sid = s1.get("stage1_id")
        if sid is None:
            sid = f"fallback-{rec_idx}"

        out_records.append(
            {
                "metadata": {
                    "dataset_name": s1.get("dataset_name", ""),
                    "sample_id": sid,
                },
                "data": cot,
            }
        )

    # 6) 写文件
    save_json(out_records, args.out_json)
    print(f"[api_anno] wrote {len(out_records)} records -> {args.out_json}")


if __name__ == "__main__":
    main()
