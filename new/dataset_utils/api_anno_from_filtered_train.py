#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
api_anno_from_filtered_train.py

功能（与 api_anno_from_stage1.py 相同任务但数据来源不同）：
1) 从 filtered_train_short3000_w_metadata.json 格式读取：
   - question 取 "role": "user" 的 content 中的第一个 text；
   - cot 为 "role": "assistant" 的 content 中所有 "type":"text" 的片段按顺序拼接。
2) 先移除 cot 文本中的所有 "<observation>" 与 "</observation>" 标签（保留中间内容），
   再构建带 <STEP_0> ... <END_STEP_0> 的标注输入并送 API（使用 prompts.py 的 prompt）。
3) 解析 API 返回文本（按 <STEP_i>..<END_STEP_i> 提取），若解析失败则回退未加标签的 cot 文本。
4) 以 metadata + data 结构（参考 filtered_train_w_metadata.json）保存输出。

用法示例：
python -m dataset_utils.api_anno_from_filtered_train \
  --input-json /path/to/filtered_train_short3000_w_metadata.json \
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
    ALIGN_SYS_PROMPT_exact,
    VTS_examples,
    examples_pool_exact,
)


# ---------------- I/O ----------------
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------- 常量/工具 ----------------
STEP_START = "<STEP_{i}>"
STEP_END = "<END_STEP_{i}>"
_STEP_RE_TPL = r"<STEP_{i}>\s*(.*?)\s*<END_STEP_{i}>"


def strip_observation_tags(text: str) -> str:
    """移除 observation 标签但保留中间内容。"""
    return text.replace("<observation>", "").replace("</observation>", "")


def parse_aligned(text: str) -> List[str]:
    """解析按 <STEP_i>..<END_STEP_i> 包裹的段落，顺序返回每段内容。"""
    out: List[str] = []
    i = 0
    while True:
        m = re.search(_STEP_RE_TPL.format(i=i), text, re.S)
        if not m:
            break
        out.append(m.group(1).strip())
        i += 1
    return out


def _format_user_prompt(text: str) -> str:
    return "Now it's your turn. ## Input: " + text + "\n\n## Your output:"


def _get_prompts_for_dataset(dataset_name: str) -> Tuple[str, str]:
    """从 examples_pool_exact 获取 (sys_prompt, examples)；无法获取时回退。"""
    entry = examples_pool_exact.get(dataset_name)
    if entry is None:
        return ALIGN_SYS_PROMPT_exact, VTS_examples
    if isinstance(entry, dict):
        sys_p = entry.get("sys_prompt", ALIGN_SYS_PROMPT_exact)
        ex = entry.get("examples", "")
        if isinstance(ex, (list, tuple)):
            ex = "".join(ex)
        if not isinstance(ex, str):
            ex = str(ex)
        return sys_p, ex
    if isinstance(entry, (tuple, list)):
        return ALIGN_SYS_PROMPT_exact, "".join(map(str, entry))
    if isinstance(entry, str):
        return ALIGN_SYS_PROMPT_exact, entry
    return ALIGN_SYS_PROMPT_exact, VTS_examples


# ---------------- 从 filtered_train_* 读取并构建 API 输入 ----------------
def extract_question_and_cot_steps(sample: Dict[str, Any]) -> Tuple[str, List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """从一条记录中抽取 question、cot 步骤（assistant 每个 text 片段为一步），并保留原 user 图片，
    同时记录 assistant 的混合序列（文本占位+图片）以保留图片在原始步骤中的相对位置。

    返回: (question_text, cot_steps_no_obs_tags, user_images, assistant_seq)
    其中 assistant_seq 是一个列表，元素要么是 {"k":"text","idx":i}（第 i 个步骤占位），
    要么是 {"k":"image","value":<原 image 项>}。
    """
    # data 为对话列表
    dialogs: List[Dict[str, Any]] = sample.get("data", [])
    question = ""
    user_images: List[Dict[str, Any]] = []
    assistant_images: List[Dict[str, Any]] = []

    # 找 user
    for turn in dialogs:
        if turn.get("role") == "user":
            for item in turn.get("content", []) or []:
                if item.get("type") == "text" and not question:
                    question = item.get("text", "")
                elif item.get("type") == "image":
                    user_images.append(item)
            break  # 仅取第一个 user

    # 找 assistant 的所有文本，每个文本片段视为一个 step；同时记录图片与文本占位的顺序
    cot_steps: List[str] = []
    assistant_seq: List[Dict[str, Any]] = []
    step_counter = 0
    for turn in dialogs:
        if turn.get("role") == "assistant":
            for item in turn.get("content", []) or []:
                if item.get("type") == "text":
                    t = item.get("text", "")
                    t = strip_observation_tags(t)
                    if t:
                        cot_steps.append(t)
                        assistant_seq.append({"k": "text", "idx": step_counter})
                        step_counter += 1
                elif item.get("type") == "image":
                    assistant_seq.append({"k": "image", "value": item})
            # 只用第一个 assistant（若有多个）
            break

    return question, cot_steps, user_images, assistant_seq


def build_alignment_text_from_steps(question: str, cot_steps: List[str]) -> str:
    # 去掉题干中的多余指令（若有）
    instr1 = "\nPut your final answer within \\boxed{}. If you cannot see relevant visual information to infer the answer from the image, just output \\boxed{None} and don't guess the answer based on your knowledge."
    instr2 = "\nPut your final answer within \\boxed{}."
    if instr1 in question:
        question = question.replace(instr1, "")
    if instr2 in question:
        question = question.replace(instr2, "")

    # 将 assistant 的每个文本片段作为一个 step 包裹
    segs: List[str] = [question]
    for i, seg in enumerate(cot_steps):
        segs.append(STEP_START.format(i=i) + seg + STEP_END.format(i=i))
    return "\n".join(segs)


# ---------------- 输出对话 ----------------
def make_final_cot(sample_meta: Dict[str, Any], question: str, aligned_steps: List[str],
                   user_images: List[Dict[str, Any]], assistant_seq: List[Dict[str, Any]]):
    """按 filtered_train_w_metadata.json 的结构输出：system/user/assistant。

    - user: 保留原 user 图片 + 问题文本；
    - assistant: 用对齐后的单段文本（或回退文本），并在末尾附上原 assistant 图片。
    """
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

    # user turn
    usr_content: List[Dict[str, Any]] = []
    usr_content.extend(user_images)
    usr_content.append({"type": "text", "text": question})
    cot.append({"role": "user", "content": usr_content})

    # assistant turn
    assist_content: List[Dict[str, Any]] = []
    # 根据记录的混合序列把图片插回原位置，并用对齐后的文本填充各步
    for el in assistant_seq:
        if el.get("k") == "text":
            idx = el.get("idx", 0)
            seg = aligned_steps[idx] if 0 <= idx < len(aligned_steps) else ""
            assist_content.append({"type": "text", "text": seg})
        elif el.get("k") == "image":
            img = el.get("value")
            if isinstance(img, dict):
                assist_content.append(img)
    # 如果对齐后有多余的步骤（超过原占位数），将其追加在末尾
    used_text_slots = [el.get("idx") for el in assistant_seq if el.get("k") == "text"]
    max_used = max(used_text_slots) + 1 if used_text_slots else 0
    if len(aligned_steps) > max_used:
        for seg in aligned_steps[max_used:]:
            assist_content.append({"type": "text", "text": seg})
    cot.append({"role": "assistant", "content": assist_content})
    return cot


# ---------------- 主流程 ----------------
def main():
    pa = argparse.ArgumentParser("Annotate CoT from filtered_train_* using API")
    pa.add_argument("--input-json", required=True, help="path to filtered_train_short3000_w_metadata.json")
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

    print("[api_anno_filtered] loading ...", args.input_json)
    in_recs: List[Dict[str, Any]] = load_json(args.input_json)
    if args.max_records:
        in_recs = in_recs[: args.max_records]
    print(f"[api_anno_filtered] {len(in_recs)} samples to annotate")

    # 预提取 question/cot，并构建 API 输入文本
    api_inputs: List[str] = []
    meta_cache: List[Dict[str, Any]] = []  # 缓存必要信息用于回填输出
    for rec in in_recs:
        question, cot_steps, user_imgs, asst_seq = extract_question_and_cot_steps(rec)
        full_text = build_alignment_text_from_steps(question, cot_steps)
        api_inputs.append(full_text)
        meta_cache.append({
            "dataset_name": (rec.get("metadata") or {}).get("dataset_name", ""),
            "sample_id": (rec.get("metadata") or {}).get("sample_id"),
            "user_images": user_imgs,
            "assistant_seq": asst_seq,
            "question": question,
        })

    # 按数据集分组以拼装 sys_prompt + examples，批量调用 API
    grouped: Dict[str, List[Tuple[int, str]]] = {}
    for idx, meta in enumerate(meta_cache):
        ds = meta.get("dataset_name") or "__UNKNOWN__"
        grouped.setdefault(ds, []).append((idx, api_inputs[idx]))

    # 收集 (原索引, 文本或 None)
    aligned_pairs: List[Tuple[int, Any]] = []
    for ds_name, items in grouped.items():
        sys_p, examples = _get_prompts_for_dataset(ds_name)
        sys_prompt = f"{sys_p}\n{examples}" if examples else sys_p

        usr_all = [_format_user_prompt(t) for _, t in items]
        for i in range(0, len(usr_all), args.batch_size):
            chunk = usr_all[i : i + args.batch_size]
            idx_chunk = [items[j][0] for j in range(i, min(i + args.batch_size, len(items)))]
            print(
                f"[api_anno_filtered] calling API {args.api_model_name} | dataset={ds_name} | batch {i}-{i+len(chunk)-1}"
            )
            resps = get_api_response(
                args.api_model_name,
                sys_prompt=sys_prompt,
                user_prompts=chunk,
                temperature=0.3,
            )
            for k, r in zip(idx_chunk, resps):
                # 允许 r 为 None（例如上游异常），此时标记 None 用于后续丢弃样本
                if isinstance(r, str):
                    aligned_pairs.append((k, r.strip()))
                else:
                    aligned_pairs.append((k, None))

    aligned_pairs.sort(key=lambda x: x[0])
    aligned_texts = [t for _, t in aligned_pairs]

    # 解析并回填输出
    out_records: List[Dict[str, Any]] = []
    dropped = 0
    for i, (meta, api_text) in enumerate(zip(meta_cache, aligned_texts)):
        # 若 API 返回为 None，直接跳过该样本
        if api_text is None:
            dropped += 1
            continue
        steps = parse_aligned(api_text)
        if not steps:
            # 回退：从我们送入的文本中解析出原始的所有 STEP 段
            steps = parse_aligned(api_inputs[i])
            if not steps:
                steps = [""]
        # 保留为多段，便于在 content 中输出多条 text
        aligned_out = steps

        cot = make_final_cot(
            sample_meta=meta,
            question=meta.get("question", ""),
            aligned_steps=aligned_out,
            user_images=meta.get("user_images", []),
            assistant_seq=meta.get("assistant_seq", []),
        )

        # 规范化 sample_id 为整数
        raw_sid = meta.get("sample_id")
        if isinstance(raw_sid, int):
            sid = raw_sid
        elif isinstance(raw_sid, str):
            m = re.search(r"\d+", raw_sid)
            sid = int(m.group(0)) if m else i
        else:
            sid = i

        out_records.append(
            {
                "metadata": {
                    "dataset_name": meta.get("dataset_name", ""),
                    "sample_id": sid,
                },
                "data": cot,
            }
        )

    save_json(out_records, args.out_json)
    if dropped:
        print(f"[api_anno_filtered] dropped {dropped} samples due to API None responses")
    print(f"[api_anno_filtered] wrote {len(out_records)} records -> {args.out_json}")


if __name__ == "__main__":
    main()
