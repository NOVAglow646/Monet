#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
api_anno_from_stage2.py

功能：
1) 读取形如 stage2_strong_out.jsonl 的样本（见附件示例）。
2) 将每条样本的 helpers.text 以 <STEP_i> ... <END_STEP_i> 包裹串联为输入文本（前面拼接问题，去掉固定的 \boxed 指令后缀）。
3) 复用 new.AAA_vllm_toolkit.api 中的 get_api_response 与 prompts（ALIGN_SYS_PROMPT_w_boxed / examples_pool_exact），
   让模型在文本中为由图像观察得到的内容（仅限出现在 <abs_vis_token></abs_vis_token> 之后）添加 <observation>...</observation> 标注。
4) 解析 API 返回内容中各 <STEP_i>... 片段，对齐回每步文本；若解析失败则回退原 helpers 文本。
5) 输出为一个 JSON 列表：每条包含基本标识、原始 API 文本 aligned_raw、解析后的 aligned_steps，以及 helpers_annotated（按原顺序更新后的每步文本与图）。

用法示例：
python -m new.dataset_utils.api_anno_from_stage2 \
  --stage2 /path/to/stage2_strong_out.jsonl \
  --out-json /path/to/stage2_aligned.json \
  --api_model_name deepseek-chat \
  --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple

from new.AAA_vllm_toolkit.api import get_api_response
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
    # ---------------- COT 构建（与 stage1 脚本一致） ----------------
    def make_final_cot(s2_rec: Dict[str, Any], aligned_steps: List[str]) -> List[Dict[str, Any]]:
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
        if s2_rec.get("image_main"):
            usr_content.append({"type": "image", "image_file_name": s2_rec["image_main"]})
        # 使用去除尾注后的 question 文本
        usr_q = _strip_boxed_instruction(s2_rec.get("question", ""))
        if usr_q:
            usr_content.append({"type": "text", "text": usr_q})
        cot.append({"role": "user", "content": usr_content})

        # assistant（文本+可能的中间图）
        assist_content: List[Dict[str, Any]] = []
        helpers = s2_rec.get("helpers", [])
        for txt, h in zip(aligned_steps, helpers):
            assist_content.append({"type": "text", "text": txt})
            ip = h.get("image_path")
            if ip:
                assist_content.append({"type": "image", "image_file_name": ip})
        cot.append({"role": "assistant", "content": assist_content})
        return cot

        return [json.loads(l) for l in f if l.strip()]


def save_json(data, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------- 构建 / 解析 ----------------
STEP_START = "<STEP_{i}>"
STEP_END = "<END_STEP_{i}>"
_STEP_RE_TPL = r"<STEP_{i}>\s*(.*?)\s*<END_STEP_{i}>"


def _strip_boxed_instruction(q: str) -> str:
    """去除常见的 \boxed 指令提示，最大程度兼容 stage1/2 的题干模板。"""
    if not isinstance(q, str):
        return ""
    # 两种常见形式
    q = q.replace(
        "\nPut your final answer within \\boxed{}. If you cannot see relevant visual information to infer the answer from the image, just output \\boxed{None} and don't guess the answer based on your knowledge.",
        "",
    )
    q = q.replace("\nPut your final answer within \\boxed{}.", "")
    return q


def build_alignment_text(sample: Dict[str, Any]) -> Tuple[str, int]:
    question = _strip_boxed_instruction(sample.get("question", ""))
    segs: List[str] = [question]
    for i, h in enumerate(sample.get("helpers", [])):
        segs.append(
            STEP_START.format(i=i) + (h.get("text", "") or "") + STEP_END.format(i=i)
        )
    return "\n".join(segs), len(segs)


def parse_aligned(text: str) -> List[str]:
    out: List[str] = []
    i = 0
    while True:
        m = re.search(_STEP_RE_TPL.format(i=i), text or "", re.S)
        if not m:
            break
        out.append(m.group(1).strip())
        i += 1
    return out


# ---------------- Prompt 组装 ----------------
def _format_user_prompt(text: str) -> str:
    return "Now it's your turn. ## Input: " + text + "\n\n## Your output:"


def _get_prompts_for_dataset(dataset_name: str) -> Tuple[str, str]:
    entry = examples_pool_exact.get(dataset_name)
    if entry is None:
        return ALIGN_SYS_PROMPT_w_boxed, VTS_examples

    if isinstance(entry, dict):
        sys_p = entry.get("sys_prompt", ALIGN_SYS_PROMPT_w_boxed)
        ex = entry.get("examples", "")
        if isinstance(ex, (list, tuple)):
            ex = "".join(ex)
        if not isinstance(ex, str):
            ex = str(ex)
        return sys_p, ex

    if isinstance(entry, (tuple, list)):
        return ALIGN_SYS_PROMPT_w_boxed, "".join(map(str, entry))
    if isinstance(entry, str):
        return ALIGN_SYS_PROMPT_w_boxed, entry

    return ALIGN_SYS_PROMPT_w_boxed, VTS_examples


# ---------------- 主流程 ----------------
def main():
    pa = argparse.ArgumentParser("Annotate helpers from Stage-2 using API")
    pa.add_argument("--stage2", required=True, help="path to stage2_strong_out.jsonl")
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

    # 1) 读取 Stage-2
    print("[api_anno_s2] loading Stage-2 ...", args.stage2)
    s2_recs: List[Dict[str, Any]] = load_jsonl(args.stage2)
    if args.max_records:
        s2_recs = s2_recs[: args.max_records]
    print(f"[api_anno_s2] {len(s2_recs)} samples to annotate")

    # 2) 构造带标记的输入文本
    full_texts: List[str] = []
    for s2 in s2_recs:
        ft, _ = build_alignment_text(s2)
        full_texts.append(ft)

    # 3) 按数据集分组构建 prompts 并调用 API
    grouped: Dict[str, List[Tuple[int, str]]] = {}
    for idx, s2 in enumerate(s2_recs):
        ds = s2.get("dataset_name", "") or "__UNKNOWN__"
        grouped.setdefault(ds, []).append((idx, full_texts[idx]))

    aligned_pairs: List[Tuple[int, str]] = []  # (original_index, aligned_text)
    for ds_name, items in grouped.items():
        sys_p, examples = _get_prompts_for_dataset(ds_name)
        sys_prompt = f"{sys_p}\n{examples}" if examples else sys_p

        usr_all = [_format_user_prompt(t) for _, t in items]
        for i in range(0, len(usr_all), args.batch_size):
            chunk = usr_all[i : i + args.batch_size]
            idx_chunk = [items[j][0] for j in range(i, min(i + args.batch_size, len(items)))]
            print(
                f"[api_anno_s2] calling API {args.api_model_name} | dataset={ds_name} | batch {i}-{i+len(chunk)-1}"
            )
            resps = get_api_response(
                args.api_model_name,
                sys_prompt=sys_prompt,
                user_prompts=chunk,
                temperature=0.3,
            )
            for k, r in zip(idx_chunk, resps):
                aligned_pairs.append((k, r.strip() if isinstance(r, str) else ""))

    aligned_pairs.sort(key=lambda x: x[0])
    aligned_texts = [t for _, t in aligned_pairs]

    assert len(aligned_texts) == len(s2_recs), (
        f"response len mismatch: {len(aligned_texts)} vs {len(s2_recs)}"
    )

    # 4) 解析回 steps，生成输出
    out_records: List[Dict[str, Any]] = []
    for rec_idx, (s2, al_txt) in enumerate(zip(s2_recs, aligned_texts)):
        helpers = s2.get("helpers", []) or []
        steps_aligned = parse_aligned(al_txt)
        fallback = False
        if len(steps_aligned) != len(helpers):
            fallback = True
            steps_aligned = [h.get("text", "") for h in helpers]

        # sample_id 补齐逻辑
        sid = s2.get("orig_idx")
        if sid is None:
            sid = s2.get("stage1_id")
        if sid is None:
            sid = f"fallback-{rec_idx}"

        helpers_annotated = []
        for i, (txt, h) in enumerate(zip(steps_aligned, helpers)):
            helpers_annotated.append(
                {
                    "step_idx": h.get("step_idx", i),
                    "text": txt,
                    "image_path": h.get("image_path"),
                    "type": h.get("type"),
                }
            )

        out_records.append(
            {
                "dataset_name": s2.get("dataset_name", ""),
                "sample_id": sid,
                "stage1_id": s2.get("stage1_id"),
                "orig_idx": s2.get("orig_idx"),
                "image_main": s2.get("image_main"),
                "aligned_raw": al_txt,
                "aligned_steps": steps_aligned,
                "fallback": fallback,
                "helpers_annotated": helpers_annotated,
            }
        )

    # 5) 写文件
    save_json(out_records, args.out_json)
    print(f"[api_anno_s2] wrote {len(out_records)} records -> {args.out_json}")


if __name__ == "__main__":
    main()
