#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 3: Alignment (全步拼接) + 分步还原 + 打包训练集
=====================================================

与上一版 Stage3 差异（按用户最新要求）：
1. **Alignment 输入** = 所有 helper step 文本 (0..K) + strong_mllm 首次答对步原始输出，按顺序拼接，并加 *显式标记*；整体送入 alignment LLM。
2. Alignment LLM 在上下文整体基础上插入 `<observation>...</observation>`；**要求原样保留每段标记**，我们据此解析回每一步文本。
3. **最终保存的 assistant content：**
   - step0..K：使用 alignment 后的分步文本（分别替换原 helper 文本）。
   - 每步后跟其图像（如有）。
   - 最后一项只放 alignment 后的 *strong_mllm 正确回答段*（不再重复之前所有文本）。

若 alignment 输出缺失或标记不完整，将回退到原始未对齐文本（保持稳健）。
"""

from __future__ import annotations
import os
import json
import re
import argparse
from typing import List, Dict, Any, Optional, Tuple

from transformers import AutoTokenizer

# --- 用户工具包 -------------------------------------------------------------
from AAA_vllm_toolkit.load_and_gen_vllm import (
    vllm_llm_init,
    vllm_llm_process_batch_data,
    vllm_generate,
)
from AAA_vllm_toolkit.deepseek_api import build_ds_client, get_ds_response
from data_utils.prompts import *

# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def save_json(data: List[Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# Alignment 输入构建 & 解析
# ---------------------------------------------------------------------------

def build_alignment_text_for_sample_last_strong(rec: Dict[str,Any]) -> Tuple[str,int]:
    """Return (full_text, num_steps=K+1). K = first_correct_helper_idx."""
    k = rec["first_correct_helper_idx"]
    helpers = rec.get("helpers", [])
    segments = []
    for i in range(k+1):
        txt = helpers[i].get("text", "")
        segments.append(STEP_START.format(i=i) + txt + STEP_END.format(i=i))
    # strong output at step k
    strong_txt = rec["strong_steps"][k].get("pred_raw", "")
    segments.append(FINAL_START + strong_txt + FINAL_END)
    full = "\n".join(segments)
    return full, k+1  # num segments incl FINAL

def build_alignment_text_for_sample(rec: Dict[str,Any]) -> Tuple[str,int]:
    """Return (full_text, num_steps=K+1). K = first_correct_helper_idx."""
    k = rec["first_correct_helper_idx"]
    helpers = rec.get("helpers", [])
    segments = []
    for i in range(len(helpers)):
        txt = helpers[i].get("text", "")
        segments.append(STEP_START.format(i=i) + txt + STEP_END.format(i=i))

    full = "\n".join(segments)
    return full, k+1  # num segments incl FINAL

# regex precompile (DOTALL)
_STEP_RE_TPL = r"<STEP_{i}>\s*(.*?)\s*<END_STEP_{i}>"
_FINAL_RE = re.compile(r"<FINAL_STEP>\s*(.*?)\s*<END_FINAL_STEP>", re.S)


def parse_aligned_text(text: str, k: int) -> Tuple[List[str], str]:
    """Parse alignment output. Return (aligned_steps[0..k], aligned_final)."""
    aligned_steps = []
    for i in range(k+1):  # steps 0..k
        pat = re.compile(_STEP_RE_TPL.format(i=i), re.S)
        m = pat.search(text)
        if m:
            aligned_steps.append(m.group(1).strip())
        else:
            aligned_steps.append("")  # fallback later
    m = _FINAL_RE.search(text)
    aligned_final = m.group(1).strip() if m else ""
    return aligned_steps, aligned_final

# ---------------------------------------------------------------------------
# Alignment 批生成
# ---------------------------------------------------------------------------

def build_align_prompts(inputs: List[str]) -> List[str]:
    return ["Now it's your turn. ## Input: " + t + "\n\n## Your output:" for t in inputs]


def batch_align_concat(dataset_name: str, full_texts: List[str], llm, sampling_params, tokenizer, batch_size: int=512, ds_client=None) -> List[str]:
    out_texts: List[str] = []
    n = len(full_texts)
    for i in range(0, n, batch_size):
        chunk = full_texts[i:i+batch_size]
        prompts = build_align_prompts(chunk)
        if llm is not None:
            inputs = vllm_llm_process_batch_data(sys_prompt=ALIGN_SYS_PROMPT_LAST_STRONG+examples_pool[dataset_name], usr_prompts=prompts, tokenizer=tokenizer)
            outs = vllm_generate(inputs, sampling_params, llm)
            for o in outs:
                out_texts.append(o.outputs[0].text.strip() if o.outputs else "")
        elif ds_client is not None:
            responses = get_ds_response(ds_client, ALIGN_SYS_PROMPT_LAST_STRONG+examples_pool[dataset_name], prompts, temperature=0.3)
            for response in responses:
                out_texts.append(response.strip())
        else:
            raise ValueError("Either llm or ds_client must be provided for alignment.")
    return out_texts

# ---------------------------------------------------------------------------
# cot_to_save 构造（使用解析后的每步文本）
# ---------------------------------------------------------------------------

def make_final_cot(sample: Dict[str,Any], aligned_steps: List[str], aligned_final: str) -> List[Dict[str,Any]]:
    k = sample["first_correct_helper_idx"]
    helpers = sample.get("helpers", [])
    # system
    cot = [{"role":"system","content":[{"type":"text","text":"You are a helpful assistant. You can generate abstract visual tokens that represent a cropped image region or images with auxiliary information like lines, bounding boxes, etc. When you decide to generate abstract visual tokens, put them in <abs_vis_token>...</abs_vis_token>."}]}]
    # user
    uc = []
    if sample.get("image_main"):
        uc.append({"type":"image","image_file_name":sample["image_main"]})
    uc.append({"type":"text","text":sample["question"]})
    cot.append({"role":"user","content":uc})
    # assistant (aligned per-step)
    ac = []
    for i in range(k+1):
        if i < len(helpers):
            txt_al = aligned_steps[i].strip() if i < len(aligned_steps) else helpers[i].get("text","")
            if not txt_al:
                txt_al = helpers[i].get("text","")  # fallback
            ac.append({"type":"text","text":txt_al})
            ip = helpers[i].get("image_path")
            if ip:
                ac.append({"type":"image","image_file_name":ip})
    # final strong segment
    final_txt = aligned_final.strip()
    ac.append({"type":"text","text":final_txt})
    cot.append({"role":"assistant","content":ac})
    return cot

# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Stage3 align all steps -> per-step output")
    ap.add_argument("--stage2", required=True, help="Stage2 JSONL")
    ap.add_argument("--llm-path", help="alignment text-only 模型路径", default="")
    ap.add_argument("--devices", default="0,1,2,3", help="GPU IDs")
    ap.add_argument("--out-json", required=True, help="输出训练 JSON list")
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--align-batch", type=int, default=4096)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    recs = load_jsonl(args.stage2)
    if args.max_records is not None:
        recs = recs[:args.max_records]
    # 保留 strong 成功纠错样本
    good = []
    for r in recs:
        k = r.get("first_correct_helper_idx")
        if k is None: continue
        steps = r.get("strong_steps", [])
        if k >= len(steps): continue
        if not steps[k].get("correct", False): continue
        good.append(r)
    print(f"[Stage3] retained {len(good)} / {len(recs)} records")

    # 构造 full_texts + seg counts
    full_texts: List[str] = []
    seg_counts: List[int] = []
    for r in good:
        ft, nseg = build_alignment_text_for_sample_last_strong(r)
        full_texts.append(ft)
        seg_counts.append(nseg)  # k+1 (helpers) + final

    # 模型
    llm = None
    tokenizer = None
    sampling_params = None
    ds_client = None
    
    if args.llm_path == "":
        ds_client = build_ds_client()
    else:
        print(f"[Stage3] loading alignment LLM: {args.llm_path}")
        llm, sampling_params = vllm_llm_init(args.llm_path)
        tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)

    # 批 alignment
    dataset_name = args.stage2.split("/")[-2]
    aligned_texts = batch_align_concat(dataset_name, full_texts, llm, sampling_params, tokenizer, batch_size=args.align_batch, ds_client=ds_client)
    assert len(aligned_texts) == len(good)

    # 解析 & 打包
    finals = []
    for rec, txt_al, nseg in zip(good, aligned_texts, seg_counts):
        k = rec["first_correct_helper_idx"]
        step_texts, final_txt = parse_aligned_text(txt_al, k)
        cot = make_final_cot(rec, step_texts, final_txt)
        finals.append(cot)

    save_json(finals, args.out_json)
    print(f"[Stage3] wrote {len(finals)} records -> {args.out_json}")

if __name__ == "__main__":
    main()
