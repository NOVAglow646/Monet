#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage-3  |  Alignment of helper CoT

读取:
  • --stage2 : stage2_strong_out.jsonl   (仅含 dataset_name / orig_idx / predictions)
  • --stage1 : stage1_policy_out.jsonl   (含完整 helper / question / images)

流程:
  1. 依据 (dataset_name, orig_idx) 从 Stage-1 查到原样本。
  2. 把所有 helper.text 包进 <STEP_i>..<END_STEP_i>，送入对齐 LLM / API。
  3. 解析对齐结果；若标记缺失则回退到原 helper.text。
  4. 生成最终训练样本 JSON list（make_final_cot）。
"""

from __future__ import annotations
import argparse, json, os, re, gc
from typing import Any, Dict, List, Tuple

from transformers import AutoTokenizer
from AAA_vllm_toolkit.load_and_gen_vllm import (
    vllm_llm_init,
    vllm_llm_process_batch_data,
    vllm_generate,
)
from AAA_vllm_toolkit.api import get_api_response          # type: ignore
from dataset_utils.prompts import examples_pool_exact            # type: ignore

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

def batch_align(
    dataset_name: str,
    full_texts: List[str],
    llm,
    sampling_params,
    tokenizer,
    batch_size: int,
    api_model: str | None,
):
    outs: List[str] = []
    sys_prompt, _ = _prompts(["dummy"], dataset_name)  # sys_part 共用
    for i in range(0, len(full_texts), batch_size):
        chunk = full_texts[i : i + batch_size]
        _, usr_prompts = _prompts(chunk, dataset_name)
        if llm:
            inputs = vllm_llm_process_batch_data(
                sys_prompt=sys_prompt, usr_prompts=usr_prompts, tokenizer=tokenizer
            )
            gen = vllm_generate(inputs, sampling_params, llm)
            outs.extend(g.outputs[0].text.strip() if g.outputs else "" for g in gen)
        else:
            resps = get_api_response(api_model, sys_prompt, usr_prompts, temperature=0.3)
            outs.extend(r.strip() for r in resps)
    return outs

# ---------- make_final_cot ----------
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
    if s1_rec.get("image_main"):
        usr_content.append({"type": "image", "image_file_name": s1_rec["image_main"]})
    usr_content.append({"type": "text", "text": s1_rec["question"]})
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

# ---------- main ----------
def main():
    pa = argparse.ArgumentParser("Stage-3 Alignment (new)")
    pa.add_argument("--stage2", required=True)
    pa.add_argument("--stage1", required=True)               # 新增
    pa.add_argument("--llm-path", default="")
    pa.add_argument("--judge_llm_tensor_parallel_size", type=int, default=4)
    pa.add_argument("--devices", default="0,1,2,3")
    pa.add_argument("--out-json", required=True)
    pa.add_argument("--max-records", type=int)
    pa.add_argument("--align-batch", type=int, default=4096)
    pa.add_argument("--api_model_name", choices=["gemini-2.5-pro", "deepseek-chat"])
    args = pa.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    # 1. 读取 Stage-1, 建立索引
    print("[Stage-3] loading Stage-1 ...")
    s1_recs = load_jsonl(args.stage1)
    s1_map: Dict[Tuple[str, int], Dict[str, Any]] = {
        (r["dataset_name"], r["orig_idx"]): r for r in s1_recs
    }

    # 2. 读取 Stage-2
    s2_recs = load_jsonl(args.stage2)
    if args.max_records:
        s2_recs = s2_recs[: args.max_records]
    print(f"[Stage-3] {len(s2_recs)} samples to align")

    # 3. 回查 Stage-1 样本并准备 alignment 文本
    full_texts, s1_samples = [], []
    for r in s2_recs:
        key = (r["dataset_name"], r["orig_idx"])
        if key not in s1_map:
            continue
        s1 = s1_map[key]
        ft, _ = build_alignment_text(s1)
        full_texts.append(ft)
        s1_samples.append(s1)

    # 4. 准备 LLM / API
    if args.llm_path:
        print(f"[Stage-3] load alignment LLM: {args.llm_path}")
        llm, sampling = vllm_llm_init(
            args.llm_path, tp=args.judge_llm_tensor_parallel_size
        )
        tok = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)
        api_model = None
    else:
        llm = tok = sampling = None
        api_model = args.api_model_name
        if api_model is None:
            raise ValueError("Either --llm-path or --api_model_name must be provided")

    # 5. 批量对齐
    ds_name = s1_samples[0]["dataset_name"] if s1_samples else "default"
    aligned = batch_align(
        ds_name,
        full_texts,
        llm,
        sampling,
        tok,
        args.align_batch,
        api_model,
    )
    assert len(aligned) == len(s1_samples)

    # 6. 解析 & 生成最终 COT
    finals = []
    for s1_rec, txt_al in zip(s1_samples, aligned):
        steps_al = parse_aligned(txt_al)
        # 若解析失败，回退到原 helper.text
        if len(steps_al) != len(s1_rec["helpers"]):
            steps_al = [h.get("text", "") for h in s1_rec["helpers"]]
        finals.append(make_final_cot(s1_rec, steps_al))

    save_json(finals, args.out_json)
    print(f"[Stage-3] wrote {len(finals)} records -> {args.out_json}")
    gc.collect()

if __name__ == "__main__":
    main()
