#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 2: Strong 模型 Helper 顺序增量评测（批推理版 + 提前答案剔除 / 最小改动）
==========================================================================

这是在你提供的 *batch* 版 Stage2 脚本基础上做的**最小必要修改**，只新增一条规则：

**若任意 helper step 文本中出现 `\boxed{...}`（不管内容是什么，也不与 GT 对比），且该 step 的索引 < strong_mllm 首次答对步 (`first_correct_helper_idx`)，则剔除该样本，不写入主输出。**

其它逻辑（批推理、token 限长、断点续跑等）保持原样；仅在写出阶段执行过滤。

可选：用 `--leak-log <path>` 将被剔除样本单独保存，便于检查；若不提供，只在 stdout 打印计数。

---
运行示例：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python stage2_strong_helper_eval_leak.py \
  --stage1 ./created_dataset/filtered_data/CoF/stage1_policy_out.jsonl \
  --out    ./created_dataset/filtered_data/CoF/stage2_strong_out.jsonl \
  --model-path /data1/qxwang/checkpoints/Qwen2.5-VL-7B-Instruct \
  --devices 0,1,2,3 \
  --token-limit 8192 \
  --resume \
  --leak-log ./created_dataset/filtered_data/CoF/stage2_leak_dropped.jsonl
```
"""

from __future__ import annotations
import os
import io
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple

from tqdm import tqdm
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor

# --- 用户工程工具 ---
from AAA_vllm_toolkit.load_and_gen_vllm import (
    vllm_mllm_init,
    vllm_mllm_process_batch_from_messages,
    count_qwen_vl_tokens,
    vllm_kill_model,
    vllm_llm_init
)
from AAA_vllm_toolkit.extract_and_check import (
    extract_boxed_answer,
    batch_judge,
    quick_batch_judge
)


# -----------------------------------------------------------------------------
# 常量
# -----------------------------------------------------------------------------
ALIGN_BOXED_INSTRUCTION = (
    " Put your final answer within \\boxed{}. If you cannot see relevant visual information to infer the answer from the image, just output \\boxed{None} and don't guess the answer based on your knowledge."
)

# -----------------------------------------------------------------------------
# 实用函数
# -----------------------------------------------------------------------------

def parse_devices(dev_str: str) -> List[str]:
    return [d for d in dev_str.split(',') if d.strip()]


def load_stage1(path: str) -> List[Dict[str, Any]]:
    """Load Stage1 records.
    支持 JSONL 或 JSON array 文件；自动过滤 policy_correct==True 样本。"""
    recs = []
    with open(path, 'r', encoding='utf-8') as f:
        head = f.read(1)
        f.seek(0)
        if head == '[':
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Stage1 file JSON array malformed")
            iter_data = data
        else:
            iter_data = [json.loads(ln) for ln in f if ln.strip()]
    for rec in iter_data:
        if rec.get('policy_correct'):
            continue
        recs.append(rec)
    return recs


def load_done_map(path: str) -> Dict[Tuple[str,int], Dict[str,Any]]:
    """Load previously completed Stage2 output (resume)."""
    m: Dict[Tuple[str,int], Dict[str,Any]] = {}
    if not os.path.isfile(path):
        return m
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            key = (rec.get('dataset_name'), rec.get('orig_idx'))
            m[key] = rec
    return m


def materialize_image(path: str) -> Image.Image:
    """完全加载图像，解除文件句柄依赖。"""
    with open(path, 'rb') as f:
        b = f.read()
    img = Image.open(io.BytesIO(b))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.load()  # force decode
    return img


# -----------------------------------------------------------------------------
# 会话构建：累积 helper (step0..step_idx)，不送主图
# -----------------------------------------------------------------------------

def build_conv_for_step(rec: Dict[str,Any], step_idx: int, load_images: bool=True) -> Tuple[List[Dict[str,Any]], List[Image.Image]]:
    """构造 strong_mllm 输入对话（仅 user）。返回 (conv, opened_images)。"""
    qtxt = rec["question"]
    if "\\boxed{}" not in qtxt:  # 再保险
        qtxt = qtxt + ALIGN_BOXED_INSTRUCTION

    content = [{"type": "text", "text": qtxt}]

    helpers = rec.get("helpers", [])
    opened: List[Image.Image] = []
    upto = min(step_idx, len(helpers) - 1)
    if upto >= 0:
        for i in range(upto + 1):
            h = helpers[i]
            htxt = h.get("text", "")
            if htxt:
                content.append({"type": "text", "text": htxt})
            ipath = h.get("image_path")
            if load_images and ipath and os.path.isfile(ipath):
                try:
                    img = materialize_image(ipath)
                    opened.append(img)
                    content.append({"type": "image", "image": img})
                except Exception as e:
                    print(f"[WARN] failed load image {ipath}: {e}")
    conv = [{"role": "user", "content": content}]
    return conv, opened


# -----------------------------------------------------------------------------
# 批推理：同一 step_idx 的多样本
# -----------------------------------------------------------------------------

def run_batch_step(
    recs: List[Dict[str,Any]],
    step_idx: int,
    model,
    processor,
    tokenizer,
    token_limit: int,
    sampling_params,
    judge_llm_dir=None
) -> List[Dict[str,Any]]:
    """批量处理 recs 在 step_idx 下的 strong 推理。"""
    def _has_image(conv):
        return any(it.get("type") == "image" for msg in conv for it in msg["content"])

    questions = []
    convs: List[Optional[List[Dict[str,Any]]]] = []
    opened_pool: List[Image.Image] = []
    for rec in recs:
        helpers = rec.get("helpers", [])
        if step_idx >= len(helpers):
            convs.append(None)
            continue
        conv, opened = build_conv_for_step(rec, step_idx, load_images=True)
        if not _has_image(conv):
            convs.append(None)
            continue
        convs.append(conv)
        opened_pool.extend(opened)
        questions.append(rec["question"])
    eff_idx = [i for i,c in enumerate(convs) if c is not None]
    if not eff_idx:
        return [
            {"pred_raw": None, "pred_extracted": None, "correct": None, "token_count": None, "truncated": False}
            for _ in recs
        ]

    eff_convs = [convs[i] for i in eff_idx]
    eff_inputs = vllm_mllm_process_batch_from_messages(eff_convs, processor)
    token_counts = count_qwen_vl_tokens(eff_inputs, tokenizer, processor)

    gen_idx = []
    gen_convs = []
    for sub_i, tok in enumerate(token_counts):
        if (token_limit is not None) and (tok is not None) and (tok > token_limit):
            pass
        else:
            gen_idx.append(sub_i)
            gen_convs.append(eff_convs[sub_i])

    sub_results = {}
    if gen_convs:
        gen_inputs = vllm_mllm_process_batch_from_messages(gen_convs, processor)
        outs = model.generate(gen_inputs, sampling_params=sampling_params, use_tqdm=True)
        for sub_i, out in zip(gen_idx, outs):
            txt = out.outputs[0].text.strip() if out.outputs else ""
            ext = extract_boxed_answer(txt)
            sub_results[sub_i] = (txt, ext)
    vllm_kill_model(model)
    
    for img in opened_pool:
        try:
            img.close()
        except Exception:
            pass

    ret: List[Dict[str,Any]] = []
    eff_tok = token_counts
    judge_preds = []
    judge_gt = []
    judge_choices = []
    judge_map = []
    eff_ret = [None] * len(eff_idx)
    for k, rec_i in enumerate(eff_idx):
        tok = eff_tok[k]
        if k not in sub_results:
            eff_ret[k] = {
                "pred_raw": None,
                "pred_extracted": None,
                "correct": None,
                "token_count": tok,
                "truncated": (tok is not None and token_limit is not None and tok > token_limit),
            }
            continue
        txt, ext = sub_results[k]
        eff_ret[k] = {
            "pred_raw": txt,
            "pred_extracted": ext,
            "correct": None,
            "token_count": tok,
            "truncated": False,
        }
        judge_preds.append(ext)
        judge_gt.append(recs[rec_i].get("gt_answer_text"))
        judge_choices.append(recs[rec_i].get("gt_choices"))
        judge_map.append(k)

    if judge_preds:
        try:
            if judge_llm_dir is not None:
                judge_llm, _ = vllm_llm_init(judge_llm_dir, tp=4)
            flags = batch_judge(judge_preds, judge_gt, judge_choices, questions=questions if len(questions)>0 else None, llm=judge_llm)
            #flags = quick_batch_judge(judge_preds, judge_gt)
        except Exception:
            # fallback：简单大小写比较
            flags = []
            for p,g,c in zip(judge_preds, judge_gt, judge_choices):
                if p is None or g is None:
                    flags.append(False)
                else:
                    flags.append(str(p).strip().lower() == str(g).strip().lower())
        for flg, sub_k in zip(flags, judge_map):
            eff_ret[sub_k]["correct"] = bool(flg)

    ret_full = []
    k = 0
    for i in range(len(recs)):
        if i in eff_idx:
            ret_full.append(eff_ret[k])
            k += 1
        else:
            ret_full.append({
                "pred_raw": None,
                "pred_extracted": None,
                "correct": None,
                "token_count": None,
                "truncated": False,
            })
    return ret_full


# -----------------------------------------------------------------------------
# 主过程
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Stage2 strong helper eval (batch + early-answer filter)")
    ap.add_argument("--stage1", required=True, help="Stage1 JSONL 输入")
    ap.add_argument("--out", required=True, help="Stage2 输出 JSONL")
    ap.add_argument("--model-path", required=True, help="strong_mllm 模型路径")
    ap.add_argument("--devices", default="0,1,2,3", help="GPU IDs, 逗号分隔")
    ap.add_argument("--token-limit", type=int, default=8192, help="Token 上限；>即 trunc")
    ap.add_argument("--max-batch", type=int, default=0, help=">0 则每轮切块；0=不切")
    ap.add_argument("--resume", action="store_true", help="断点续跑")
    ap.add_argument("--max-samples", type=int, default=None, help="调试：限制样本数")
    ap.add_argument("--leak-log", type=str, default=None, help="保存被剔除样本 JSONL (可选)")
    ap.add_argument("--judge_llm_dir", type=str, default=None)
    args = ap.parse_args()

    devs = parse_devices(args.devices)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    # 模型（tp=GPU数）
    print(f"[Stage2] loading strong_mllm: {args.model_path}")
    model, sampling_params = vllm_mllm_init(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path)

    # 读 Stage1
    recs = load_stage1(args.stage1)
    if args.max_samples is not None:
        recs = recs[: args.max_samples]
    print(f"[Stage2] loaded {len(recs)} policy-wrong samples from Stage1.")

    # resume map
    done_map = load_done_map(args.out) if args.resume else {}
    if done_map:
        print(f"[Stage2] resume: found {len(done_map)} done records in {args.out}.")

    # state 初始化（剔除已完成）
    state = []  # list[dict]
    for rec in recs:
        key = (rec.get('dataset_name'), rec.get('orig_idx'))
        if key in done_map:
            continue
        state.append({
            'rec': rec,
            'done': False,
            'strong_steps': [],
            'first_correct_helper_idx': None,
            'truncated': False,
        })
    print(f"[Stage2] {len(state)} samples to process.")

    # 计算最大 helper 步数
    max_steps = 0
    for st in state:
        n = len(st['rec'].get('helpers', []))
        if n > max_steps:
            max_steps = n

    # 主循环：逐 step 批推理
    for step_idx in range(max_steps):
        active_idx = [i for i,st in enumerate(state) if (not st['done']) and (step_idx < len(st['rec'].get('helpers', [])))]
        if not active_idx:
            continue
        if args.max_batch and args.max_batch > 0:
            chunks = [active_idx[i:i+args.max_batch] for i in range(0,len(active_idx),args.max_batch)]
        else:
            chunks = [active_idx]

        print(f"[Stage2] step{step_idx}: {len(active_idx)} active")
        for ch in chunks:
            batch_recs = [state[i]['rec'] for i in ch]
            res_list = run_batch_step(batch_recs, step_idx, model, processor, tokenizer, args.token_limit, sampling_params, judge_llm_dir=args.judge_llm_dir)
            # 写回
            for loc_i, st_idx in enumerate(ch):
                st = state[st_idx]
                ret = res_list[loc_i]
                st['strong_steps'].append({
                    'ctx_upto_step': step_idx,
                    'pred_raw': ret['pred_raw'],
                    'pred_extracted': ret['pred_extracted'],
                    'correct': ret['correct'],
                    'token_count': ret['token_count'],
                    'truncated': ret['truncated'],
                })
                if ret['truncated']:
                    st['truncated'] = True
                    st['done'] = True
                elif ret['correct'] and st['first_correct_helper_idx'] is None:
                    st['first_correct_helper_idx'] = step_idx
                    st['done'] = True

    # 合并 + 写出（含已完成）
    out_records = list(done_map.values())
    for st in state:
        rec = dict(st['rec'])
        rec['strong_steps'] = st['strong_steps']
        rec['first_correct_helper_idx'] = st['first_correct_helper_idx']
        rec['truncated'] = st['truncated']
        out_records.append(rec)

    # 稳定排序（按 dataset_name, orig_idx）
    out_records.sort(key=lambda r: (r.get('dataset_name'), r.get('orig_idx')))

    # === 新增：提前答案过滤 ===
    dropped = []
    kept = []
    for rec in out_records:
        fc = rec.get('first_correct_helper_idx')
        if fc is None:
            kept.append(rec)  # strong 从未答对 -> 保留（如需剔除可改这里）
            continue
        # 检测 helper 中最早出现 \boxed 的步（不关心内容，只看出现）
        leak_idx = None
        for i,h in enumerate(rec.get('helpers', [])):
            txt = (h.get('text') or "")
            tnorm = txt.replace('\\\\','\\').lower()
            if '\\boxed' in tnorm or "answer" in tnorm:  # 兼容 "\\boxed" / "\boxed"
                leak_idx = i
                break
        if leak_idx is not None and leak_idx < fc + 1: # +1 因为 strong_mllm 只看 ctx_upto_step
            dropped.append(rec)
        else:
            kept.append(rec)

    # 写主输出
    with open(args.out, 'w', encoding='utf-8') as f:
        for i,rec in enumerate(kept):
            if i: f.write('\n')
            f.write(json.dumps(rec, ensure_ascii=False))
    print(f"[Stage2] wrote {len(kept)} records -> {args.out}")

    # 可选泄露日志
    if args.leak_log:
        with open(args.leak_log, 'w', encoding='utf-8') as f:
            for i,rec in enumerate(dropped):
                if i: f.write('\n')
                f.write(json.dumps(rec, ensure_ascii=False))
        print(f"[Stage2] dropped {len(dropped)} leak records -> {args.leak_log}")
    else:
        print(f"[Stage2] dropped {len(dropped)} leak records (no log saved)")

if __name__ == "__main__":
    main()
