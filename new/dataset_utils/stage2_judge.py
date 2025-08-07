#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage-2 (Part 2) — judge predictions with leak filtering and statistics.
"""

from __future__ import annotations
import argparse, json
from typing import Any, Dict, List, Tuple
import os
from AAA_vllm_toolkit.load_and_gen_vllm import vllm_llm_init, vllm_kill_model
from AAA_vllm_toolkit.extract_and_check import quick_batch_judge, llm_batch_judge, llm_batch_extract, data_spec_batch_judge


# --------------------------------------------------------------------------- #
def load_stage1_map(path: str):
    m: Dict[Tuple[str, int], Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        iterator = json.load(f) if head == "[" else (json.loads(l) for l in f if l.strip())
        for rec in iterator:
            m[(rec["dataset_name"], rec["orig_idx"])] = rec
    return m


def process(
    batch_pred: List[Dict[str, Any]],
    s1_map: Dict[Tuple[str, int], Dict[str, Any]],
    judge_mode: str,
    judge_llm,
    use_llm_to_extract_answer: bool,
    writer,
):
    """Return (total, leak_drop, no_correct_drop, kept)."""
    flat_preds, flat_gt, flat_choices, flat_q, sid_map = [], [], [], [], []
    for sid, item in enumerate(batch_pred):
        key = (item["dataset_name"], item["orig_idx"])
        rec = s1_map[key]
        gt_text = rec.get("gt_answer_text")
        choices = rec.get("gt_choices")
        q = rec["question"]
        for p in item["predictions"]:
            flat_preds.append(p)
            flat_gt.append(gt_text)
            flat_choices.append(choices)
            flat_q.append(q)
            sid_map.append(sid)

    # correctness flags
    if use_llm_to_extract_answer:
        flat_gt = llm_batch_extract(flat_gt, judge_llm, flat_q, item["dataset_name"])

    llm_judged = [0]*len(flat_preds)  
    quick_judged = [0]*len(flat_preds)  
    data_spec_judged = [0]*len(flat_preds)      

    if "quick" in judge_mode:
        quick_judged = quick_batch_judge(flat_preds, flat_gt, flat_choices)
    if "llm" in judge_mode:    
        llm_judged = llm_batch_judge(flat_preds, flat_gt, judge_llm, flat_q)
    if "data_spec" in judge_mode:
        data_spec_judged = data_spec_batch_judge(flat_preds, flat_gt, item["dataset_name"])
    
    flags = [llm or quick or data_spec for llm, quick, data_spec in zip(llm_judged, quick_judged, data_spec_judged)]
    
    per_sample_ok: List[List[bool]] = [[] for _ in batch_pred]
    for flg, sid in zip(flags, sid_map):
        per_sample_ok[sid].append(bool(flg))

    total, leak_drop, no_correct_drop, kept = 0, 0, 0, 0

    for item, ok_list in zip(batch_pred, per_sample_ok):
        total += 1
        key = (item["dataset_name"], item["orig_idx"])
        rec = s1_map[key]
        helpers = rec["helpers"]

        # first correct step
        first_correct = next((i for i, ok in enumerate(ok_list) if ok), None)
        if first_correct is None:
            no_correct_drop += 1
            continue

        # leak detection
        leak_idx = None
        for i, h in enumerate(helpers):
            txt = (h.get("text") or "").replace("\\\\", "\\").lower()
            if "\\boxed" in txt or "answer" in txt:
                leak_idx = i
                break
        if leak_idx is not None and leak_idx < first_correct + 1:
            leak_drop += 1
            continue

        # keep
        kept += 1
        json.dump(
            {
                "dataset_name": item["dataset_name"],
                "orig_idx": item["orig_idx"],
                "predictions": item["predictions"],
            },
            writer,
            ensure_ascii=False,
        )
        writer.write("\n")
    writer.flush()
    return total, leak_drop, no_correct_drop, kept


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser("stage2_judge with stats")
    ap.add_argument("--infer-file", required=True)
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge_llm_dir", type=str, default="")
    ap.add_argument("--judge_llm_tensor_parallel_size", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8192)
    #ap.add_argument("--use_llm_to_judge", action="store_true", default=False)
    ap.add_argument("--devices", default="0,1,2,3")
    ap.add_argument("--max_samples", type=int, default=1000000)
    ap.add_argument("--use_llm_to_extract_answer", action="store_true", default=False)
    ap.add_argument("--judge_mode", choices=["quick", "llm", "data_spec"], nargs='+', )
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    s1_map = load_stage1_map(args.stage1)

    judge_llm = None
    if "llm" in args.judge_mode:
        judge_llm, _ = vllm_llm_init(
            args.judge_llm_dir, tp=args.judge_llm_tensor_parallel_size, gpu_memory_utilization=0.95,max_model_len=7400
        )

    writer = open(args.out, "w", encoding="utf-8")
    cnt = 0
    total_in = leak_drop = no_correct_drop = kept = 0
    buf: List[Dict[str, Any]] = []
    with open(args.infer_file, "r", encoding="utf-8") as f:
        for ln in f:
            buf.append(json.loads(ln))
            if len(buf) >= args.batch or len(buf) >= args.max_samples:
                t, l, n, k = process(buf, s1_map, args.judge_mode, judge_llm, args.use_llm_to_extract_answer, writer)
                total_in += t
                leak_drop += l
                no_correct_drop += n
                kept += k
                if len(buf) >= args.max_samples:
                    break
                buf.clear()
        if buf:
            t, l, n, k = process(buf, s1_map, args.judge_mode, judge_llm, args.use_llm_to_extract_answer, writer)
            total_in += t
            leak_drop += l
            no_correct_drop += n
            kept += k

    writer.close()
    if judge_llm:
        vllm_kill_model(judge_llm)

    print(
        f"[Judge Stats] total_in: {total_in}  |  "
        f"leak_dropped: {leak_drop}  |  "
        f"no_correct: {no_correct_drop}  |  kept: {kept}"
    )


if __name__ == "__main__":
    main()
