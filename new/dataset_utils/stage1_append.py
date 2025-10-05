#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 1 (append mode): 解析 -> 按批次推理 -> 判分 -> 逐批次 append 写出

区别于原版：不再一次性收集所有样本后统一落盘，而是每处理完一个 batch 就将结果追加到目标 jsonl 文件，便于长任务断点续跑与更快产出部分结果。
"""

from __future__ import annotations
import os
import re
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm

from transformers import AutoProcessor, AutoTokenizer  # trust_remote_code needed

from AAA_vllm_toolkit.load_and_gen_vllm import (
    vllm_mllm_init,
    vllm_kill_model,
)

try:
    from .judge_pipeline import sequential_judge_predictions
    from .stage1 import (
        DEFAULT_DATASETS,
        build_policy_conversation,
        save_images_for_sample,
        run_policy_batch,
        Stage1PolicyDPWorkerPool,
        parse_pixelreasoner,
        parse_com,
        parse_cof,
        parse_refocus,
        parse_visual_cot_all,
        parse_visual_cot_w_choices,
        parse_visual_cot_wo_choices,
        parse_zebra_cot,
        parse_vts,
    )
except Exception:
    from judge_pipeline import sequential_judge_predictions
    from stage1 import (
        DEFAULT_DATASETS,
        build_policy_conversation,
        save_images_for_sample,
        run_policy_batch,
        Stage1PolicyDPWorkerPool,
        parse_pixelreasoner,
        parse_com,
        parse_cof,
        parse_refocus,
        parse_visual_cot_all,
        parse_visual_cot_w_choices,
        parse_visual_cot_wo_choices,
        parse_zebra_cot,
        parse_vts,
    )


def iter_samples(dataset_name: str, dataset_path: Path) -> List[Dict[str, Any]]:
    # 保持与 stage1 读取逻辑一致（有些数据集一次性 load）。
    # 为简洁与一致性，这里复用与 stage1 相同的加载方式。
    ds_cfg = DEFAULT_DATASETS[dataset_name]
    if str(ds_cfg["dataset_path"]).endswith(".jsonl"):
        samples = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        if "CoM" in dataset_name:
            samples = samples[0]
        return samples
    elif str(ds_cfg["dataset_path"]).endswith(".json"):
        with open(dataset_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        return samples
    else:
        # 其他格式（如 parquet）在原 stage1 中通过 datasets.load_dataset 加载。
        # 为避免引入额外复杂度，这里不特别处理；如需支持，可仿照 stage1 实现。
        raise ValueError("Unsupported dataset format for append mode. Use json/jsonl paths.")


def main():
    ap = argparse.ArgumentParser("stage1_append")
    ap.add_argument("--dataset-name", type=str, required=True)
    ap.add_argument("--policy-model-path", type=str, required=True)
    ap.add_argument("--devices", type=str, default="0,1,2,3")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--out-root", type=str, default="./created_dataset/filtered_data/")
    ap.add_argument("--start-id", type=int, default=0)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--judge_llm_dir", type=str, default=None)
    ap.add_argument("--judge_llm_tensor_parallel_size", type=int, default=2)
    ap.add_argument("--policy_mllm_tensor_parallel_size", type=int, default=2)
    ap.add_argument("--judge_mode", choices=["llm", "quick", "data_spec", "api"], nargs="+")
    # API judging
    ap.add_argument("--api_name", type=str, default=None, choices=["gemini-2.5-pro", "deepseek-chat"])
    ap.add_argument("--api_max_workers", type=int, default=32)
    ap.add_argument("--api_temperature", type=float, default=0.0)
    # dataset paths
    ap.add_argument("--dataset_path", type=str, default=None)
    ap.add_argument("--dataset_images_root", type=str, default=None)
    # performance knobs
    ap.add_argument("--policy_batch", type=int, default=256, help="微批 generate 大小")
    ap.add_argument("--batch", type=int, default=8192, help="外层处理批大小（样本数）")
    ap.add_argument("--groups_per_gpu", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    # 数据集路径
    ds_cfg = DEFAULT_DATASETS[args.dataset_name]
    dataset_path = Path(ds_cfg["dataset_path"]) if args.dataset_path is None else Path(args.dataset_path)
    dataset_images_root = Path(ds_cfg["dataset_images_root"]) if args.dataset_images_root is None else Path(args.dataset_images_root)

    # 输出路径与文件名规则
    out_root = Path(args.out_root) / args.dataset_name
    out_root.mkdir(parents=True, exist_ok=True)
    out_img_dir = out_root / "images"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    suffix_num = None
    if args.dataset_path is not None:
        m = re.search(r"_(\d+)\.json$", args.dataset_path)
        if m:
            suffix_num = m.group(1)
    if suffix_num is not None:
        out_jsonl = out_root / f"stage1_policy_out_{suffix_num}.jsonl"
    else:
        out_jsonl = out_root / "stage1_policy_out.jsonl"

    # 先清空旧文件，再按批次 append 写入
    open(out_jsonl, "w", encoding="utf-8").close()
    writer = open(out_jsonl, "a", encoding="utf-8")

    # 设备与 DP 分组（允许同卡多引擎）
    devices_list = [int(x) for x in args.devices.split(",") if x.strip() != ""]
    tp = args.policy_mllm_tensor_parallel_size
    base_degree = len(devices_list) // tp if tp > 0 else 0
    base_groups = [devices_list[i * tp : (i + 1) * tp] for i in range(base_degree)] if base_degree > 0 else []
    if args.groups_per_gpu > 1 and base_groups:
        dp_groups: List[List[int]] = []
        for _ in range(args.groups_per_gpu):
            dp_groups.extend(base_groups)
    else:
        dp_groups = base_groups
    use_dp = len(dp_groups) > 1
    dp_pool = None
    if args.groups_per_gpu > 1:
        eff_util = args.gpu_memory_utilization * args.groups_per_gpu
        if eff_util > 0.95:
            print(f"[WARN] groups_per_gpu({args.groups_per_gpu}) * gpu_memory_utilization({args.gpu_memory_utilization}) = {eff_util:.2f} > 0.95，可能 OOM 或抖动。请下调其中之一。")

    # 单引擎路径下，初始化一次模型与处理器；DP 路径下，初始化持久化 worker 池
    policy_mllm = sampling_params = None
    tokenizer = processor = None
    if not use_dp:
        print("[Init policy_mllm]")
        policy_mllm, sampling_params = vllm_mllm_init(
            args.policy_model_path,
            tp=args.policy_mllm_tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.policy_model_path, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(args.policy_model_path)
    else:
        print("[Init DP persistent worker pool]")
        dp_pool = Stage1PolicyDPWorkerPool(
            dp_device_groups=dp_groups,
            policy_model_path=args.policy_model_path,
            tp=tp,
            max_model_len=args.max_model_len,
            batch_size=args.policy_batch,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

    # 读取样本并按批处理
    raw_samples = iter_samples(args.dataset_name, dataset_path)
    total = len(raw_samples)
    if args.limit > 0:
        total = min(total, args.limit)

    sid = args.start_id
    kept = 0
    skipped_parse = 0
    skipped_overlen = 0
    skipped_policy_correct = 0

    parsed_recs_batch: List[Dict[str, Any]] = []
    convs_batch: List[list] = []
    questions_batch: List[str] = []

    def flush_batch():
        nonlocal sid, kept, skipped_overlen, skipped_policy_correct
        if not parsed_recs_batch:
            return
        # 推理
        if use_dp:
            raw_outs, extr_outs, keep_mask = run_policy_batch(
                None,
                convs_batch,
                None,
                None,
                None,
                max_model_len=args.max_model_len,
                dp_device_groups=dp_groups,
                policy_model_path=args.policy_model_path,
                tp=tp,
                batch_size=args.policy_batch,
                gpu_memory_utilization=args.gpu_memory_utilization,
                persistent_pool=dp_pool,
            )
        else:
            raw_outs, extr_outs, keep_mask = run_policy_batch(
                policy_mllm,
                convs_batch,
                sampling_params,
                processor,
                tokenizer,
                max_model_len=args.max_model_len,
                batch_size=args.policy_batch,
                dp_device_groups=None,
                policy_model_path=None,
                tp=tp,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
        # 判分
        gts = [r["gt_answer_text"] for r in parsed_recs_batch]
        judged = sequential_judge_predictions(
            extr_outs,
            gts,
            judge_mode=args.judge_mode,
            dataset_name=args.dataset_name,
            judge_llm_dir=args.judge_llm_dir,
            judge_llm_tensor_parallel_size=args.judge_llm_tensor_parallel_size,
            questions=questions_batch,
            api_name=args.api_name,
            api_max_workers=args.api_max_workers,
            api_kwargs={"temperature": args.api_temperature} if args.api_temperature is not None else None,
        )
        # 逐条写出（append）
        for rec, raw_pred, extr_pred, keep, ok in zip(parsed_recs_batch, raw_outs, extr_outs, keep_mask, judged):
            if not keep:
                skipped_overlen += 1
                continue
            if ok:
                skipped_policy_correct += 1
                continue
            # 保存图像
            paths = save_images_for_sample(sid, out_img_dir, rec["main_image"], rec["helpers"])
            main_path = paths[0]
            # 回填 helper 路径并释放 PIL
            img_idx = 1
            for h in rec["helpers"]:
                if h.get("image") is not None:
                    h["image_path"] = paths[img_idx]
                    img_idx += 1
                else:
                    h["image_path"] = None
                h["image"] = None
            out_item = {
                "dataset_name": args.dataset_name,
                "stage1_id": sid,
                "orig_idx": rec.get("orig_idx"),
                "qid": rec.get("qid"),
                "question": rec["question"],
                "gt_choices": rec["gt_choices"],
                "gt_answer_text": rec["gt_answer_text"],
                "image_main": main_path,
                "helpers": rec["helpers"],
                "policy_pred_raw": raw_pred,
                "policy_pred_extracted": extr_pred,
                "policy_correct": False,
            }
            writer.write(json.dumps(out_item, ensure_ascii=False) + "\n")
            kept += 1
            sid += 1
        writer.flush()

        # 清空批缓存
        parsed_recs_batch.clear()
        convs_batch.clear()
        questions_batch.clear()

    # 解析+积累批次
    total_iter = total if args.limit > 0 else len(raw_samples)
    for idx, samp in enumerate(tqdm(raw_samples[:total_iter], desc="Parse+enqueue")):
        if args.limit > 0 and idx >= args.limit:
            break
        rec = None
        if args.dataset_name == "PixelReasoner":
            rec = parse_pixelreasoner(samp, dataset_images_root)
        elif "CoM" in args.dataset_name:
            rec = parse_com(samp, dataset_images_root, w_mathvista=("w_MathVista" in args.dataset_name))
        elif args.dataset_name == "CoF":
            rec = parse_cof(samp, dataset_images_root)
        elif args.dataset_name == "ReFocus":
            rec = parse_refocus(samp, dataset_images_root)
        elif args.dataset_name == "Visual_CoT":
            rec = parse_visual_cot_all(samp, dataset_images_root, args.dataset_name, idx)
        elif args.dataset_name == "Visual_CoT_v7w":
            rec = parse_visual_cot_w_choices(samp, dataset_images_root, args.dataset_name, idx)
        elif args.dataset_name == "Visual_CoT_gqa":
            rec = parse_visual_cot_wo_choices(samp, dataset_images_root, args.dataset_name, idx)
        elif "Zebra_CoT" in args.dataset_name:
            rec = parse_zebra_cot(samp, dataset_images_root)
        elif "VTS" in args.dataset_name:
            rec = parse_vts(samp, dataset_images_root)
        if rec is None:
            skipped_parse += 1
            continue
        rec["orig_idx"] = idx
        conv = build_policy_conversation(rec["question"], rec["main_image"])
        parsed_recs_batch.append(rec)
        convs_batch.append(conv)
        questions_batch.append(rec["question"])
        if len(parsed_recs_batch) >= args.batch:
            flush_batch()

    # 尾批
    flush_batch()

    writer.close()
    if policy_mllm is not None:
        vllm_kill_model(policy_mllm)
    if dp_pool is not None:
        dp_pool.close()

    print("================ SUMMARY ================")
    print(f"dataset_name         : {args.dataset_name}")
    print(f"raw_samples          : {total}")
    print(f"skipped_parse_fail   : {skipped_parse}")
    print(f"skipped_token_overlen: {skipped_overlen}")
    print(f"skipped_policy_correct: {skipped_policy_correct}")
    print(f"stage1_kept (wrong)  : {kept}")
    print(f"output_jsonl         : {out_jsonl}")
    print("==========================================")


if __name__ == "__main__":
    main()
