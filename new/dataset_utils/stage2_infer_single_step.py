#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage-2 (Part 1, single-step) — inference only.

For each sample, run strong MLLM once with a prompt that contains:
- Question text first
- All helper images concatenated (each as {"type": "image", "image": PIL.Image})
- Then a tail text: "Here are the helper images that can help you answer the question."

Output (file A) keeps minimal IDs and a single-item `predictions` list per sample.
Other logic and CLI remain consistent with stage2_infer, but we do not iterate steps.
"""

from __future__ import annotations
import argparse, io, json, os, gc
from collections import deque
from typing import Any, Dict, List, Optional
import multiprocessing as mp
from itertools import islice
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor

from AAA_vllm_toolkit.load_and_gen_vllm import (
    vllm_mllm_init,
    vllm_mllm_process_batch_from_messages,
    vllm_wake_model,
    vllm_kill_model,
    count_qwen_vl_tokens,
)
from AAA_vllm_toolkit.extract_and_check import extract_boxed_answer


# ---------- helpers ----------------------------------------------------------
def iter_stage1(path: str):
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            for rec in json.load(f):
                if not rec.get("policy_correct"):
                    yield rec
        else:
            for ln in islice(f, 0, None):
                if ln.strip():
                    rec = json.loads(ln)
                    if not rec.get("policy_correct"):
                        yield rec


def load_image(path: str):
    with open(path, "rb") as f:
        b = f.read()
    im = Image.open(io.BytesIO(b)).convert("RGB")
    im.load()
    return im


def build_conv(rec: Dict[str, Any]):
    """Build a single conversation:
    - question first
    - then ALL helper images (if file exists), each as {type: image, image: PIL.Image}
    - finally append a tail text hint
    """
    content = [{"type": "text", "text": rec["question"]}]
    opened = []
    helpers = rec.get("helpers", [])
    for h in helpers:
        p = h.get("image_path")
        if p and os.path.isfile(p):
            try:
                img = load_image(p)
                content.append({"type": "image", "image": img})
                opened.append(img)
            except Exception as e:
                print(f"[WARN] {p}: {e}")
    content.append({
        "type": "text",
        "text": "Here are the helper images that can help you answer the question.",
    })
    return [{"role": "user", "content": content}], opened


def run_batch_single(
    recs: List[Dict[str, Any]],
    model,
    processor,
    tokenizer,
    token_limit: int,
    sampling_params,
):
    def _has_image(conv):
        return any(it.get("type") == "image" for msg in conv for it in msg["content"])

    def _all_image_size_valid(conv):
        for msg in conv:
            for it in msg["content"]:
                if it.get("type") == "image":
                    img = it.get("image")
                    if img.size[0] <= 0 or img.size[1] <= 0:
                        return False
                    if img.size[0]/img.size[1] >= 200 or img.size[0]/img.size[1] <= 1/200:
                        return False
        return True

    convs, eff_idx, opened_pool = [], [], []
    for i, r in enumerate(recs):
        conv, opened = build_conv(r)
        # 跳过无图或异常尺寸
        if not _has_image(conv) or not _all_image_size_valid(conv):
            convs.append(None)
            continue
        convs.append(conv)
        eff_idx.append(i)
        opened_pool.extend(opened)

    if not eff_idx:
        return [None] * len(recs)

    eff_convs = [convs[i] for i in eff_idx]
    inputs = vllm_mllm_process_batch_from_messages(eff_convs, processor)
    tcnts = count_qwen_vl_tokens(inputs, tokenizer, processor)

    gen_idx, gen_convs = [], []
    for sub_i, t in enumerate(tcnts):
        if token_limit and t and t > token_limit:
            continue
        gen_idx.append(sub_i)
        gen_convs.append(eff_convs[sub_i])

    answers_sub: Dict[int, Optional[str]] = {}

    valid_cnt = len(gen_convs)
    print(f"Single-step: {valid_cnt} remaining")
    if gen_convs:
        gen_inputs = vllm_mllm_process_batch_from_messages(gen_convs, processor)
        outs = model.generate(gen_inputs, sampling_params=sampling_params, use_tqdm=True)
        for sub_i, o in zip(gen_idx, outs):
            txt = o.outputs[0].text.strip() if o.outputs else ""
            answers_sub[sub_i] = extract_boxed_answer(txt)

    for img in opened_pool:
        try:
            img.close()
        except Exception:
            pass

    ret = [None] * len(recs)
    for k, ridx in enumerate(eff_idx):
        ret[ridx] = answers_sub.get(k)
    return ret


# ---------------- DP helpers -------------------------
def _dp_stage2_worker_entry_single(
    queue,
    device_ids: List[int],
    recs: List[Dict[str, Any]],
    model_path: str,
    tp: int,
    token_limit: int,
    shard_idx: int,
):
    try:
        import os as _os
        from transformers import AutoTokenizer as _AutoTokenizer, AutoProcessor as _AutoProcessor
        from AAA_vllm_toolkit.load_and_gen_vllm import (
            vllm_mllm_init as _vllm_mllm_init,
            vllm_kill_model as _vllm_kill_model,
        )
        _os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in device_ids)
        tok = _AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        proc = _AutoProcessor.from_pretrained(model_path)
        model, sampling = _vllm_mllm_init(model_path, tp=tp)
        try:
            preds = run_batch_single(recs, model, proc, tok, token_limit, sampling)
        finally:
            try:
                _vllm_kill_model(model)
            except Exception:
                pass
        queue.put((shard_idx, preds))
    except Exception:
        import traceback as _tb
        _tb.print_exc()
        queue.put((shard_idx, [None] * len(recs)))


def run_batch_single_dp(
    recs: List[Dict[str, Any]],
    dp_device_groups: List[List[int]],
    model_path: str,
    tp: int,
    token_limit: int,
) -> List[Optional[str]]:
    num_workers = len(dp_device_groups)
    n = len(recs)
    if n == 0:
        return []
    # contiguous split
    shard_sizes = [(n + i) // num_workers for i in range(num_workers)]
    total_assigned = sum(shard_sizes)
    if total_assigned != n:
        shard_sizes[0] += (n - total_assigned)
    shards = []
    idx = 0
    for sz in shard_sizes:
        shards.append(recs[idx : idx + sz])
        idx += sz

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs: List[mp.Process] = []
    for shard_idx, (devs, shard) in enumerate(zip(dp_device_groups, shards)):
        if not shard:
            queue.put((shard_idx, []))
            continue
        p = ctx.Process(
            target=_dp_stage2_worker_entry_single,
            args=(queue, devs, shard, model_path, tp, token_limit, shard_idx),
        )
        p.daemon = False
        p.start()
        procs.append(p)

    received = 0
    outputs = [None] * num_workers  # type: ignore
    while received < num_workers:
        shard_idx, preds = queue.get()
        outputs[shard_idx] = preds
        received += 1

    for p in procs:
        p.join()

    ret: List[Optional[str]] = []
    for out in outputs:
        if out is None:
            continue
        ret.extend(out)
    return ret


# ---------- main -------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser("stage2_infer_single_step")
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--strong_mllm_tensor_parallel_size", type=int, default=4)
    ap.add_argument("--devices", default="0,1,2,3")
    ap.add_argument("--token-limit", type=int, default=8192)
    ap.add_argument("--max-batch", type=int, default=0)
    # judge-side params kept for CLI 兼容但无效
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    # 设备与 DP 判定
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    devices_list = [int(x) for x in args.devices.split(",") if x.strip() != ""]
    tp = args.strong_mllm_tensor_parallel_size
    dp_degree = len(devices_list) // tp if tp > 0 else 0
    use_dp = dp_degree > 1
    dp_groups = [devices_list[i * tp : (i + 1) * tp] for i in range(dp_degree)] if use_dp else []

    if not use_dp:
        model, sampling = vllm_mllm_init(
            args.model_path, tp=args.strong_mllm_tensor_parallel_size
        )
        tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        proc = AutoProcessor.from_pretrained(args.model_path)

    writer = open(args.out, "w", encoding="utf-8")
    active: deque = deque()

    cnt = 0
    for rec in iter_stage1(args.stage1):
        if args.max_samples is not None and cnt >= args.max_samples:
            break
        cnt += 1
        active.append({"rec": rec, "preds": [None]})
        if args.max_batch and len(active) >= args.max_batch:
            flush_single(
                active,
                writer,
                model if not use_dp else None,
                proc if not use_dp else None,
                tok if not use_dp else None,
                args.token_limit,
                sampling if not use_dp else None,
                use_dp=use_dp,
                dp_groups=dp_groups,
                model_path=args.model_path,
                tp=tp,
            )
            active.clear()

    if active:
        flush_single(
            active,
            writer,
            model if not use_dp else None,
            proc if not use_dp else None,
            tok if not use_dp else None,
            args.token_limit,
            sampling if not use_dp else None,
            use_dp=use_dp,
            dp_groups=dp_groups,
            model_path=args.model_path,
            tp=tp,
        )

    writer.close()


def flush_single(
    states: deque,
    writer,
    model,
    proc,
    tok,
    token_limit,
    sampling_params,
    *,
    use_dp: bool = False,
    dp_groups: Optional[List[List[int]]] = None,
    model_path: Optional[str] = None,
    tp: int = 1,
):
    batch_recs = [st["rec"] for st in states]
    if use_dp:
        preds = run_batch_single_dp(
            batch_recs,
            dp_groups or [],
            model_path=model_path,
            tp=tp,
            token_limit=token_limit,
        )
    else:
        preds = run_batch_single(
            batch_recs, model, proc, tok, token_limit, sampling_params
        )
    for st, p in zip(states, preds):
        st["preds"][0] = p

    for st in states:
        out = {
            "dataset_name": st["rec"]["dataset_name"],
            "orig_idx": st["rec"]["orig_idx"],
            "predictions": st["preds"],
        }
        json.dump(out, writer, ensure_ascii=False)
        writer.write("\n")
    writer.flush()
    gc.collect()


if __name__ == "__main__":
    main()
