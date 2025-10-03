#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
判分流水线（可复用）
按 quick -> data_spec -> llm 的顺序逐步判断，只对前一阶段判错的样本继续后续判断；
如果某种判分方法未在 judge_mode 中指定，则跳过该方法。

依赖 AAA_vllm_toolkit 中已有判分与 LLM 工具函数。
"""

from typing import List, Optional, Sequence, Callable, Any, Tuple
import os
import concurrent.futures as cf
import traceback

from AAA_vllm_toolkit.load_and_gen_vllm import (
    vllm_llm_init,
    vllm_kill_model,
)
from AAA_vllm_toolkit.extract_and_check import (
    quick_batch_judge,
    data_spec_batch_judge,
    llm_batch_extract,
    llm_batch_judge,
)
from AAA_vllm_toolkit.api import get_api_response


def _strip_boxed_instruction(q: str) -> str:
    if not isinstance(q, str):
        return q
    return (
        q.replace("Put the letter of your choice within \\boxed{}.", "")
        .replace("Put your final answer within \\boxed{}.", "")
        .replace("Given the answer in a single word and put it within \\boxed{}.", "")
        .strip()
    )


def judge_wrap_fn(pred: Optional[str], gt: Optional[str], question: Optional[str]) -> Tuple[str, str]:
    """构建用于 API 判分的 system 与 user prompt。
    目标：让模型仅输出 yes 或 no，以判断 pred 是否与 gt 一致（可语义等价）。
    返回: (sys_prompt, user_prompt)
    """
    sys_prompt = (
        "You are a strict answer judge. Given the question, a model's predicted answer, and the ground-truth answer, "
        "determine if the prediction is correct. Consider semantic equivalence, case/format variations, "
        "and numeric equivalence if applicable. Only reply with 'yes' or 'no'."
    )
    user_prompt = (
        f"Question: {question if question is not None else ''}\n"
        f"Predicted Answer: {pred if pred is not None else ''}\n"
        f"Ground Truth Answer: {gt if gt is not None else ''}\n"
        "Does the predicted answer exactly or semantically match the ground-truth? Reply 'yes' or 'no'."
    )
    return sys_prompt, user_prompt


def _api_call_wrapper(
    api_name: str,
    pred: Optional[str],
    gt: Optional[str],
    question: Optional[str],
    dataset_name: str,
    api_kwargs: Optional[dict] = None,
) -> Optional[bool]:
    """子进程内执行：构造判分提示并调用指定 API（gemini-2.5-pro / deepseek-chat）。
    若三次调用都失败，返回 None 以让上层保留之前的判分结果。
    """
    try:
        # 缺少必要字段直接判错，避免浪费 API 额度
        if not pred or pred is None or gt is None:
            return False
        sys_prompt, user_prompt = judge_wrap_fn(pred, gt, question)
        # 最多 3 次重试
        attempts = 3
        last_text = None
        for _ in range(attempts):
            responses = get_api_response(api_name, sys_prompt, [user_prompt], **(api_kwargs or {}))
            if responses and isinstance(responses[0], str) and responses[0].strip():
                last_text = responses[0]
                break
        if last_text is None:
            return None  # 三次都失败，交由上层保留之前结果
        t = last_text.strip().lower()
        if "yes" in t and "no" not in t:
            return True
        if "no" in t and "yes" not in t:
            return False
        # 模糊/空输出按错处理
        return False
    except Exception:
        traceback.print_exc()
        return None


def sequential_judge_predictions(
    extr_outs: List[Optional[str]],
    gts: List[Optional[str]],
    judge_mode: Optional[Sequence[str]] = None,
    *,
    dataset_name: str = "",
    judge_llm_dir: Optional[str] = None,
    judge_llm_tensor_parallel_size: int = 2,
    questions: Optional[List[str]] = None,
    # API 判分相关（可选）
    api_name: Optional[str] = None,
    api_max_workers: int = 32,  # 默认并行度，可由环境变量 API_JUDGE_WORKERS 覆盖
    api_kwargs: Optional[dict] = None,  # 透传给 API 的其他参数（如 temperature, model_name for deepseek）
) -> List[bool]:
    """
    顺序：quick -> data_spec -> llm。
    - 仅对上一阶段判为错的样本继续判断。
    - 未在 judge_mode 中声明的方法会被跳过。

    返回与 extr_outs 同长的 bool 列表，True 表示“判为正确”。
    """
    n = len(extr_outs)
    if len(gts) != n:
        raise ValueError("Length mismatch: extr_outs and gts must be same length")

    judge_mode = list(judge_mode or [])
    ordered_modes = [m for m in ["quick", "data_spec", "llm", "api"] if m in judge_mode]

    judged: List[bool] = [False] * n  # True 表示已被判为正确

    # 预处理问题（去掉 boxed 指令）
    questions_wo_inst: Optional[List[str]] = None
    if questions is not None:
        questions_wo_inst = [_strip_boxed_instruction(q) for q in questions]

    # 阶段 1：quick
    if "quick" in ordered_modes:
        idxs = [i for i in range(n) if not judged[i]]
        if idxs:
            sub_pred = [extr_outs[i] for i in idxs]
            sub_gt = [gts[i] for i in idxs]
            results = quick_batch_judge(sub_pred, sub_gt)
            for i, r in zip(idxs, results):
                judged[i] = bool(r)

    # 阶段 2：data_spec
    if "data_spec" in ordered_modes:
        idxs = [i for i in range(n) if not judged[i]]
        if idxs:
            sub_pred = [extr_outs[i] for i in idxs]
            sub_gt = [gts[i] for i in idxs]
            results = data_spec_batch_judge(sub_pred, sub_gt, dataset_name)
            for i, r in zip(idxs, results):
                judged[i] = bool(r)

    # 阶段 3：llm
    if "llm" in ordered_modes:
        idxs = [i for i in range(n) if not judged[i]]
        if idxs and judge_llm_dir:
            judge_llm, _ = vllm_llm_init(
                judge_llm_dir, tp=judge_llm_tensor_parallel_size
            )
            try:
                sub_pred = [extr_outs[i] for i in idxs]
                sub_gt = [gts[i] for i in idxs]
                sub_q = [questions_wo_inst[i] for i in idxs] if questions_wo_inst else None

                # VTS 需要先抽取 GT
                if "VTS" in dataset_name:
                    sub_gt_extracted = llm_batch_extract(
                        sub_gt, judge_llm, questions=sub_q if sub_q else None, dataset_name="VTS"
                    )
                else:
                    sub_gt_extracted = sub_gt

                results = llm_batch_judge(
                    sub_pred,
                    sub_gt_extracted,
                    judge_llm,
                    questions=sub_q if sub_q else None,
                )
                for i, r in zip(idxs, results):
                    judged[i] = bool(r)
            finally:
                # 主动释放 judge LLM
                try:
                    vllm_kill_model(judge_llm)
                except Exception:
                    pass

    # 阶段 4：api（位于 llm 之后，仅对仍未判定为正确的样本并行调用外部 API）
    if "api" in ordered_modes:
        idxs = [i for i in range(n) if not judged[i]]
        # 允许从环境变量读取 API 名称
        api_name_eff = api_name or os.environ.get("API_JUDGE_NAME")
        if not idxs or not api_name_eff:
            pass  # 无待判断或未设置 api_name
        else:
            try:
                max_workers = int(os.environ.get("API_JUDGE_WORKERS", api_max_workers))
            except Exception:
                max_workers = api_max_workers

            sub_pred = [extr_outs[i] for i in idxs]
            sub_gt = [gts[i] for i in idxs]
            sub_q = [questions_wo_inst[i] for i in idxs] if questions_wo_inst else [None] * len(idxs)

            results_map: dict[int, Optional[bool]] = {}
            with cf.ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = [
                    ex.submit(
                        _api_call_wrapper,
                        api_name_eff,
                        p,
                        g,
                        q,
                        dataset_name,
                        api_kwargs,
                    )
                    for p, g, q in zip(sub_pred, sub_gt, sub_q)
                ]
                future_to_idx = {fut: i for fut, i in zip(futures, idxs)}
                for fut in cf.as_completed(futures):
                    i_future = future_to_idx[fut]
                    try:
                        results_map[i_future] = fut.result()  # Optional[bool]
                    except Exception:
                        traceback.print_exc()
                        results_map[i_future] = False

            for i in idxs:
                r = results_map.get(i, None)
                if r is None:
                    # 三次调用都失败，保留之前的判分结果（即不修改 judged[i]）
                    continue
                judged[i] = bool(r)

    return judged
