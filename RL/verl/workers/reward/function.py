# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import os
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, TypedDict
import re
import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto, DataProtoItem
from .config import RewardConfig, RuleBasedJudgeConfig
import numpy as np
import pdb
import inspect
class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


SequentialRewardFunction = Callable[[str, str], RewardScore]

BatchRewardFunction = Callable[[List[str], List[str]], List[RewardScore]]

SingleRuleBasedJudgeFunction = Callable[[str, str], RewardScore]

BatchRuleBasedJudgeFunction = Callable[[List[str], List[str]], List[RewardScore]]


class FunctionRewardManager(ABC):
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.config = config
        self.tokenizer = tokenizer

    @abstractmethod
    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """Compute reward for a batch of data."""
        ...


class SequentialFunctionRewardManager(FunctionRewardManager):
    reward_fn: SequentialRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        response_ids = data.batch["responses"]
        response_length = data.batch["response_mask"].sum(dim=-1)
        for i in range(len(data)):
            valid_response_ids = response_ids[i][: response_length[i]]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            ground_truth = data.non_tensor_batch["ground_truth"][i]

            score = self.reward_fn(response_str, ground_truth)
            reward_tensor[i, response_length[i] - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


def replace_abs_vis_token_content(s: str) -> str:
    pattern = re.compile(r'(<abs_vis_token>)(.*?)(</abs_vis_token>)', flags=re.DOTALL)
    return pattern.sub(r'\1<latent>\3', s)

class BatchFunctionRewardManager(FunctionRewardManager):
    reward_fn: BatchRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        response_str, ground_truth = [], []
        response_ids = data.batch["responses"]
        response_length = data.batch["response_mask"].sum(dim=-1)
        for i in range(len(data)):
            valid_response_ids = response_ids[i][: response_length[i]]
            if "avt" in self.config.reward_function:
                response_str_=replace_abs_vis_token_content(self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)).replace("<|endoftext|>", "").replace("<|im_end|>", "")
            else:
                response_str_=self.tokenizer.decode(valid_response_ids, skip_special_tokens=self.config.skip_special_tokens)
            response_str.append(response_str_)
            ground_truth.append(data.non_tensor_batch["ground_truth"][i])

        #breakpoint()
        extra_kwargs = {}
        try:
            sig = inspect.signature(self.reward_fn)
            if "length_penalty_weight" in sig.parameters:
                extra_kwargs["length_penalty_weight"] = self.config.length_penalty_weight
        except Exception:
            pass

        if "single_step_rewards" in data.non_tensor_batch:  # mc List[float]
            scores = self.reward_fn(response_str, data.non_tensor_batch["single_step_rewards"])
        elif "full_step_rewards" in data.non_tensor_batch:  # mc2 List[List[float]]
            scores = self.reward_fn(
            response_str,
            data.non_tensor_batch["full_step_rewards"],
            resp_lengths=response_length,
            ref_resp_lengths=data.non_tensor_batch["ref_resp_lengths"],
            **extra_kwargs,
            )
        elif "correctness" in data.non_tensor_batch:  # avt List[bool]
            #pdb.set_trace()
            scores = self.reward_fn(
            response_str,
            data.non_tensor_batch["correctness"],
            resp_lengths=response_length,
            ref_resp_lengths=data.non_tensor_batch["ref_resp_lengths"],
            **extra_kwargs,
            )
        else:
            scores = self.reward_fn(
            response_str,
            ground_truth,
            resp_lengths=response_length,
            ref_resp_lengths=data.non_tensor_batch["ref_resp_lengths"],
            **extra_kwargs,
            )
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i, score in enumerate(scores):
            if "overall" in score: # mc and greedy
                 reward_tensor[i, response_length[i] - 1] = score["overall"]
            elif "overall_step_wise" in score: # mc2
                poss = data.non_tensor_batch["step_end_positions"][i]
                reward_tensor[i, poss] = torch.tensor(score["overall_step_wise"], dtype=reward_tensor.dtype)
                
            for key, value in score.items():
                if not (isinstance(value, np.floating) or isinstance(value, float)):
                    continue
                reward_metrics[key].append(value)
        #breakpoint()
        return reward_tensor, reward_metrics







class FunctionRuleBasedJudgeManager(ABC):
    """RuleBasedJudge manager for rule-based rule_based_judge."""

    def __init__(self, config: RuleBasedJudgeConfig, tokenizer: PreTrainedTokenizer):
        if config.judge_function is None:
            raise ValueError("RuleBasedJudge function is not provided.")

        if not os.path.exists(config.judge_function):
            raise FileNotFoundError(f"RuleBasedJudge function file {config.judge_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_rule_based_judge_fn", config.judge_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_rule_based_judge_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load rule_based_judge function: {e}")

        if not hasattr(module, config.judge_function_name):
            raise AttributeError(f"Module {module} does not have function {config.judge_function_name}.")

        rule_based_judge_fn = getattr(module, config.judge_function_name)
        print(f"Using rule_based_judge function `{config.judge_function_name}` from `{config.judge_function}`.")
        self.rule_based_judge_fn = rule_based_judge_fn
        self.config = config
        self.tokenizer = tokenizer

    @abstractmethod
    def compute_rule_based_judge(self, data: DataProto) -> bool:
        """Compute rule_based_judge for a batch of data."""
        ...
    
    def compute_rule_based_judge_with_string(self, response_str: str, ground_truth: str) -> bool:
        """Compute rule_based_judge for a single response string."""
        ...


class SingleFunctionRuleBasedJudgeManager(FunctionRuleBasedJudgeManager):
    rule_based_judge_fn: SingleRuleBasedJudgeFunction

    def compute_rule_based_judge(self, data: DataProtoItem) -> bool:
        response_ids = data.batch["responses"]
        response_length = data.batch["response_mask"].sum(dim=-1)

        valid_response_ids = response_ids[: response_length]
        response_str = self.tokenizer.decode(
            valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
        )
        ground_truth = data.non_tensor_batch["ground_truth"]
        try:
            correctness = self.rule_based_judge_fn(response_str, ground_truth)
        except Exception as e:
            print(f"Rule-based judge error: {e}")
            correctness = False
        return correctness, response_str

    def compute_rule_based_judge_with_string(self, response_str: str, ground_truth: str) -> bool:
        try:
            correctness = self.rule_based_judge_fn(response_str, ground_truth)
        except Exception as e:
            print(f"Rule-based judge error: {e}")
            correctness = False
        return correctness

class BatchFunctionRuleBasedJudgeManager(FunctionRuleBasedJudgeManager):
    rule_based_judge_fn: BatchRuleBasedJudgeFunction

    def compute_rule_based_judge(self, data: DataProto) -> List[bool]:
        correctness = []
        response_strs = []
        response_ids = data.batch["responses"]
        response_length = data.batch["response_mask"].sum(dim=-1)
        for i in range(len(data)):
            valid_response_ids = response_ids[i][: response_length[i]]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=self.config.skip_special_tokens)
            response_strs.append(response_str)
            ground_truth = data.non_tensor_batch["ground_truth"][i]
            try:
                correctness = self.rule_based_judge_fn(response_str, ground_truth)
            except Exception as e:
                print(f"Rule-based judge error: {e}")
                correctness = False
            correctness.append(correctness)
        return correctness

