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
"""
Reward config
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RewardConfig:
    reward_type: str = "batch"
    reward_function: Optional[str] = None
    reward_function_kwargs: dict = field(default_factory=dict)
    skip_special_tokens: bool = True
    num_cpus: int = 1
    length_penalty_weight: float = 0.001
    repetition_penalty: bool = False
    """auto keys"""
    reward_function_name: Optional[str] = field(default=None, init=False)

    def post_init(self):
        if self.reward_function is not None:  # support custom reward function, e.g., ./math.py:main
            if ":" not in self.reward_function:
                self.reward_function_name = "main"
            else:
                self.reward_function, self.reward_function_name = self.reward_function.rsplit(":", maxsplit=1)

            if os.path.exists(self.reward_function):  # ray job uses absolute path
                self.reward_function = os.path.abspath(self.reward_function)
            else:
                self.reward_function = None


@dataclass
class RuleBasedJudgeConfig:
    judge_type: str = "batch"
    judge_function: Optional[str] = None
    skip_special_tokens: bool = True
    num_cpus: int = 1
    judge_server_name: str = "rule_based_judge_server"
    api_key: Optional[str] = None
    api_name: str = "gemini-2.5-pro"
    api_kwargs: dict = field(default_factory=dict)

    def post_init(self):
        if self.judge_function is not None:  # support custom reward function, e.g., ./math.py:main
            if ":" not in self.judge_function:
                self.judge_function_name = "main"
            else:
                self.judge_function, self.judge_function_name = self.judge_function.rsplit(":", maxsplit=1)

            if os.path.exists(self.judge_function):  # ray job uses absolute path
                self.judge_function = os.path.abspath(self.judge_function)
            else:
                self.judge_function = None