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

import apply_monet_vllm
import json
import pdb
import ray
from omegaconf import OmegaConf

from ..single_controller.ray import RayWorkerGroup
from ..utils.tokenizer import get_processor, get_tokenizer
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import BatchFunctionRewardManager, SequentialFunctionRewardManager, BatchFunctionRuleBasedJudgeManager, SingleFunctionRuleBasedJudgeManager
from .config import PPOConfig
from .data_loader import create_dataloader
from .ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
import os
from verl.trainer.save_any_log import setup_tee_logger
import datetime

# please make sure main_task is not scheduled on head
@ray.remote(num_cpus=2)
class Runner:
    """A runner for RL training."""

    def run(self, config: PPOConfig):
        #print(os.environ["CUDA_VISIBLE_DEVICES"])
        # print config
        #print(json.dumps(config.to_dict(), indent=2))
        import torch, os, sys
        print("Torch version :", torch.__version__)
        print("Torch path    :", torch.__file__)
        print("Python exec   :", sys.executable)
        print("CUDA_VISIBLE_DEVICES :", os.environ.get("CUDA_VISIBLE_DEVICES"))
        print("http_proxy =", os.getenv("http_proxy"))
        print("https_proxy=", os.getenv("https_proxy"))
        # instantiate tokenizer
        tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        #breakpoint()
        processor = get_processor(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )

        #TRAIN_ENV = {"env": {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}}
        # define worker classes
        ray_worker_group_cls = RayWorkerGroup
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(FSDPWorker),
            Role.Critic: ray.remote(FSDPWorker),
            Role.RefPolicy: ray.remote(FSDPWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        if config.worker.reward.reward_type == "sequential":
            RewardManager = SequentialFunctionRewardManager
        elif config.worker.reward.reward_type == "batch":
            RewardManager = BatchFunctionRewardManager
        else:
            raise NotImplementedError(f"Unknown reward type {config.worker.reward.reward_type}.")

        if config.worker.rule_based_judge.judge_type == "single":
            RuleBasedJudgeManager = SingleFunctionRuleBasedJudgeManager
        elif config.worker.rule_based_judge.judge_type == "batch":
            RuleBasedJudgeManager = BatchFunctionRuleBasedJudgeManager
        else:
            raise NotImplementedError(f"Unknown reward type {config.worker.rule_based_judge.judge_type}.")

        RemoteRewardManager = ray.remote(RewardManager).options(num_cpus=config.worker.reward.num_cpus)
        reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)
        val_reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)
        
        RemoteRuleBasedJudgeManager = ray.remote(RuleBasedJudgeManager).options(num_cpus=config.worker.rule_based_judge.num_cpus, name="rule_based_judge_server")
        config.worker.rule_based_judge.judge_server_name = "rule_based_judge_server"
        rule_based_judge_fn = RemoteRuleBasedJudgeManager.remote(config.worker.rule_based_judge, tokenizer)

        train_dataloader, val_dataloader = create_dataloader(config.data, tokenizer, processor)
        
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            rule_based_judge=rule_based_judge_fn,
        )
        trainer.init_workers()
        trainer.fit()


def main():
    cli_args = OmegaConf.from_cli()
    default_config = OmegaConf.structured(PPOConfig())
    
    
    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)

    ppo_config = OmegaConf.merge(default_config, cli_args)
    ppo_config: PPOConfig = OmegaConf.to_object(ppo_config)
    ppo_config.deep_post_init()
    #time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #setup_tee_logger(f"training_logs/detailed_logs/{ppo_config.trainer.experiment_name}_{time_str}.txt")
    if not ray.is_initialized():
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
                "PYTHONUNBUFFERED": "1",
                # Respect user's verbosity preferences; default to quiet unless explicitly enabled
                "RAY_DEBUG": os.getenv("RAY_DEBUG", "0"),
                "RAY_LOG_TO_STDERR": os.getenv("RAY_LOG_TO_STDERR", "0"),
            }
        }
        # Allow forcing Ray local_mode (single-process) for debugging when environment is unstable
        local_mode = os.getenv("RAY_LOCAL_MODE", "0").lower() in ("1", "true", "yes")
        # Respect explicit local forcing to avoid connecting to an external cluster by accident
        # Priority: USE_RAY_LOCAL=1 -> address="local"; else honor RAY_ADDRESS if set; else None
        address = None
        try:
            _force_local = os.getenv("USE_RAY_LOCAL", "1").lower() in ("1", "true", "yes")
        except Exception:
            _force_local = True
        if _force_local:
            address = "local"
        else:
            address = os.getenv("RAY_ADDRESS", None)
        # Configure robust temp and spilling directories with short paths to avoid AF_UNIX 107-byte limit
        default_spill = "/tmp/ray_spill"
        default_temp = "/tmp/ray_tmp"
        spill_dir = os.path.abspath(os.getenv("RAY_SPILL_DIR", default_spill))
        temp_dir = os.path.abspath(os.getenv("RAY_TMPDIR", default_temp))
        os.makedirs(spill_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        try:
            object_store_mem = int(os.getenv("RAY_OBJECT_STORE_MEMORY", str(128 * 1024 ** 2)))
        except Exception:
            object_store_mem = 128 * 1024 ** 2
        # Extend worker register timeout; env override if provided
        try:
            register_timeout = int(os.getenv("RAY_WORKER_REGISTER_TIMEOUT_SECONDS", "300"))
        except Exception:
            register_timeout = 300
        if local_mode:
            os.environ.update({
                "RANK": "0",
                "WORLD_SIZE": "1",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": "29500"
            })
        # Cap advertised resources to avoid spawning excessive worker processes that can starve agents during startup
        
        #pdb.set_trace()
        try:
            advertised_cpus = int(os.getenv("RAY_NUM_CPUS", "16"))
        except Exception:
            advertised_cpus = 16
        try:
            advertised_gpus = int(os.getenv("RAY_NUM_GPUS", str(getattr(ppo_config.trainer, "n_gpus_per_node", 1))))
        except Exception:
            advertised_gpus = getattr(ppo_config.trainer, "n_gpus_per_node", 1)
        ray.init(
            address=address,
            runtime_env=runtime_env,
            local_mode=local_mode,  # resources={"embed_gpu": 1},
            include_dashboard=False,
            _temp_dir=temp_dir,
            object_store_memory=object_store_mem,
            object_spilling_directory=spill_dir,
            _system_config={
                "worker_register_timeout_seconds": register_timeout,
            },
            num_cpus=advertised_cpus,
            num_gpus=advertised_gpus,
        )
        #ray.util.connect_to_new_cluster(ray_debugger=True)
    runner = Runner.remote()
    ray.get(runner.run.remote(ppo_config))
    
    #runner = Runner()
    #runner.run(ppo_config)


if __name__ == "__main__":
    main()
