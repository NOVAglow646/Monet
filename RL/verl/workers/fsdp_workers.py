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
The main entry point to run the PPO algorithm
"""

from typing import Literal, Optional, Union, List

import numpy as np
import psutil
import torch
import torch.distributed as dist
from accelerate import init_empty_weights
from codetiming import Timer
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    GenerationConfig,
    PreTrainedModel,
)
from transformers.modeling_utils import no_init_weights

from ..models.monkey_patch import apply_ulysses_patch
from ..protocol import DataProto
from ..single_controller.base import Worker
from ..single_controller.base.decorator import Dispatch, register
from ..utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from ..utils.flops_counter import FlopsCounter
from ..utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_fn,
    load_fsdp_model,
    load_fsdp_optimizer,
    offload_fsdp_model,
    offload_fsdp_optimizer,
)
from ..utils.model_utils import print_gpu_memory_usage, print_model_size
from ..utils.tokenizer import get_processor, get_tokenizer
from ..utils.torch_dtypes import PrecisionType
from ..utils.torch_functional import AnyPrecisionAdamW, get_constant_schedule_with_warmup
from .config import ActorConfig, CriticConfig, FSDPConfig, ModelConfig, OptimConfig, RefConfig, WorkerConfig
from .rollout import vLLMRollout
from .sharding_manager import FSDPVLLMShardingManager
from .sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

import os

import ray
from transformers import AutoTokenizer, AutoModel, AutoConfig, GenerationConfig
from transformers.configuration_utils import PretrainedConfig
import torch.nn.functional as F
from tqdm import tqdm
from Monet.RL.examples.reward_function.monet_reward_function import extract_and_check as easyr1_monet_extract_and_check
from Monet.RL.examples.reward_function.monet_reward_function import extract_and_check_api as easyr1_monet_extract_and_check_api
from Monet.RL.examples.reward_function.monet_reward_function import rule_then_api_batch_judge
from tools.custom_api import build_deepseek_client, build_gemini_client
from vllm import LLM
from openai import OpenAI
import datetime
def _to_config(sub: object, parent_model_type: str):
    """Convert a dict sub-config into a proper PretrainedConfig instance."""
    if not isinstance(sub, dict):
        return sub
    mt = sub.get("model_type", parent_model_type or "auto")
    try:
        conf_cls = AutoConfig.for_model(mt)
        return conf_cls.from_dict(sub)
    except Exception:
        return PretrainedConfig.from_dict(sub)

def _sanitize_mm_config(cfg, torch_dtype):
    """Ensure Qwen2.5-VL config is decoder-only and nested dicts are converted."""
    # Force decoder-only
    if getattr(cfg, "is_encoder_decoder", None):
        cfg.is_encoder_decoder = False

    parent_mt = getattr(cfg, "model_type", None) or "auto"

    # Convert nested dicts to Configs
    for key in ("text_config", "vision_config", "decoder", "encoder"):
        sub = getattr(cfg, key, None)
        if isinstance(sub, dict):
            setattr(cfg, key, _to_config(sub, parent_mt))
        # If HF injected a None-like decoder/encoder, make sure it does not exist
        if key in ("decoder", "encoder") and getattr(cfg, key, None) is None:
            try:
                delattr(cfg, key)
            except Exception:
                pass

    # generation_config can be dict or None
    gen = getattr(cfg, "generation_config", None)
    if isinstance(gen, dict):
        try:
            setattr(cfg, "generation_config", GenerationConfig.from_model_config(cfg))
        except Exception:
            setattr(cfg, "generation_config", GenerationConfig.from_dict(gen))
    # Hints HF may read later
    setattr(cfg, "attn_implementation", "flash_attention_2")
    if not hasattr(cfg, "torch_dtype"):
        setattr(cfg, "torch_dtype", torch_dtype)

    return cfg

class FSDPWorker(Worker):
    def __init__(
        self,
        config: WorkerConfig,
        role: Literal["actor", "critic", "rollout", "ref", "actor_rollout", "actor_rollout_ref"],
    ):
        super().__init__()
        self.config = config
        self.role = role
        
        # modified by qixun
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        if not dist.is_initialized():
            # modified by qixun
            dist.init_process_group(backend="nccl",
                                    init_method="env://",
                                    world_size=int(os.environ["WORLD_SIZE"]),
                                    rank=int(os.environ["RANK"]),
                                    timeout=datetime.timedelta(minutes=240),
                                    pg_options=dist.ProcessGroupNCCL.Options(is_high_priority_stream=False))
                                                                            

        # improve numerical stability
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_critic = self.role == "critic"
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        self.embed_model = None
        self.embed_tokenizer = None
            
        if self._is_rollout and "api" in self.config.rule_based_judge.judge_function_name:
            if self.config.rule_based_judge.api_name == 'deepseek-chat':
                self.api_client = build_deepseek_client()
            elif self.config.rule_based_judge.api_name == 'gemini-2.5-pro':
                self.api_client = build_gemini_client()
            else:
                raise ValueError(f"API {self.config.rule_based_judge.api_name} not supported.")

        self._use_param_offload = False
        self._use_optimizer_offload = False
        if self._is_actor:
            self._use_param_offload = self.config.actor.offload.offload_params
            self._use_optimizer_offload = self.config.actor.offload.offload_optimizer
            self._init_config(self.config.actor, "actor")
        elif self._is_critic:
            self._use_param_offload = self.config.critic.offload.offload_params
            self._use_optimizer_offload = self.config.critic.offload.offload_optimizer
            self._init_config(self.config.critic, "critic")
        elif self._is_ref:  # NOTE: it seems that manual offload is slower than FSDP offload
            self._use_param_offload = self.config.ref.offload.offload_params
            self._init_config(self.config.ref, "ref")

    def _init_config(
        self, config: Union[ActorConfig, CriticConfig, RefConfig], role: Literal["actor", "critic", "ref"]
    ):
        world_size = dist.get_world_size()
        fsdp_size = config.fsdp.fsdp_size
        if fsdp_size <= 0 or fsdp_size >= world_size:
            self.device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
        else:  # hsdp
            self.device_mesh = init_device_mesh(
                "cuda", mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=("ddp", "fsdp")
            )

        if config.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(
                    world_size // config.ulysses_sequence_parallel_size,
                    config.ulysses_sequence_parallel_size,
                ),
                mesh_dim_names=("dp", "sp"),
            )
        else:
            self.ulysses_device_mesh = None

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        if not hasattr(config, "global_batch_size"):  # ref model
            return

        #breakpoint()
        if self.config.rollout.n > 1:
            config.global_batch_size *= self.config.rollout.n
            self.print_rank0(f"{role} will use global batch size {config.global_batch_size}.")

        config.global_batch_size_per_device = (
            config.global_batch_size // self.device_mesh.size() * config.ulysses_sequence_parallel_size
        )
        if config.global_batch_size_per_device == 0:
            raise ValueError(f"{role} global batch size * ulysses size must be larger than num gpus.")

        if config.global_batch_size_per_device % config.micro_batch_size_per_device_for_update != 0:
            raise ValueError(f"{role} global batch size per device must be divisible by the micro batch size.")

        if (
            config.fsdp.enable_cpu_offload
            and config.global_batch_size_per_device != config.micro_batch_size_per_device_for_update
        ):
            raise ValueError(f"{role} cannot use FSDP's CPU offload when gradient accumulation is enabled.")

    def _build_model_optimizer(
        self,
        model_config: ModelConfig,
        fsdp_config: FSDPConfig,
        optim_config: Optional[OptimConfig],
        padding_free: bool = False,
    ) -> None:
        self.tokenizer = get_tokenizer(
            model_config.tokenizer_path,
            trust_remote_code=model_config.trust_remote_code,
            use_fast=True,
        )
        self.processor = get_processor(
            model_config.tokenizer_path,
            trust_remote_code=model_config.trust_remote_code,
            use_fast=True,
        )
        self.model_config = AutoConfig.from_pretrained(
            model_config.model_path,
            trust_remote_code=model_config.trust_remote_code,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_config.override_config,
        )

        try:
            self.generation_config = GenerationConfig.from_pretrained(model_config.model_path)
        except Exception:
            self.generation_config = GenerationConfig.from_model_config(self.model_config)

        self.print_rank0(f"Model config: {self.model_config}")

        if padding_free:
            apply_ulysses_patch(self.model_config.model_type)
            self.print_rank0("Ulysses patch applied!")

        if fsdp_config.torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor or self._is_critic else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(fsdp_config.torch_dtype)


        #print(AutoModelForVision2Seq._model_mapping.keys())
        #print("########################")
        #print(type(self.model_config))
        if self._is_critic:
            auto_class = AutoModelForTokenClassification
        elif type(self.model_config) in AutoModelForVision2Seq._model_mapping.keys():
            auto_class = AutoModelForVision2Seq
            print("Auto class is AutoModelForVision2Seq")
        else:
            auto_class = AutoModelForCausalLM
            print("Auto class is AutoModelForCausalLM")

        cfg = self.model_config
        if (not fsdp_config.enable_rank0_init) or self.device_mesh.get_local_rank("fsdp") == 0:
            cfg = _sanitize_mm_config(cfg, torch_dtype)
            model = auto_class.from_pretrained(
                model_config.model_path,
                config=cfg,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                device_map="cpu" if fsdp_config.enable_rank0_init else "cuda",
                low_cpu_mem_usage=True,
                trust_remote_code=model_config.trust_remote_code,
            )
        else:
            '''with no_init_weights(), init_empty_weights():
                model = auto_class.from_config(
                    self.model_config,
                    torch_dtype=torch_dtype,
                    attn_implementation="flash_attention_2",
                    trust_remote_code=model_config.trust_remote_code,
                )'''
            # Make sure these fields exist on the config (HF will read them later)
            setattr(cfg, "torch_dtype", torch_dtype)                  # dtype hint for later init/load
            setattr(cfg, "attn_implementation", "flash_attention_2")  # pick FA2 backend
            # 'trust_remote_code' is irrelevant for from_config

            cfg = _sanitize_mm_config(cfg, torch_dtype) # by AXZ

            for k in ("text_config","vision_config","decoder","encoder","generation_config"):
                v = getattr(cfg, k, None)
                #print(k, type(v))

            with no_init_weights(), init_empty_weights():
                model = auto_class.from_config(cfg)  # do NOT pass extra kwargs

        assert isinstance(model, PreTrainedModel)  # lint
        model.tie_weights()  # avoid hanging
        model = model.to(torch_dtype)
        if model_config.enable_gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if not (self._is_actor or self._is_critic):
            model.requires_grad_(False)

        if model_config.freeze_vision_tower:
            if hasattr(model, "visual"):
                model.visual.requires_grad_(False)
                fsdp_config.use_orig_params = True
                self.print_rank0("Vision tower is set to not trainable.")
            else:
                self.print_rank0("No vision tower found.")

        dist.barrier(device_ids=[torch.cuda.current_device()])
        print_model_size(model)
        print_gpu_memory_usage("After huggingface model init")
        mixed_precision = MixedPrecision(
            param_dtype=PrecisionType.to_dtype(fsdp_config.mp_param_dtype),
            reduce_dtype=PrecisionType.to_dtype(fsdp_config.mp_reduce_dtype),
            buffer_dtype=PrecisionType.to_dtype(fsdp_config.mp_buffer_dtype),
        )
        auto_wrap_policy = get_fsdp_wrap_policy(model)
        self.print_rank0(f"FSDP wrap policy: {auto_wrap_policy}.")

        if self.device_mesh.ndim == 2:
            if fsdp_config.enable_full_shard:
                sharding_strategy = ShardingStrategy.HYBRID_SHARD
            else:
                sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
        else:
            if fsdp_config.enable_full_shard:
                sharding_strategy = ShardingStrategy.FULL_SHARD
            else:
                sharding_strategy = ShardingStrategy.SHARD_GRAD_OP

        if fsdp_config.enable_cpu_offload:
            cpu_offload = CPUOffload(offload_params=True)
        else:
            cpu_offload = None

        if fsdp_config.enable_rank0_init:
            sync_module_states = True
            param_init_fn = get_init_fn(model, device="cuda") if self.rank != 0 else None
        else:
            sync_module_states = False
            param_init_fn = None

        self.fsdp_module = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            param_init_fn=param_init_fn,
            device_id=torch.cuda.current_device(),
            sync_module_states=sync_module_states,
            forward_prefetch=False,
            use_orig_params=fsdp_config.use_orig_params,
            device_mesh=self.device_mesh,
        )
        print_gpu_memory_usage("After FSDP module init")

        if self._is_actor or self._is_critic:
            if optim_config.strategy == "adamw":
                self.optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, self.fsdp_module.parameters()),
                    lr=optim_config.lr,
                    betas=optim_config.betas,
                    weight_decay=optim_config.weight_decay,
                    fused=True,
                )
            elif optim_config.strategy == "adamw_bf16":
                self.optimizer = AnyPrecisionAdamW(
                    filter(lambda p: p.requires_grad, self.fsdp_module.parameters()),
                    lr=optim_config.lr,
                    betas=optim_config.betas,
                    weight_decay=optim_config.weight_decay,
                )
            else:
                raise NotImplementedError(f"Optimizer {optim_config.strategy} not supported.")

            num_warmup_steps = int(optim_config.lr_warmup_ratio * optim_config.training_steps)
            self.lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps
            )
            print_gpu_memory_usage("After optimizer init")
        else:
            self.optimizer, self.lr_scheduler = None, None

    def _build_rollout(self) -> None:
        tp_size = self.config.rollout.tensor_parallel_size
        dp_size = self.world_size // tp_size
        assert self.world_size % tp_size == 0, (
            f"rollout world size: {self.world_size} is not divisible by tp size: {tp_size}"
        )
        rollout_device_mesh = init_device_mesh("cuda", mesh_shape=(dp_size, tp_size), mesh_dim_names=("dp", "tp"))
        if self.config.rollout.sampling_strategy in ["monet"]:
            self.hash_server = ray.get_actor(self.config.rollout.monet.hash_server_name)
        else:
            self.hash_server = None

        self.rule_based_judge_server = ray.get_actor(self.config.rule_based_judge.judge_server_name) if self.config.rule_based_judge.judge_server_name else None


        self.embed_tokenizer = None

        # -----------------------------------------------------------
        
        self.rollout = vLLMRollout(
            model_path=self.config.actor.model.model_path,
            config=self.config.rollout,
            tokenizer=self.tokenizer,
            processor=self.processor,
            hash_server=self.hash_server,
            rule_based_judge_server=self.rule_based_judge_server,
            embed_model = self.embed_model,
            embed_tokenizer = self.embed_tokenizer,
        )
        self.rollout_sharding_manager = FSDPVLLMShardingManager(
            module=self.fsdp_module,
            inference_engine=self.rollout.inference_engine,
            device_mesh=rollout_device_mesh,
        )

        print_gpu_memory_usage("After vllm init")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        #breakpoint()
        if self._is_critic:
            model_config = self.config.critic.model
            fsdp_config = self.config.critic.fsdp
            optim_config = self.config.critic.optim
            padding_free = self.config.critic.padding_free
            role = "critic"
        elif self._is_actor:
            model_config = self.config.actor.model
            fsdp_config = self.config.actor.fsdp
            optim_config = self.config.actor.optim
            padding_free = self.config.actor.padding_free
            role = "actor"
        elif self._is_ref:
            model_config = self.config.actor.model
            fsdp_config = self.config.ref.fsdp
            optim_config = None
            padding_free = self.config.ref.padding_free
            role = "ref"
        else:
            raise ValueError(f"Unknown role {role}.")

        if self._is_actor or self._is_critic or self._is_ref:
            self._build_model_optimizer(
                model_config=model_config,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                padding_free=padding_free,
            )
            if self._use_param_offload:
                offload_fsdp_model(self.fsdp_module)
                print_gpu_memory_usage(f"After offload {role} model during init")

            if self._use_optimizer_offload:
                offload_fsdp_optimizer(optimizer=self.optimizer)
                print_gpu_memory_usage(f"After offload {role} optimizer during init")

        if self._is_actor:
            from .actor.dp_actor import DataParallelPPOActor  # lazy import

            self.actor = DataParallelPPOActor(
                config=self.config.actor,
                actor_module=self.fsdp_module,
                actor_optimizer=self.optimizer,
            )

        if self._is_critic:
            from .critic.dp_critic import DataParallelPPOCritic  # lazy import

            self.critic = DataParallelPPOCritic(
                config=self.config,
                critic_module=self.fsdp_module,
                critic_optimizer=self.optimizer,
            )

        if self._is_rollout:
            self._build_rollout()

        if self._is_ref:
            from .actor.dp_actor import DataParallelPPOActor  # lazy import

            self.ref_policy = DataParallelPPOActor(
                config=self.config.ref,
                actor_module=self.fsdp_module,
            )

        if self._is_actor or self._is_critic:
            self.flops_counter = FlopsCounter(self.model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.fsdp_module,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
            )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, path: str):
        assert self._is_actor or self._is_critic
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        self.checkpoint_manager.save_checkpoint(path)
        dist.barrier()
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path: str):
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        self.checkpoint_manager.load_checkpoint(path)
        dist.barrier()
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:  # avoid OOM in resuming
            offload_fsdp_optimizer(self.optimizer)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        assert self._is_actor
        data = data.to(torch.cuda.current_device())

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            load_fsdp_optimizer(optimizer=self.optimizer)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)

            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu_actor"] = (
                estimated_flops * self.config.actor.ppo_epochs / (promised_flops * self.world_size)
            )
            metrics["perf/max_memory_allocated_gb"] = (
                torch.cuda.max_memory_allocated() - self.rollout_sharding_manager.freed_bytes
            ) / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = (
                torch.cuda.max_memory_reserved() - self.rollout_sharding_manager.freed_bytes
            ) / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            self.lr_scheduler.step()
            lr = self.lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr

            # Metrics should be in non_tensor_batch instead of meta_info, as DataProto not concat meta_info.
            output = DataProto(
                non_tensor_batch={
                    key: np.array([value] if np.isscalar(value) else value) for key, value in metrics.items()
                }
            )

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            offload_fsdp_optimizer(optimizer=self.optimizer)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        assert self._is_rollout
        #breakpoint()
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        with self.rollout_sharding_manager:
            # after parameters sync with rollout, offload actor model to CPU
            if self._use_param_offload:
                offload_fsdp_model(self.fsdp_module)

            if self._use_optimizer_offload:
                offload_fsdp_optimizer(optimizer=self.optimizer)

            #breakpoint()
            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            #breakpoint()
            mode = prompts.meta_info["mode"]
            if mode == "test":
                output = self.rollout.generate_sequences(prompts=prompts)
            elif mode == "train_pre_gen":
                if self.config.rollout.sampling_strategy in ["monet"]:
                    output = self.rollout.generate_sequences(prompts=prompts)
                else:
                    raise NotImplementedError(f"Sampling strategy {self.config.rollout.sampling_strategy} not supported for {mode} mode.")
            elif mode == "train_pre_gen_online":
                if self.config.rollout.sampling_strategy in ["monet"]:
                    output = self.rollout.generate_sequences_monet(prompts=prompts)
                else:
                    raise NotImplementedError(f"Sampling strategy {self.config.rollout.sampling_strategy} not supported for {mode} mode.")
            elif mode == "train_rl_gen":
                if self.config.rollout.sampling_strategy == "greedy":
                    output = self.rollout.generate_sequences(prompts=prompts)
                elif self.config.rollout.sampling_strategy in ["monet"]:
                    output = self.rollout.generate_sequences_monet(prompts=prompts)
                else:
                    raise NotImplementedError(f"Sampling strategy {self.config.rollout.sampling_strategy} not supported for {mode} mode.")
            else:
                raise NotImplementedError(f"Mode {mode} not supported.")
            output = self.rollout_sharding_manager.postprocess_data(output)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_probs(self, data: DataProto):
        assert self._is_actor
        data = data.to(torch.cuda.current_device())
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["temperature"] = self.config.rollout.temperature
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.actor.compute_log_prob(data=data)
            output = DataProto.from_dict(
                tensors={"old_log_probs": output}, meta_info={"temperature": self.config.rollout.temperature}
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.fsdp_module._handle.reshard(True)

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_probs(self, data: DataProto):
        assert self._is_ref
        data = data.to(torch.cuda.current_device())
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        data.meta_info["temperature"] = self.config.rollout.temperature
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.ref_policy.compute_log_prob(data=data)
            output = DataProto.from_dict(tensors={"ref_log_probs": output})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.fsdp_module._handle.reshard(True)

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        assert self._is_critic
        data = data.to(torch.cuda.current_device())
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={"values": values})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        data = data.to(torch.cuda.current_device())
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            load_fsdp_optimizer(optimizer=self.optimizer)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name="update_critic", logger=None) as timer:
                metrics = self.critic.update_critic(data=data)

            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu_critic"] = (
                estimated_flops * self.config.actor.ppo_epochs / (promised_flops * self.world_size)
            )

            self.lr_scheduler.step()
            lr = self.lr_scheduler.get_last_lr()[0]
            metrics["critic/lr"] = lr

            # Metrics should be in non_tensor_batch instead of meta_info, as DataProto not concat meta_info.
            output = DataProto(
                non_tensor_batch={
                    metric: np.array([value] if np.isscalar(value) else value) for metric, value in metrics.items()
                }
            )

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            offload_fsdp_optimizer(optimizer=self.optimizer)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_embed_service(self):
        return self.embed_service
    

    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_embeds(self, data: DataProto):
        assert self._is_rollout
        batch_steps: np.array[List[str]] = data.non_tensor_batch["steps"]
        batch_embeds = []
        for steps in batch_steps:
            batch_embeds.append( self.compute_embeds_fn(steps))
        return DataProto(non_tensor_batch={"embeds": np.array(batch_embeds, dtype=object)})

    def compute_embeds_fn(self, texts):
        outputs = self.embed_model.embed(texts, use_tqdm=False)
        #outputs = ray.get(self.embed_model.encode.remote(texts, use_tqdm=False)) # old, worked
        return torch.tensor([o.outputs.embedding for o in outputs]).detach().half().cpu().numpy().copy()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rule_based_judge(self, data: DataProto):
        assert self._is_rollout
        correctness = []
        response_strs = []
        response_ids = data.batch["responses"]
        response_length = data.batch["response_mask"].sum(dim=-1)
        if self.config.rollout.offline_difficulty_sampling:
            position = 2
        elif self.config.rollout.online_difficulty_sampling:
            position = 3

        # parrallel rule -> api judge
        if self.config.rule_based_judge.judge_function_name=="rule_then_api_batch_judge":
            for i in range(len(data)):
                valid_response_ids = response_ids[i][: response_length[i]]
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                response_strs.append(response_str)
            correctness = rule_then_api_batch_judge(
                questions=data.non_tensor_batch["problem"],
                preds=response_strs,
                gts=data.non_tensor_batch["ground_truth"],
                api_name=self.config.rule_based_judge.api_name,
                api_kwargs=self.config.rule_based_judge.api_kwargs,
                client=self.api_client,
                repetition_penalty=self.config.reward.repetition_penalty,
            )
        else:
            for i in tqdm(range(len(data)), desc="Rule-based judge", position=position, disable=self.rank != 0):
                valid_response_ids = response_ids[i][: response_length[i]]
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                response_strs.append(response_str)
                ground_truth = data.non_tensor_batch["ground_truth"][i]
                question = data.non_tensor_batch["problem"][i]
                if self.config.rule_based_judge.judge_function == "./examples/reward_function/monet_reward_function.py:extract_and_check":
                    correctness.append(easyr1_monet_extract_and_check(response_str, ground_truth))
                elif self.config.rule_based_judge.judge_function == "./examples/reward_function/monet_reward_function.py:extract_and_check_api":
                    correctness.append(easyr1_monet_extract_and_check_api(question, response_str, ground_truth, self.api_client))
                else:
                    raise NotImplementedError(f"Rule-based judge function {self.config.rule_based_judge.judge_function} not supported.")
            
        return DataProto(non_tensor_batch={"correctness": correctness, "response_strs": response_strs})
    
