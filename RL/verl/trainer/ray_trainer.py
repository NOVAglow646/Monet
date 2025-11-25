import os
import pdb
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any, Dict, List, Optional, Type, Set

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, remove_obsolete_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import FunctionRewardManager, FunctionRuleBasedJudgeManager
from . import core_algos
from .config import PPOConfig
from .metrics import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics
from tools.api_judge import api_batch_judge
from tools.custom_api import build_deepseek_client, build_gemini_client

import random
from torch.utils.data import Dataset    
from ..utils.dataset import RLHFDataset, collate_fn
from torch.utils.data import RandomSampler, SequentialSampler
#from tools.compute_embeds import compute_embeds_fn
from tools.actors import StepHashServer, SampleHashServer
from tools.actors import EmbedServer
import matplotlib.pyplot as plt
import re

def replace_abs_vis_token_content(s: str) -> str:
    pattern = re.compile(r'(<abs_vis_token>)(.*?)(</abs_vis_token>)', flags=re.DOTALL)
    return pattern.sub(r'\1<latent>\3', s)

class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REMAX = "remax"
    RLOO = "rloo"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.KLController, kl_penalty="kl"):
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = core_algos.compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = VF.masked_mean(kld, mask=response_mask, dim=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()
    metrics = {"critic/kl": current_kl, "critic/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(config: PPOConfig, data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0, sampling_strategy: str = "greedy") -> DataProto:
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    index = data.non_tensor_batch["uid"]
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch["values"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards, values, response_mask, gamma, lam
        )
    elif adv_estimator == AdvantageEstimator.GRPO:
        if sampling_strategy in ["greedy"]:
            advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards, response_mask, index)
        elif sampling_strategy in ["monet"]:
            advantages, returns = core_algos.compute_grpo_latent_advantage(token_level_rewards, response_mask, index)

    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards, response_mask, gamma
        )
    elif adv_estimator == AdvantageEstimator.REMAX:
        reward_baselines = data.batch["reward_baselines"]
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards, reward_baselines, response_mask
        )
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards, response_mask, index)
    else:
        raise NotImplementedError

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[FunctionRewardManager] = None,
        val_reward_fn: Optional[FunctionRewardManager] = None,
        rule_based_judge: Optional[FunctionRuleBasedJudgeManager] = None,
        #embed_model: Optional[torch.nn.Module] = None,
        #embed_tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.rule_based_judge = rule_based_judge
        self.hybrid_engine = config.worker.hybrid_engine
        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, (
                f"ActorRollout should be included in {role_worker_mapping.keys()}."
            )
        else:
            raise NotImplementedError

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if Role.RefPolicy in role_worker_mapping and not config.algorithm.disable_kl:
            self.use_reference_policy = True
            self.kl_ctrl = core_algos.get_kl_controller(config.algorithm)
        else:
            self.use_reference_policy = False
            self.kl_ctrl = core_algos.FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")


        if self.config.data.pr_batch_size != -1:
            if config.data.pr_batch_size % config.worker.actor.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by actor global batch size.")

            if (
                config.data.pr_batch_size * config.worker.rollout.n
            ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
                )
            
        else:
            if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by actor global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
                )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")
        
        self.base_dataset = self.train_dataloader.dataset
        self.correct_pool = defaultdict(list)
        self.correct_gen_out_pool = defaultdict(list)
        self.selected_sample_statistics = defaultdict(int)

        if self.config.worker.rollout.sampling_strategy in ["monet"]:
            self.sample_hash_server_main = SampleHashServer.options(
                name="sample_hash_server_main"
            ).remote()
            #print("type of self.sample_hash_server_main:", type(self.sample_hash_server_main))
            self.config.worker.rollout.monet.hash_server_name = "sample_hash_server_main"
            ray.get(self.sample_hash_server_main.ping.remote())

        if "api" in self.config.worker.rule_based_judge.judge_function_name:
            if self.config.worker.rule_based_judge.api_name in ['deepseek-chat', 'deepseek']:
                self.client = build_deepseek_client()
            elif self.config.worker.rule_based_judge.api_name == 'gemini-2.5-pro':
                self.client = build_gemini_client()
            else:
                self.client = None
                raise ValueError(f"API {self.config.worker.rule_based_judge.api_name} not supported.")

    def _maybe_log_val_generations(
        self, inputs: List[str], outputs: List[str], labels: List[str], scores: List[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> Dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        for batch_dict in self.val_dataloader:
            
            test_batch = DataProto.from_single_dict(batch_dict)
            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            if "multi_modal_data" in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "global_index"],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                )

            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_gen_batch.meta_info["mode"] = "test"
            test_output_gen_batch = self.actor_rollout_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size)

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            # (output_ids[3]==151666).nonzero()
            # (output_ids[3]==151667).nonzero()
            # output_texts[5]
            #breakpoint()
            
            output_texts = [replace_abs_vis_token_content(self.tokenizer.decode(ids, skip_special_tokens=False)).replace("<|endoftext|>", "").replace("<|im_end|>", "") for ids in output_ids]
            # self.tokenizer.decode(output_ids[5], skip_special_tokens=False).replace("<|endoftext|>", "").replace("<|im_end|>", "")
            #breakpoint() replace_abs_vis_token_content(self.tokenizer.decode(output_ids[1], skip_special_tokens=False)).replace("<|endoftext|>", "").replace("<|im_end|>", "")
            #pdb.set_trace()
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            if 'api' in self.config.worker.rule_based_judge.judge_function_name:
                #breakpoint()
                correctness_list = api_batch_judge(
                    questions=test_batch.non_tensor_batch["problem"].tolist(),
                    preds=output_texts,
                    gts=test_batch.non_tensor_batch["ground_truth"].tolist(),
                    api_name=self.config.worker.rule_based_judge.api_name,
                    api_kwargs=self.config.worker.rule_based_judge.api_kwargs,
                    client=self.client,
                    repetition_penalty=self.config.worker.reward.repetition_penalty,
                )
                #correctness_list = ray.get(self.rule_based_judge.judge.remote(output_texts, test_batch.non_tensor_batch["ground_truth"].tolist()))
                test_batch.non_tensor_batch["correctness"] = correctness_list
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}

        return {"val/reward_score": reward_score, **val_reward_metrics}

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout], config=self.config.worker, role="actor_rollout"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.worker, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        print('start building worker group')
        all_wg: Dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)
        print('done building worker group')
        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()
        
    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path, self.global_step, self.config.trainer.save_limit
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_wg.save_checkpoint(actor_path)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        last_global_step_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(last_global_step_path, "w") as f:
            f.write(str(self.global_step))

        if self.config.worker.rollout.sampling_strategy in ["monet"]:
            self.sample_hash_server_main.save_info.remote(filepath=folder_path, overwrite=True)
        

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is None:
            return

        if "global_step_" not in self.config.trainer.load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {self.config.trainer.load_checkpoint_path}.")
        self.global_step = int(self.config.trainer.load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(self.config.trainer.load_checkpoint_path, "actor")
        self.actor_rollout_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(self.config.trainer.load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(self.config.trainer.load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

        self.step_hash_server_main.load_info.remote(self.config.trainer.load_checkpoint_path)

    def _balance_batch(self, batch: DataProto, metrics: Dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        val_metrics: Optional[Dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        
        for epoch in tqdm(range(self.config.trainer.total_epochs), desc="Epoch", position=0):
            
            if self.config.worker.rollout.offline_difficulty_sampling:
                if self.config.worker.rollout.sampling_strategy in ["monet"]:
                    self.pre_generate_monet_offline(epoch=epoch)
                
                epoch_new_selected_statistics = {}
                for global_id in self.correct_pool.keys():
                    if self.selected_sample_statistics[global_id] == 0:
                        epoch_new_selected_statistics[global_id] = 1
                    self.selected_sample_statistics[global_id] += 1

            if self.config.worker.rollout.online_difficulty_sampling:
                if self.config.worker.rollout.sampling_strategy in ["monet"]:
                    self.correct_pool.clear() # epoch start
                    if self.config.data.shuffle:
                        train_dataloader_generator = torch.Generator()
                        train_dataloader_generator.manual_seed(self.config.data.seed)
                        sampler = RandomSampler(data_source=self.base_dataset, generator=train_dataloader_generator)
                    else:
                        sampler = SequentialSampler(data_source=self.base_dataset)

                    out_data_loader = StatefulDataLoader(
                        dataset=self.base_dataset,
                        batch_size=self.config.data.online_accum_size,
                        sampler=sampler,
                        num_workers=8,
                        collate_fn=collate_fn,
                        pin_memory=False,
                        drop_last=False,
                    )
                    
                    for large_batch_dict in tqdm(out_data_loader, desc="Trasersing all training data", position=1):
                        self.pre_generate_monet_online(DataProto.from_single_dict(large_batch_dict))
                        if len(self.correct_pool) < self.config.data.rollout_batch_size:
                            print("Not enough samples to form a batch. Continue to the next large batch.")
                            continue
                        for batch in tqdm(self.train_dataloader, desc="Running step", position=5 if self.config.worker.rollout.sampling_strategy in ["monet"] else 1):
                            self.global_step += 1
                            if self.global_step > self.training_steps:
                                break

                            metrics, timing_raw = {}, {}
                            #pdb.set_trace()
                            with timer("step", timing_raw):
                                self._balance_batch(batch, metrics=metrics)
                                self.post_generate_update(metrics, timing_raw, batch)
            
            else:  # default / offline
                for batch_dict in tqdm(self.train_dataloader, desc="Running step", position=4 if self.config.worker.rollout.sampling_strategy in ["monet"] else 1):
                    self.global_step += 1
                    if self.global_step > self.training_steps:
                        break

                    metrics, timing_raw = {}, {}
                    #reakpoint()
                    batch: DataProto = DataProto.from_single_dict(batch_dict)

                    # pop those keys for generation
                    #breakpoint()
                    batch.meta_info["mode"] = "train_rl_gen"
                    gen_batch = self.build_gen_batch(batch)
                    sample_idx = gen_batch.non_tensor_batch["global_index"]
                    sample_idx = [item for item in sample_idx for _ in range(self.config.worker.rollout.n)]
                    with timer("step", timing_raw):
                        # generate a batch
                        #breakpoint()
                        with timer("gen", timing_raw):  # wg: worker group
                            gen_batch.meta_info["mode"] = "train_rl_gen"
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                        batch = self.post_generate_process(metrics, timing_raw, gen_batch, gen_batch_output)

                        self.post_generate_update(metrics, timing_raw, batch)

                # collect metrics
                num_gpus = self.resource_pool_manager.get_num_gpus()
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

                self.logger.log(data=metrics, step=self.global_step)

            

        # perform validation after training
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics: {convert_dict_to_str(val_metrics)}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()


    def pre_generate_monet_offline(self, epoch: int):
        """
        Generate rollouts for the whole training set, and collect the correct answers for each sample,
        so that later RL rollouts can use these correct reponses as intermediate steps to get step-wise MC rewards.
        """
        self.correct_pool.clear() # epoch start
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.seed)
            sampler = RandomSampler(data_source=self.base_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.base_dataset)

        data_loader = StatefulDataLoader(
            dataset=self.base_dataset,
            batch_size=self.config.data.pr_batch_size,
            sampler=sampler,
            num_workers=8,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=False,
        )
        
        for batch_dict in tqdm(data_loader, desc="Pre-generation", position=1):
            metrics, timing_raw = {}, {}
            batch = DataProto.from_single_dict(batch_dict)
            batch.meta_info["mode"] = "train_pre_gen"  
            gen_batch = self.build_gen_batch(batch)
            sample_idx = gen_batch.non_tensor_batch["global_index"]
            sample_idx = [item for item in sample_idx for _ in range(self.config.worker.rollout.n)]
            with timer("pre_step", timing_raw):
                with timer("pre_gen", timing_raw):
                    gen_batch.meta_info["mode"] = "train_pre_gen"     
                    gen_out = self.actor_rollout_wg.generate_sequences(gen_batch)
                #breakpoint()
                batch = batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
                batch = batch.union(gen_out)
                #print("1 len(batch)=",len(batch)) # pr_bsz * rollout.n
                batch.non_tensor_batch.pop("multi_modal_data", None)
                self._balance_batch(batch, metrics=metrics)
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                
                # after all samples in the batch are judged, use ray.get to get the results
                judge_results_and_answer_texts_proto = self.actor_rollout_wg.compute_rule_based_judge(batch, position=2)
                judge_results, answer_texts = judge_results_and_answer_texts_proto.non_tensor_batch["correctness"], judge_results_and_answer_texts_proto.non_tensor_batch["response_strs"]

                
                print("Pre-rollout acc of a batch:", f"acc={judge_results.sum().item() / len(judge_results)}") # pr_bsz * rollout.n
                # update the step hash server with the new samples
                upd_refs = []
                upd_min_correct_len_refs = []
                for i, (sample_id, correct, answer_text) in enumerate(zip(sample_idx, judge_results, answer_texts)):
                    resp_tks_len = batch.batch['responses'][i][batch.batch['response_mask'][i].bool()].shape[0]
                    if correct:
                        upd_min_correct_len_ref = self.sample_hash_server_main.update_min_mean_correct_resp_len.remote(sample_id, resp_tks_len)
                        upd_min_correct_len_refs.append(upd_min_correct_len_ref)
                ray.get(upd_min_correct_len_refs) # update the min correct response length for each question    

                # traverse pr_bsz*rollout.n samples
                batch_rollout_record = defaultdict(list)
                #for sample_id, correct, answer_text, steps_embeds, steps in tqdm(zip(sample_idx, judge_results, answer_texts, batch_steps_embeds, batch_steps), desc="Update step hash", position=3, total=len(sample_idx)):
                batch_update_info = defaultdict(list)
                for i, (sample_id, correct, answer_text) in enumerate(zip(sample_idx, judge_results, answer_texts)):

                    batch_update_info[sample_id].append((int(sample_id), correct))
                    
                    batch_rollout_record[sample_id].append(1)
                    self.correct_pool[sample_id].append(answer_text)

                #breakpoint()
                # discard samples that are too easy
                static_correct_pool_keys = list(self.correct_pool.keys())
                for sample_id in static_correct_pool_keys:
                    if len(batch_rollout_record[sample_id]) == 0: # this sample_id is from previous batches and not recorded in the batch_rollout_record of the current batch
                        continue
                    if batch_rollout_record[sample_id].count(1)/len(batch_rollout_record[sample_id]) > self.config.worker.rollout.monet.select_acc_threshold:
                        #self.correct_pool.pop(sample_id)
                        pass
                
        print(f"len(self.correct_pool)={len(self.correct_pool)}")
        self._rebuild_train_dataloader_with_correct_pool(self.base_dataset)
        
    def pre_generate_monet_online(self, large_batch):
        """
        Traverse the large batch, generate rollouts for each small batch, and collect the samples with correct answers and avg accuracy below threshold
        """
        self.correct_pool.clear()
        ori_bsz = self.config.data.rollout_batch_size
        for b in tqdm(range(0, len(large_batch), ori_bsz), desc="Online pre-generation", position=2):
            metrics, timing_raw = {}, {}
            batch = large_batch[b:b+ori_bsz]
            batch.meta_info["mode"] = "train_pre_gen"  
            gen_batch = self.build_gen_batch(batch)
            sample_idx = gen_batch.non_tensor_batch["global_index"]
            sample_idx = [item for item in sample_idx for _ in range(self.config.worker.rollout.n)]
            with timer("pre_step", timing_raw):
                with timer("pre_gen", timing_raw):
                    gen_batch.meta_info["mode"] = "train_pre_gen_online"     
                    gen_out = self.actor_rollout_wg.generate_sequences(gen_batch)
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )   

                batch = batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
                batch = batch.union(gen_out)
                batch.non_tensor_batch.pop("multi_modal_data", None)
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # after all samples in the batch are judged, use ray.get to get the results
                judge_results_and_answer_texts_proto = self.actor_rollout_wg.compute_rule_based_judge(data=batch)
                judge_results, answer_texts = judge_results_and_answer_texts_proto.non_tensor_batch["correctness"], judge_results_and_answer_texts_proto.non_tensor_batch["response_strs"]
                batch.non_tensor_batch["correctness"] = judge_results

                print("Pre-rollout acc of a batch:", f"acc={judge_results.sum().item() / len(judge_results)}") # ori_bsz * rollout.n

                upd_min_correct_len_refs = []
                for i, (sample_id, correct, answer_text) in enumerate(zip(sample_idx, judge_results, answer_texts)):
                    resp_tks_len = batch.batch['responses'][i][batch.batch['response_mask'][i].bool()].shape[0]
                    if correct:
                        upd_min_correct_len_ref = self.sample_hash_server_main.update_min_mean_correct_resp_len.remote(sample_id, resp_tks_len)
                        upd_min_correct_len_refs.append(upd_min_correct_len_ref)
                ray.get(upd_min_correct_len_refs) # update the min correct response length for each question    

                batch_rollout_record = defaultdict(list)
                batch_update_info = defaultdict(list)
                for i, (sample_id, correct, answer_text) in enumerate(zip(sample_idx, judge_results, answer_texts)):

                    batch_update_info[sample_id].append((int(sample_id), correct))
                    self.correct_gen_out_pool[sample_id].append(batch[i:i+1])
                    if correct:
                        batch_rollout_record[sample_id].append(1)
                        self.correct_pool[sample_id].append(answer_text)
                        # Store a slice to preserve a batch dimension (DataProto of size 1)
                        # This avoids TensorDict cat error (batch_size=[] when concatenating DataProtoItem)
                    else:
                        batch_rollout_record[sample_id].append(0)

                # discard samples that are too easy
                static_correct_pool_keys = list(self.correct_pool.keys())
                for sample_id in static_correct_pool_keys:
                    if len(batch_rollout_record[sample_id]) == 0: # this sample_id is from previous batches and not recorded in the batch_rollout_record of the current batch
                        continue
                    sample_pre_rollout_acc = batch_rollout_record[sample_id].count(1)/len(batch_rollout_record[sample_id])
                    if sample_pre_rollout_acc > self.config.worker.rollout.monet.select_acc_threshold:
                        self.correct_pool.pop(sample_id)
                
        print(f"len(self.correct_pool)={len(self.correct_pool)}")
        self.build_train_dataloader_with_correct_gen_out_pool(self.base_dataset)
        
    def post_generate_process(self, metrics, timing_raw, gen_batch, gen_batch_output):

        if self.config.algorithm.adv_estimator == "remax":
            with timer("gen_max", timing_raw):
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                batch = batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

        batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
            )   
        batch = batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)

        batch = batch.union(gen_batch_output)
        batch.non_tensor_batch.pop("multi_modal_data", None)

        # balance the number of valid tokens on each dp rank.
        # Note that this breaks the order of data inside the batch.
        # Please take care when you implement group based adv computation such as GRPO and rloo
        self._balance_batch(batch, metrics=metrics)

        # compute global_valid tokens
        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()    

        return batch

    def post_generate_update(self, metrics, timing_raw, batch):
        with timer("reward", timing_raw):
            # batch.non_tensor_batch should have "correctness" here
            reward_ref = self.reward_fn.compute_reward.remote(batch)

        # recompute old_log_probs
        #breakpoint()
        with timer("old", timing_raw):
            old_log_probs = self.actor_rollout_wg.compute_log_probs(batch)
            batch = batch.union(old_log_probs)

        # compute ref_log_probs
        if self.use_reference_policy:
            with timer("ref", timing_raw):
                ref_log_probs = self.ref_policy_wg.compute_ref_log_probs(batch)
                batch = batch.union(ref_log_probs)

        # compute values
        if self.use_critic:
            with timer("values", timing_raw):
                values = self.critic_wg.compute_values(batch)
                batch = batch.union(values)
        #breakpoint()
        with timer("adv", timing_raw):
            # get token level scores
            reward_tensor, reward_metrics = ray.get(reward_ref)
            batch.batch["token_level_scores"] = reward_tensor
            reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
            metrics.update(reward_metrics)

            # apply kl penalty if available
            if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                # apply kl penalty to reward
                batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                metrics.update(kl_metrics)
            else:
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

            # compute advantages, executed on the driver process
            batch = compute_advantage(
                self.config,
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                sampling_strategy=self.config.worker.rollout.sampling_strategy
            )

        # update critic
        if self.use_critic:
            with timer("update_critic", timing_raw):
                critic_output = self.critic_wg.update_critic(batch)

            critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
            metrics.update(critic_metrics)

        # update actor
        #breakpoint()
        if self.config.trainer.critic_warmup <= self.global_step:
            with timer("update_actor", timing_raw):
                actor_output = self.actor_rollout_wg.update_actor(batch)

            actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
            metrics.update(actor_metrics)

        # validate
        if (
            self.val_reward_fn is not None
            and self.config.trainer.val_freq > 0
            and self.global_step % self.config.trainer.val_freq == 0
        ):
            with timer("validation", timing_raw):
                val_metrics = self._validate()

            metrics.update(val_metrics)

        if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
            with timer("save_checkpoint", timing_raw):
                self._save_checkpoint()

    def _rebuild_train_dataloader_with_correct_pool(self, base_dataset):
        '''
        Rebuild the train_dataloader with samples that have correct answers and acc<select_acc_threshold.
        '''
        class CorrectAnswerDataset(Dataset):
            def __init__(self, base_dataset, correct_pool):
                self.base_dataset = base_dataset
                self.correct_pool = correct_pool
                self.qids = list(correct_pool.keys()) # ids of samples that have correct answers and acc<select_acc_threshold
            def __len__(self):
                return len(self.qids)

            def __getitem__(self, idx):
                qid = self.qids[idx]
                sample = self.base_dataset[qid]
                sample["mc_raw_prompt_ids"] = sample["raw_prompt_ids"]
                sample["correct_solutions_text"] = random.choice(self.correct_pool[qid])
                return sample

        correct_ds = CorrectAnswerDataset(base_dataset, self.correct_pool)

        self.train_dataloader = StatefulDataLoader(
            dataset=correct_ds,
            batch_size=self.config.data.rollout_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=8,
            pin_memory=False,
            drop_last=True,
        )
        
    def build_train_dataloader_with_correct_gen_out_pool(self, base_dataset):
        '''
        Rebuild the train_dataloader with samples that have correct answers and acc<select_acc_threshold.
        '''
        class CorrectGenOutDataset(Dataset):
            def __init__(self, base_dataset, correct_pool, correct_gen_out_pool):
                self.base_dataset = base_dataset
                self.correct_pool = correct_pool
                self.correct_gen_out_pool = correct_gen_out_pool
                self.qids = list(correct_pool.keys()) # ids of samples that have correct answers and acc<select_acc_threshold
            def __len__(self):
                return len(self.qids)

            def __getitem__(self, idx):
                qid = self.qids[idx]
                group_list = self.correct_gen_out_pool[qid]
                sample = DataProto.concat(group_list) # concat the group
                return sample

        correct_ds = CorrectGenOutDataset(base_dataset, self.correct_pool, self.correct_gen_out_pool)

        def collate_fn_gen_out(features: List[DataProto]):
            return DataProto.concat(features)

        self.train_dataloader = StatefulDataLoader(
            dataset=correct_ds,
            batch_size=self.config.data.rollout_batch_size,
            shuffle=True,
            collate_fn=collate_fn_gen_out,
            num_workers=8,
            pin_memory=False,
            drop_last=True,
        )

    def build_gen_batch(self, batch: DataProto) -> None:
        if "multi_modal_data" in batch.non_tensor_batch.keys():
            if self.config.worker.rollout.sampling_strategy == "greedy":
                gen_batch = batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"]
                )
                if batch.meta_info["mode"] == "train_pre_gen":
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "global_index"]
                    )
                elif batch.meta_info["mode"] == "train_rl_gen":
                    gen_batch = batch.pop(
                            batch_keys=["input_ids", "attention_mask", "position_ids"],
                            non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "problem", "ground_truth", "prompt_before_processor", "global_index", "correct_solutions_text"]
                        )
            elif self.config.worker.rollout.sampling_strategy == "monet":
                gen_batch = batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "global_index", "problem", "ground_truth"]
                )
        else:
            gen_batch = batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids"],
            )
        return gen_batch

    @staticmethod
    def split_solution_into_steps(solution: str, delim: str = "### Step") -> List[List[str]]:
        steps = solution.split(delim)
        steps = [re.sub(r"^ \d+(\.\d+)?: ", "", step).strip() for step in steps]
        return steps
