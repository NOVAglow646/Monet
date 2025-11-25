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
Implement Actor
"""

import math
import os
from collections import defaultdict
from typing import Any, Dict, Optional
import pdb
import torch
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers.modeling_flash_attention_utils import index_first_axis, pad_input, unpad_input
#from verl.workers.actor.fa_shim import index_first_axis, pad_input, unpad_input # implementation by AXZ


from ...protocol import DataProto
from ...trainer import core_algos
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .base import BasePPOActor
from .config import ActorConfig


__all__ = ["DataParallelPPOActor"]

def collect_varlen_segment_indices(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    start_id: int,
    end_id: int,
) -> torch.Tensor:
    """
    Collect indices (on the varlen/unpadded sequence) for token positions strictly inside
    matched (start, end) segments, skipping the first matched segment per sequence.

    Args:
        input_ids: LongTensor of shape (B, S).
        attention_mask: Bool/Long/Byte Tensor of shape (B, S); 1=valid, 0=pad/ignored.
        start_id: int, the start marker token id.
        end_id: int, the end marker token id.

    Returns:
        LongTensor of shape (K,), where each element is the index on the unpadded varlen
        sequence (i.e., the [1, L] after unpad+transpose) corresponding to a kept position.
    """
    assert input_ids.dim() == 2, "input_ids must be 2D (B, S)"
    assert attention_mask.shape == input_ids.shape, "attention_mask must match input_ids"

    device = input_ids.device
    B, S = input_ids.shape

    # Ensure mask is 0/1 long tensor
    mask = attention_mask.to(dtype=torch.long)

    # Flatten mask to compute varlen positions: prefix sum gives mapping to [0..T-1]
    # For any flat position p with mask[p]==1, its varlen index is prefix[p]-1.
    mask_flat = mask.reshape(-1)                                      # (B*S,)
    prefix = torch.cumsum(mask_flat, dim=0)                           # (B*S,)
    # We will only index prefix at places where mask==1.

    varlen_indices_per_batch = []
    varlen_indices_by_batch = []
    for b in range(B):
        varlen_indices = []
        row_ids = input_ids[b]                                        # (S,)
        row_mask = mask[b]                                            # (S,)

        # Find all start/end positions (on [0..S-1])
        starts = (row_ids == start_id).nonzero(as_tuple=False).squeeze(-1)  # (Ns,) or empty
        ends   = (row_ids == end_id).nonzero(as_tuple=False).squeeze(-1)    # (Ne,) or empty

        if starts.numel() == 0 or ends.numel() == 0:
            varlen_indices_by_batch.append(varlen_indices)
            continue

        # Two-pointer greedy matching: for each start, find the nearest end to its right.
        i_ptr, j_ptr = 0, 0
        matched = []  # list of (s, e), with e > s
        while i_ptr < starts.numel() and j_ptr < ends.numel():
            s_pos = starts[i_ptr].item()
            # Move j_ptr until we find an end strictly to the right of s_pos
            while j_ptr < ends.numel() and ends[j_ptr].item() <= s_pos:
                j_ptr += 1
            if j_ptr >= ends.numel():
                break
            e_pos = ends[j_ptr].item()
            matched.append((s_pos, e_pos))
            i_ptr += 1
            j_ptr += 1

        if len(matched) <= 0:
            # Nothing (or only the first segment which we must skip)
            varlen_indices_by_batch.append(varlen_indices)
            continue

        for (s_pos, e_pos) in matched[:]:
            inner = torch.arange(s_pos, e_pos, device=device, dtype=torch.long)  # (Lseg,)
            
            # Filter by attention mask (positions not in varlen stream should be dropped)
            inner_valid = inner[row_mask[inner] == 1]
            if inner_valid.numel() == 0:
                continue

            # Map (b, pos) -> flat index -> varlen index
            flat_pos = b * S + inner_valid                               # (Lkeep,)
            # mask_flat[flat_pos] must be 1 here; varlen idx = prefix - 1
            var_idx = prefix[flat_pos] - 1                                # still on device, Long
            varlen_indices_per_batch.append(var_idx)
            varlen_indices.append(var_idx)
        varlen_indices_by_batch.append(varlen_indices)
    if len(varlen_indices_per_batch) == 0:
        return torch.empty(0, dtype=torch.long, device=device), varlen_indices_by_batch

    # Concatenate all batches; these indices correspond to positions on the
    # unpadded [1, total_nnz] sequence (i.e., after unpad + transpose).
    return varlen_indices_per_batch, varlen_indices_by_batch

def compute_latent_log_probs(latent_poss, latents, last_hidden_state, sigma=1.0):
    """
    Compute log-prob under a headless isotropic Gaussian:
        z ~ N(mu=last_hidden_state[..., latent_poss, :], sigma^2 I)
    where `latents` is the sampled z used in rollout.

    Args:
        latent_poss: 1D LongTensor/list of positions for latent tokens (length L).
        latents:     Tensor of shape [L, D], rollout latents (z) at those positions.
        last_hidden_state: Tensor of shape [B, T, D], hidden states; we use batch 0.

    Returns:
        logp_sum: scalar tensor, sum of log-probs over all latent positions.
                  This is the sample-level log-prob commonly used in PPO/GRPO.
    """
    # Shape: [L, D]
    latent_outputs = last_hidden_state[0, latent_poss, :]
    latents = latents.to(latent_outputs)

    # Per-position log-prob of N(mu=latent_outputs, sigma^2 I)
    # log N(z; mu, sigma^2 I) = -0.5 * ||z - mu||^2 / sigma^2 - (D/2)*log(2*pi*sigma^2), removed const
    diff2 = (latents - latent_outputs).pow(2).sum(dim=-1)           # [L]
    latent_log_probs = - 0.5 * diff2 / (sigma ** 2)               # [L]
    return latent_log_probs


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits

    def _forward_micro_batch(self, micro_batch: Dict[str, torch.Tensor], temperature: float) -> torch.Tensor:
        """
        Returns:
            log_probs: # (bs, response_len)
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        latent_poss = None
        latents = None
        if self.config.sampling_strategy == 'avt' and not self.config.ablate_latent:
            try:
                latent_poss = []
                start_id = int(os.getenv("ABS_VIS_START_ID"))
                end_id = int(os.getenv("ABS_VIS_END_ID"))
                _, varlen_by_batch = collect_varlen_segment_indices(
                    input_ids=micro_batch["input_ids"], # (micro_batch["input_ids"][1]==151666).nonzero() (micro_batch["input_ids"][1]==151667).nonzero()
                    attention_mask=micro_batch["attention_mask"],
                    start_id=start_id, end_id=end_id,
                )

                #latent_cnt = sum([1 for lat in micro_batch['latents'] if lat is not None])
                #if len(latent_poss) != latent_cnt:
                #    print(f"[WARNING] Number of latents {latent_cnt} != number of latent pad segments {len(latent_poss)}. Skip this mirco batch for latent policy gradient computing.")

                latents_list, per_sample = [], []
                for i, lat in enumerate(micro_batch['latents']):
                    if lat is not None:
                        t = torch.tensor(lat)  # (steps, D)
                        poss_cnt = sum(v.numel() for v in varlen_by_batch[i]) if i < len(varlen_by_batch) else 0
                        if t.shape[0]!=poss_cnt:
                            print(f"[WARNING] A latent segment in a sample has different numbers of latent {t.shape[0]} and latent pad {poss_cnt}. Skip this sample for latent policy gradient computing.")
                            continue
                        latents_list.append(t)
                        latent_poss.extend(varlen_by_batch[i])
                        per_sample.append((i, t.shape[0], poss_cnt))

                if len(latents_list) > 0 and len(latent_poss) > 0:
                    latent_poss = torch.cat(latent_poss, dim=0)
                    latents = torch.cat(latents_list, dim=0).to(input_ids.device)
                    #if os.getenv("AVT_DEBUG") == "1":
                    if latents.shape[0] != latent_poss.shape[0]:
                        print(f"[WARNING] latents.shape[0] != latent_poss.shape[0], per-sample (idx, lat, poss)={per_sample}, total lat={latents.shape[0]}, poss={int(latent_poss.numel())}. Skip this mirco batch for latent policy gradient computing", flush=True)
                        output_hidden_states = False
                        latent_poss = None
                        latents = None
                        #if self.rank == 0:
                        #    pdb.set_trace()
                else:
                    latent_poss = None
                
                if latents is not None and latent_poss is not None:
                    output_hidden_states = True
                else:
                    output_hidden_states = False
            except Exception:
                print(f"[WARNING] Unexpected error before the latent importance sampling. Fall back to vanilla prob computation for this mirco batch.")
                output_hidden_states = False
                pass
        else:
            output_hidden_states = False

        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                )
        #breakpoint()
        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(
                input_ids.unsqueeze(-1), attention_mask
            )  # input_ids_rmpad (total_nnz, ...)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_sequence_parallel_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_sequence_parallel_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_sequence_parallel_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

            # only pass input_ids and position_ids to enable flash_attn_varlen
            # breakpoint()
            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
                latent_poss=latent_poss,
                latents=latents,
                output_hidden_states=output_hidden_states, # AXZ
                #return_dict=True # AXZ
            )  # prevent model thinks we are generating
            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
            logits_rmpad.div_(temperature)
            #breakpoint()
            # ((total_nnz / sp) + pad)
            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)
            if self.config.sampling_strategy == 'avt' and not self.config.ablate_latent:
                if latents is not None:
                    latent_log_probs = compute_latent_log_probs(latent_poss, latents, output.hidden_states[-1], sigma=self.config.avt_rl_sigma)
                    log_probs[latent_poss] = latent_log_probs.to(log_probs.dtype)
                    #pdb.set_trace()
                    # compute_latent_log_probs(latent_poss, latents, output.hidden_states[-1], sigma=10).mean()
            
            # gather log_prob if sp > 1
            if self.config.ulysses_sequence_parallel_size > 1:
                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            # pad back to (bsz, seqlen)
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
        else:
            #breakpoint()
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits: torch.Tensor = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
            log_probs = self.log_probs_from_logits(logits, responses)  # (bsz, response_length)

        return log_probs

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            self.actor_optimizer.step()

        self.actor_optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if "multi_modal_inputs" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["multi_modal_inputs"]
        else:
            non_tensor_select_keys = []

        if self.config.sampling_strategy == "avt":
            non_tensor_select_keys.append('latents')
        #
        micro_batches = data.select(select_keys, non_tensor_select_keys).split(
            self.config.micro_batch_size_per_device_for_experience
        )
        log_probs_lst = []
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=5 if self.config.sampling_strategy == "mc" else 2)
        #breakpoint()
        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
            log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        return log_probs

    def update_policy(self, data: DataProto) -> Dict[str, Any]:
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if self.config.use_kl_loss and not self.config.disable_kl:
            select_keys.append("ref_log_probs")

        if "multi_modal_inputs" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["multi_modal_inputs"]
        else:
            non_tensor_select_keys = []
        
        if self.config.sampling_strategy == "avt":
            non_tensor_select_keys.append('latents')

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        #breakpoint()
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)
        #breakpoint()
        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=6 if self.config.sampling_strategy == "mc" else 2)

            for mini_batch in mini_batches:
                gradient_accumulation = (
                    self.config.global_batch_size_per_device // self.config.micro_batch_size_per_device_for_update
                )
                micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)
                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=7 if self.config.sampling_strategy == "mc" else 3)

                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    responses = model_inputs["responses"]
                    response_length = responses.size(1)
                    attention_mask = model_inputs["attention_mask"]
                    response_mask = attention_mask[:, -response_length:]
                    old_log_probs = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    # all return: (bsz, response_length)
                    log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
                    entropy_loss = -VF.masked_mean(log_probs, response_mask)  # estimator of entropy loss

                    pg_loss, pg_clipfrac_higher, pg_clipfrac_lower, ppo_kl = core_algos.compute_policy_loss(
                        old_log_probs=old_log_probs,
                        log_probs=log_probs,
                        advantages=advantages,
                        response_mask=response_mask,
                        clip_ratio_low=self.config.clip_ratio_low,
                        clip_ratio_high=self.config.clip_ratio_high,
                        clip_ratio_dual=self.config.clip_ratio_dual,
                    )
                    if "ref_log_probs" in model_inputs:
                        ref_log_probs = model_inputs["ref_log_probs"]
                        # compute kl loss
                        kld = core_algos.compute_kl(
                            log_probs=log_probs,
                            ref_log_probs=ref_log_probs,
                            kl_penalty=self.config.kl_penalty,
                        )
                        kl_loss = VF.masked_mean(kld, response_mask)
                        pg_loss = pg_loss + kl_loss * self.config.kl_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_coef

                    loss = pg_loss / gradient_accumulation
                    loss.backward()

                    batch_metrics = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac_higher": pg_clipfrac_higher.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        "actor/entropy_loss": entropy_loss.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                    }
                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        return metrics
