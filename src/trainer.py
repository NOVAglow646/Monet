from trl import SFTTrainer, SFTConfig
from typing import Optional
from transformers import TrainerCallback
import logging
import torch
import os, csv, torch, datetime
import gc
import numpy as np
from .utils import SFTRepAnalyzer
import math
from time import time


def compute_latents_only_loss(latents, loss_for_latents):
    def _flatten_tensors(x):
                # Flatten nested [list/tuple of Tensors] into a flat list of Tensors
                if isinstance(x, (list, tuple)):
                    out = []
                    for y in x:
                        out.extend(_flatten_tensors(y))
                    return out
                return [x]

    ce_vec_list = _flatten_tensors(latents)
    grads = torch.autograd.grad(
        outputs=loss_for_latents,
        inputs=ce_vec_list,
        retain_graph=True,   # we won't reuse the 3rd graph
        create_graph=False,   # stop higher-order graph
        allow_unused=True     # in case some ce vectors are not used
    )

    # Replace None with zeros for unused elements
    safe_grads = []
    for v, g in zip(ce_vec_list, grads):
        if g is None:
            # Create a zero tensor on the same device/dtype/shape
            g = torch.zeros_like(v)
        safe_grads.append(g.detach())  # detach to stop any 3rd-forward param path

    proxy_loss = torch.stack([(v * g).sum() for v, g in zip(ce_vec_list, safe_grads)]).sum()
    return proxy_loss


class CustomTrainerStage1(SFTTrainer):
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        (ce_loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        predict_embeddings = outputs.hidden_states
        image_out_mask = inputs["image_out_mask"]

        shift_image_mask = image_out_mask[:, -(predict_embeddings.shape[1] - 1) :].to(predict_embeddings.device)
        shift_predict_embeddings = predict_embeddings[..., :-1, :][shift_image_mask.to(predict_embeddings.device) != 0].contiguous()

        input_embeddings = outputs.inputs_embeds
        gt_embeddings = input_embeddings[..., 1:, :][shift_image_mask.to(input_embeddings.device) != 0].contiguous()
        
        sim_loss = torch.nn.functional.cosine_similarity(gt_embeddings, shift_predict_embeddings).mean()
        sim_loss = 1 - sim_loss

        loss = 0.1 * ce_loss + sim_loss
        return (loss, outputs) if return_outputs else loss

class CustomTrainerStage2(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        (ce_loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        loss = ce_loss
        return (loss, outputs) if return_outputs else loss
    
class CustomTrainerAVTStage1(SFTTrainer):
    def __init__(self, *args, **kwargs): 
        self.exp_name =kwargs.pop('exp_name')
        super().__init__(*args, **kwargs)
        self.weight = self.args.alignment_weight
        self.ce_emphasize_factor: float = float(getattr(self.args, 'ce_emphasize_factor', 1.0))
        # 仅 rank‑0 进程写文件，防止多卡重复
        self.is_main_process = (
            not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )
    # Gradient checkpointing is controlled per-forward in compute_loss
        # 日志文件路径
        log_dir = self.args.logging_dir or "./logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        self.loss_log_path = os.path.join(log_dir, f"loss_history/loss_history_w{self.weight}_{self.exp_name}_{timestamp}.csv")

        # 如果文件不存在，就写表头
        if self.is_main_process and not os.path.exists(self.loss_log_path):
            with open(self.loss_log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "global_step","epoch",
                    "loss_total",
                    "loss_student_ce",
                    "loss_align"
                ])
        
        self._al_loss_cum = 0.0       # cumulative alignment loss since last log
        self._al_steps = 0            # number of micro-steps accumulated
        self.student_ce_loss_cum = 0.0        # cumulative student CE loss
        self.student_ce_loss_steps = 0
                
    def alignment_loss(self, student_reps_all_layers, teacher_reps_all_layers):
        layerwise_sim_record = []
        bsz = len(student_reps_all_layers[0]) if len(student_reps_all_layers) > 0 else 1
        total_loss = 0.0
        for student_reps_l, teacher_reps_l in zip(student_reps_all_layers, teacher_reps_all_layers):
            layer_loss = 0.0
            layerwise_sim = 0.0
            for student_rep_l_b, teacher_rep_l_b in zip(student_reps_l, teacher_reps_l):
                if student_rep_l_b.shape[0] == 0 or teacher_rep_l_b.shape[0] == 0:
                    continue
                # 避免零范数导致 NaN，增加 eps
                sim = torch.nn.functional.cosine_similarity(student_rep_l_b, teacher_rep_l_b, eps=1e-6).mean()
                layer_loss += (1 - sim)
                layerwise_sim += float(sim.detach().item())
            total_loss += layer_loss / max(1, bsz)
            layerwise_sim_record.append(layerwise_sim / max(1, bsz))
        total_loss = total_loss / max(1, len(student_reps_all_layers))
        return total_loss
        #if return_outputs:
        #    return {"total_loss": loss, "ce_loss": ce_loss, "sim_loss": sim_loss}, outputs
        #return {"total_loss": loss, "ce_loss": ce_loss, "sim_loss": sim_loss}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        inputs['latent_mode'] = False
        inputs['input_ids'] = inputs['teacher_input_ids']
        inputs['attention_mask'] = inputs['teacher_attention_mask']
        inputs['pixel_values'] = inputs['user_assistant_pixel_values']
        inputs['image_grid_thw'] = inputs['user_assistant_image_grid_thw']
        inputs['labels'] = None #inputs['teacher_labels'] # We needn't compute the ce loss for the teacher input in this stage
        inputs['alignment_poss'] = inputs['teacher_alignment_poss']
        inputs['image_out_mask'] = inputs['teacher_image_out_mask']
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        inputs['loss_type'] = ['ce']
        
        with torch.no_grad():
            teacher_outputs = model(**inputs, return_dict=True, output_hidden_states=True)
            
        inputs['latent_mode'] = True
        inputs['input_ids'] = inputs['student_input_ids']
        inputs['attention_mask'] = inputs['student_attention_mask']
        inputs['pixel_values'] = inputs['user_pixel_values']
        inputs['image_grid_thw'] = inputs['user_image_grid_thw']
        inputs['labels'] = inputs['student_labels']
        inputs['alignment_poss'] = inputs['student_alignment_poss']
        inputs['image_out_mask'] = inputs['student_image_out_mask']
        inputs['teacher_hidden_states_for_alignment'] = teacher_outputs.hidden_states
        model.gradient_checkpointing_disable()
        inputs['loss_type'] = ['alignment']
        (alignment_loss, student_outputs) = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
 
        inputs['latent_mode'] = False
        inputs['ce_patch_pos'] = student_outputs.ce_patch_pos
        inputs['ce_patch_vec'] = student_outputs.ce_patch_vec
        inputs['ce_emphasize_factor'] = self.ce_emphasize_factor
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        inputs['loss_type'] = ['ce']
        (student_ce_loss, student_outputs) = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        alignment_loss = alignment_loss.to(student_ce_loss.device, dtype=student_ce_loss.dtype)

        # If self.weight might be a tensor on CPU or a list, normalize it
        if isinstance(self.weight, torch.Tensor):
            self.weight = self.weight.to(student_ce_loss.device, dtype=student_ce_loss.dtype)
        else:
            # cast python float to the same dtype for safety
            self.weight = student_ce_loss.new_tensor(float(self.weight))
        loss = student_ce_loss + self.weight *alignment_loss

        outputs_student_loss = student_ce_loss.item()

        # Avoid per-step cache clearing which forces device sync and hurts utilization.
        # Just release references; optionally run a light GC periodically.
        del student_outputs, teacher_outputs
        step = int(getattr(self.state, 'global_step', 0) or 0)
        if self.is_main_process and step > 0 and (step % 20 == 0):
            try:
                gc.collect()
                torch.cuda.empty_cache()
                # DO NOT call torch.cuda.empty_cache() every step; it stalls the GPU.
            except Exception:
                pass
        
        # -------- wandb logging --------
        al_val = float(alignment_loss.detach().item()) if torch.is_tensor(alignment_loss) else float(alignment_loss)
        self._al_loss_cum += al_val
        self._al_steps += 1
        self.student_ce_loss_cum += float(student_ce_loss.detach().item())
        self.student_ce_loss_steps += 1

        # --------  local logging  --------
        if self.is_main_process:
            with open(self.loss_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.state.global_step,
                    self.state.epoch,
                    loss.item(),
                    outputs_student_loss,
                    alignment_loss.item() if isinstance(alignment_loss, torch.Tensor) else alignment_loss,
                ])
        # --------------------------------------------
        
        
        return (loss, None) if return_outputs else loss
    
    def log(self, logs: dict, start_time: float | None = None):
        # Merge our rolling averages into the standard logs once per logging call
        merged = dict(logs)
        if self._al_steps > 0:
            merged["alignment_loss"] = round(self._al_loss_cum / max(1, self._al_steps), 6)
        if self.student_ce_loss_steps > 0:
            merged["student_ce_loss"] = round(self.student_ce_loss_cum / max(1, self.student_ce_loss_steps), 6)

        # Reset accumulators after logging so the next window starts fresh
        self._al_loss_cum = 0.0
        self._al_steps = 0
        self.student_ce_loss_cum = 0.0
        self.student_ce_loss_steps = 0

        # Call parent to keep default behavior (console/TB/W&B/etc.)
        return super().log(merged, start_time)
    
class CustomTrainerSFT(SFTTrainer):
    def __init__(self, *args, **kwargs):
        self.exp_name =kwargs.pop('exp_name')
        # accept processing_class (preferred) and fall back to tokenizer for backward compat
        if 'processing_class' not in kwargs and 'tokenizer' in kwargs:
            kwargs['processing_class'] = kwargs.pop('tokenizer')
        super().__init__(*args, **kwargs)
        self.weight = 1.0
        # ce_emphasize_factor warmup configuration
        # Target factor (backward compatible with existing arg name)
        self._ce_emphasize_target: float = float(getattr(self.args, 'ce_emphasize_factor', 1.0))
        # Optional absolute warmup steps takes precedence over ratio
        self._ce_emphasize_warmup_steps: int = int(getattr(self.args, 'ce_emphasize_warmup_steps', 0) or 0)

        # Representation analysis
        self.rep_analyzer = None
        args_cfg = self.args
        if getattr(args_cfg, 'sft_analysis_enable', False):
            self.rep_analyzer = SFTRepAnalyzer(
                save_dir=args_cfg.sft_analysis_save_dir,
                categories=args_cfg.sft_analysis_categories,
                dataset_names=self.args.dataset_names,
                exp_name=self.args.exp_name
            )
            if getattr(self, 'is_main_process', True):
                logging.info(f"[SFT Analysis] Analyzer initialized. Save dir={args_cfg.sft_analysis_save_dir}; Categories={args_cfg.sft_analysis_categories}")
            # subset selection deferred to training loop external orchestration
        # 仅 rank‑0 进程写文件，防止多卡重复
        self.is_main_process = (
            not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )

        self.observation_token_acc = 0.
        self.observation_token_acc_step = 0
        self.teacher_ce_cum = 0.0        # cumulative student CE loss
        self.teacher_ce_steps = 0
        

    def _current_ce_emphasize_factor(self) -> float:
        """Linear warmup from 1.0 -> target over N steps or ratio of total steps.

        Priority: ce_emphasize_warmup_steps (absolute) > ce_emphasize_warmup_ratio > no warmup.
        After warmup, clamp to target. If target <= 1.0, return target directly.
        """
        target = float(self._ce_emphasize_target)
        if target == 1.0:
            return 1.0

        # Decide warmup steps
        warmup_steps = int(self._ce_emphasize_warmup_steps or 0)

        if warmup_steps <= 0:
            # No warmup configured or cannot determine steps
            return target

        # Use current global_step (before increment) for smooth schedule in training loop
        gs = int(getattr(self.state, 'global_step', 0) or 0)
        progress = min(1.0, max(0.0, gs / float(max(1, warmup_steps))))
        return float(1.0 + (target - 1.0) * progress)


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        inputs['latent_mode'] = False
        inputs['input_ids'] = inputs['teacher_input_ids']
        inputs['attention_mask'] = inputs['teacher_attention_mask']
        inputs['pixel_values'] = inputs['teacher_pixel_values']
        inputs['image_grid_thw'] = inputs['teacher_image_grid_thw']
        inputs['labels'] = inputs['teacher_labels']
        # Collect available position categories dynamically (supports non_observation_poss)
        '''poss_dict = {}
        for k in ['boxed_start_poss','observation_poss','non_observation_poss']:
            if k in inputs:
                poss_dict[k] = inputs[k]'''
        inputs['ce_emphasize_poss'] = inputs['teacher_observation_poss']
        # Dynamic warmup factor passed to model.forward
        inputs['ce_emphasize_factor'] = self.args.ce_emphasize_factor
        inputs['loss_type'] = ['ce']
        inputs['compute_emphasize_acc'] = True
        (teacher_ce_loss, teacher_outputs) = super().compute_loss(
                model, 
                inputs,
                return_outputs=True, num_items_in_batch=num_items_in_batch
            )

        self.teacher_ce_cum += teacher_ce_loss.item()
        self.teacher_ce_steps += 1

        if getattr(teacher_outputs, 'mean_emphasize_acc', None) is not None:
            self.observation_token_acc += getattr(teacher_outputs, 'mean_emphasize_acc')
            self.observation_token_acc_step += 1

        # Representation analysis update
        '''if self.rep_analyzer is not None and 'sample_id' in inputs:
            with torch.no_grad():
                sample_ids = inputs['sample_id']  # assume shape (B,)
                hidden_states = teacher_outputs.hidden_states  # list[L] each (B,S,H)
                if hidden_states is not None:
                    for b, sid in enumerate(sample_ids):
                        sid_int = int(sid)
                        if self.rep_analyzer.is_tracked(sid_int):
                            # expect each value is List[List[int]] of length B
                            pos_dict = {}
                            for cat, batch_poss in poss_dict.items():
                                if isinstance(batch_poss, (list, tuple)) and len(batch_poss) > b:
                                    pos_dict[cat] = batch_poss[b]
                            # Build baseline lazily first time
                            if sid_int not in self.rep_analyzer.baseline:
                                self.rep_analyzer.build_baseline(sid_int, [h[[b]] for h in hidden_states])
                            # Update current cos
                            self.rep_analyzer.update(
                                sample_id=sid_int,
                                hidden_states=[h[[b]] for h in hidden_states],
                                pos_dict=pos_dict,
                                epoch=int(self.state.epoch if self.state.epoch is not None else 0),
                                global_step=int(self.state.global_step)
                            )'''

        del teacher_outputs
        gc.collect()
        torch.cuda.empty_cache()
        
        return (teacher_ce_loss, None) if return_outputs else teacher_ce_loss

    def on_epoch_end(self):
        # 注意：HF Trainer 不会自动调用子类自定义的 on_epoch_end；实际汇总通过回调实现。
        return super().on_epoch_end()

    def log(self, logs: dict, start_time: float | None = None):
        # Merge our rolling averages into the standard logs once per logging call
        merged = dict(logs)
        if self.teacher_ce_steps > 0:
            merged["student_ce_loss"] = round(self.teacher_ce_cum / max(1, self.teacher_ce_steps), 6)
            self.teacher_ce_cum = 0.0
            self.teacher_ce_steps = 0
        if self.observation_token_acc_step > 0:
            merged["observation_token_acc"] = round(self.observation_token_acc/ max(1, self.observation_token_acc_step), 6)
            self.observation_token_acc = 0.
            self.observation_token_acc_step = 0

        # Call parent to keep default behavior (console/TB/W&B/etc.)
        return super().log(merged, start_time)

class CustomTrainerAVT_V2_Stage1(SFTTrainer):
    def __init__(self, *args, **kwargs):
        self.exp_name = kwargs.pop('exp_name')
        # accept processing_class (preferred) and fall back to tokenizer for backward compat
        if 'processing_class' not in kwargs and 'tokenizer' in kwargs:
            kwargs['processing_class'] = kwargs.pop('tokenizer')
        super().__init__(*args, **kwargs)

        # Representation analysis
        self.rep_analyzer = None
        args_cfg = self.args
        if getattr(args_cfg, 'sft_analysis_enable', False):
            self.rep_analyzer = SFTRepAnalyzer(
                save_dir=args_cfg.sft_analysis_save_dir,
                categories=args_cfg.sft_analysis_categories,
                dataset_names=self.args.dataset_names,
                exp_name=self.args.exp_name
            )
            if getattr(self, 'is_main_process', True):
                logging.info(f"[SFT Analysis] Analyzer initialized. Save dir={args_cfg.sft_analysis_save_dir}; Categories={args_cfg.sft_analysis_categories}")
            # subset selection deferred to training loop external orchestration
        # 仅 rank‑0 进程写文件，防止多卡重复
        self.is_main_process = (
            not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )
        self.ce_emphasize_factor = self.args.ce_emphasize_factor
        self.observation_token_acc = 0.
        self.observation_token_acc_step = 0

        # Cache special token ids for attention attribution
        try:
            _tok = getattr(self, 'processing_class', None)
            _tok = getattr(_tok, 'tokenizer', None)
            if _tok is not None:
                # Each returns a 1x1 tensor, take scalar id
                self._img_pad_id = int(_tok("<|image_pad|>", return_tensors="pt")["input_ids"][0][0].item())
                self._latent_pad_id = int(_tok("<abs_vis_token_pad>", return_tensors="pt")["input_ids"][0][0].item())
            else:
                # Fallback to config-known latent id if tokenizer missing
                self._img_pad_id = None
                self._latent_pad_id = int(getattr(self.model.config, 'latent_token_id', -1))
        except Exception:
            self._img_pad_id = None
            self._latent_pad_id = int(getattr(self.model.config, 'latent_token_id', -1))

        # Rolling accumulators for logging
        self._obs_to_img_att_cum = 0.0
        self._obs_to_img_att_steps = 0
        self._obs_to_latent_att_cum = 0.0
        self._obs_to_latent_att_steps = 0
        # New: rolling accumulators for question image vs rest images, latent, and other-prev tokens
        self._obs_to_qimg_att_mean_cum = 0.0
        self._obs_to_qimg_att_mean_steps = 0
        self._obs_to_qimg_att_sum_cum = 0.0
        self._obs_to_qimg_att_sum_steps = 0

        self._obs_to_img_rest_att_mean_cum = 0.0
        self._obs_to_img_rest_att_mean_steps = 0
        self._obs_to_img_rest_att_sum_cum = 0.0
        self._obs_to_img_rest_att_sum_steps = 0

        self._obs_to_latent_att_sum_cum = 0.0  # keeping existing mean; add sum
        self._obs_to_latent_att_sum_steps = 0

        self._obs_to_other_prev_att_mean_cum = 0.0
        self._obs_to_other_prev_att_mean_steps = 0
        self._obs_to_other_prev_att_sum_cum = 0.0
        self._obs_to_other_prev_att_sum_steps = 0

        # Keep last-per-layer attention stats for saving at log step
        self._last_attn_layer_stats = None  # dict or None
        self._attn_save_dir = os.path.join(self.args.logging_dir or './logs', 'attn_analysis')
        os.makedirs(self._attn_save_dir, exist_ok=True)
        # Rolling avg for auxiliary emphasize_latent_attn loss
        self._emph_loss_cum = 0.0
        self._emph_steps = 0

        # align vision latent loss
        self.align_vision_latent_loss_cum = 0.
        self.align_vision_latent_loss_steps = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        inputs['stage'] = 'avt_v2_stage1'
        inputs['latent_mode'] = True
        inputs['loss_type'] = []
        #inputs['enable_ce_checkpoint'] = False
        outputs = model(**inputs, return_dict=True, output_hidden_states=False)

        # Separate, no_grad forward for attention analysis (interval-gated) BEFORE enabling grad checkpointing
        # This ensures no state toggles between training forward and backward.
        if getattr(self.args, 'attn_analysis', False):
            self.attn_analysis(model, inputs, outputs)

        # After analysis, run the CE training forward with grad checkpointing enabled and use_cache disabled
        # Enforce use_cache=False to avoid recompute mismatches with checkpointing
        try:
            if getattr(model.config, 'use_cache', None) is not False:
                model.config.use_cache = False
        except Exception:
            pass
        
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        inputs['latent_mode'] = False
        inputs['ce_patch_pos'] = outputs.ce_patch_pos
        inputs['ce_patch_vec'] = outputs.ce_patch_vec
        inputs['ce_emphasize_poss'] = inputs['observation_poss']
        # Dynamic warmup factor passed to model.forward
        inputs['ce_emphasize_factor'] = self.ce_emphasize_factor
        inputs['loss_type'] = ['ce']
        
        # Decide whether to also compute emphasize_latent_attn in same forward
        use_emph = bool(getattr(self.args, 'use_emphasize_latent_attn_loss', False))
        emph_coef = float(getattr(self.args, 'emphasize_latent_attn_coef', 1.0)) 
        if use_emph:
            inputs['loss_type'].append('emphasize_latent_attn')
            inputs['collect_emphasize_attn'] = True
            # which query tokens to use (default observation_poss)
            inputs['emphasize_query_poss'] = inputs.get('observation_poss', None)
            inputs['emphasize_topk_layers'] = self.args.emphasize_topk_layers
            inputs['attn_loss_layers'] = self.args.attn_loss_layers
        
        if self.args.use_align_vision_latent_loss_projector:
            inputs['loss_type'].append('align_vision_latent_projector')
            inputs['latent_size'] = self.args.latent_size

        if self.args.use_align_vision_latent_loss_pooling:
            inputs['loss_type'].append('align_vision_latent_pooling')
            inputs['latent_size'] = self.args.latent_size

        inputs['compute_emphasize_acc'] = True
        # Ensure training forward does NOT request attentions (prevents checkpoint recompute mismatch)
        inputs.pop('output_attentions', None)
        inputs.pop('attn_analysis', None)
        inputs.pop('attention_mask_4d')

        teacher_ce_loss, teacher_output = super().compute_loss(
                model, 
                inputs,
                return_outputs=True, num_items_in_batch=num_items_in_batch
            )

        # If auxiliary loss is present, combine here
        total_loss = teacher_ce_loss
        emph_loss_val = None
        if use_emph and hasattr(teacher_output, 'loss_dict') and isinstance(teacher_output.loss_dict, dict):
            emph = teacher_output.loss_dict.get('emphasize_latent_attn', None)
            if emph is not None:
                emph_loss_val = float(emph.detach().item())
                # Combine into the returned loss
                total_loss = teacher_ce_loss + emph * emph_coef
                # accumulate for logging
                self._emph_loss_cum += emph_loss_val
                self._emph_steps += 1
        
        if getattr(teacher_output, 'mean_emphasize_acc', None) is not None:
            self.observation_token_acc += getattr(teacher_output, 'mean_emphasize_acc')
            self.observation_token_acc_step += 1

        if self.args.use_align_vision_latent_loss_projector:
            align_vision_latent_loss = teacher_output.loss_dict['align_vision_latent_projector']
            total_loss += self.args.align_vision_latent_loss_weight * align_vision_latent_loss
            self.align_vision_latent_loss_cum += align_vision_latent_loss.item()
            self.align_vision_latent_loss_steps += 1
        
        if self.args.use_align_vision_latent_loss_pooling:
            align_vision_latent_loss = teacher_output.loss_dict['align_vision_latent_pooling']
            total_loss += self.args.align_vision_latent_loss_weight * align_vision_latent_loss
            self.align_vision_latent_loss_cum += align_vision_latent_loss.item()
            self.align_vision_latent_loss_steps += 1

        latent_only_loss = compute_latents_only_loss(outputs.ce_patch_vec, total_loss)
        total_loss = latent_only_loss * self.args.emphasize_latent_weight + total_loss

        # For return_outputs == True, we must return our combined loss
        if return_outputs:
            return (total_loss, None)

        # Light-touch cleanup without forcing GPU sync every step
        #del teacher_outputs
        step = int(getattr(self.state, 'global_step', 0) or 0)
        if step % 50 == 0:
            try:
                gc.collect()
                # Avoid calling empty_cache() each step
                torch.cuda.empty_cache()
            except Exception:
                pass

        return total_loss

    def attn_analysis(self, model, inputs, outputs):
        try:
            step_now = int(getattr(self.state, 'global_step', 0) or 0)
            interval = int(getattr(self.args, 'attn_analysis_interval', 0) or getattr(self.args, 'logging_steps', 0) or 50)
            if interval <= 0:
                interval = 50
            if (step_now % interval) == 0:
                analysis_inputs = dict(inputs)
                analysis_inputs['output_attentions'] = True
                analysis_inputs['attn_analysis'] = True
                analysis_inputs['loss_type'] = []
                analysis_inputs['labels'] = None
                analysis_inputs['latent_mode'] = False
                analysis_inputs['ce_patch_pos'] = outputs.ce_patch_pos
                analysis_inputs['ce_patch_vec'] = outputs.ce_patch_vec
                bool_attn_mask_4d = inputs['attention_mask_4d']['full_attention']
                analysis_inputs['attention_mask_4d'] = {"full_attention": (~bool_attn_mask_4d).float()*-1e7}
                prev_attn_impl = getattr(model.config, '_attn_implementation', None)
                was_training = model.training
                model.eval()
                model.config._attn_implementation = 'eager'

                with torch.no_grad():
                    analysis_out = model(**analysis_inputs, return_dict=True, output_hidden_states=False)

                atts = getattr(analysis_out, 'attentions', None)
                input_ids = analysis_inputs.get('input_ids', None)
                obs_poss = inputs.get('observation_poss', None)
                if atts is not None and input_ids is not None and obs_poss is not None and len(atts) > 0:
                    layers = []
                    for a in atts:
                        t = a[0] if isinstance(a, (list, tuple)) else a
                        if torch.is_tensor(t) and t.dim() == 4:
                            layers.append(t)
                    if len(layers) > 0:
                        B, H, S, _ = layers[0].shape
                        device = layers[0].device
                        ids = input_ids.to(device)
                        # Build obs/query mask (B,S)
                        obs_mask = torch.zeros((B, S), dtype=torch.bool, device=device)
                        for b, poss in enumerate(obs_poss):
                            if poss is None:
                                continue
                            valid = [p for p in poss if isinstance(p, int) and 0 <= p < S]
                            if len(valid) > 0:
                                obs_mask[b, torch.tensor(valid, device=device, dtype=torch.long)] = True

                        # Image masks
                        img_mask = (ids == self._img_pad_id).to(device) if self._img_pad_id is not None else None
                        # Split question image vs rest images per sample: first contiguous block of img tokens
                        qimg_mask = None
                        img_rest_mask = None
                        if img_mask is not None:
                            qimg_mask = torch.zeros_like(img_mask)
                            img_rest_mask = torch.zeros_like(img_mask)
                            for b in range(B):
                                row = img_mask[b]
                                idx = (row.nonzero(as_tuple=False).flatten())
                                if idx.numel() == 0:
                                    continue
                                start = int(idx[0].item())
                                # extend contiguous block
                                end = start
                                while end + 1 < row.numel() and row[end + 1].item():
                                    end += 1
                                qimg_mask[b, start : end + 1] = True
                                # rest are all other image positions
                                if idx.numel() > (end - start + 1):
                                    img_rest_mask[b] = row & (~qimg_mask[b])
                # Latent masks
                latent_mask = (ids == self._latent_pad_id).to(device) if (self._latent_pad_id is not None and self._latent_pad_id >= 0) else None
                # Other (non-image, non-latent) masks
                other_mask = None
                if img_mask is not None or latent_mask is not None:
                    other_mask = torch.ones((B, S), dtype=torch.bool, device=device)
                    if img_mask is not None:
                        other_mask &= (~img_mask)
                    if latent_mask is not None:
                        other_mask &= (~latent_mask)

                # Strictly lower triangular mask for k < q (S,S)
                tril = torch.tril(torch.ones((S, S), dtype=torch.bool, device=device), diagonal=-1)

                def _mean_and_sum(att: torch.Tensor, mask_k: torch.Tensor, restrict_prev: bool = False):
                    """
                    mean: 与原先一致，按 (B,H,q,k) 对所有有效 pair 求平均（再对 batch 取平均）。
                    sum: 按需求修改：先对同一个 query 的该类 key 做 sum（跨 k），
                            然后对所有 query 做平均（再对头与 batch 做平均）。
                    """
                    if mask_k is None:
                        return None, None
                    # Masks
                    q_mask_b1qs1 = obs_mask[:, None, :, None]  # (B,1,S,1)
                    k_mask_b11s  = mask_k[:, None, None, :]    # (B,1,1,S)
                    pair_mask = q_mask_b1qs1 & k_mask_b11s     # (B,1,S,S)
                    if restrict_prev:
                        pair_mask = pair_mask & tril[None, None, :, :]

                    # If no valid pairs, return Nones
                    num_pairs = pair_mask.sum(dim=(1, 2, 3))  # (B,)
                    if (num_pairs == 0).all():
                        return None, None

                    # mean over pairs and heads (old definition)
                    masked_sum_all = (att * pair_mask).sum(dim=(1, 2, 3))  # (B,)
                    denom = num_pairs * max(1, H)
                    mean_b = masked_sum_all / denom.clamp_min(1)
                    valid_b = num_pairs > 0
                    mean_val = mean_b[valid_b].mean() if valid_b.any() else None

                    # sum metric (new definition):
                    # 1) per-query sum over keys in category: sum_k att[b,h,q,k]
                    #    pair_mask ensures only selected keys and queries contribute.
                    per_q_sum = (att * pair_mask).sum(dim=-1)  # (B,H,S)
                    # 2) average across queries (only where obs_mask True)
                    obs_mask_bhs = obs_mask[:, None, :]  # (B,1,S)
                    q_count = obs_mask.sum(dim=-1)       # (B,)
                    # Avoid div by 0: only keep batches with q_count>0
                    if (q_count > 0).any():
                        # Sum over queries then divide by #queries per batch
                        per_bh_avg = (per_q_sum * obs_mask_bhs).sum(dim=-1)  # (B,H)
                        per_bh_avg = per_bh_avg / q_count.clamp_min(1)[:, None]
                        # 3) average across heads, then across valid batches
                        per_b_avg = per_bh_avg.mean(dim=1)  # (B)
                        valid_q = q_count > 0
                        sum_val = per_b_avg[valid_q].mean() if valid_q.any() else None
                    else:
                        sum_val = None

                    return mean_val, sum_val

                # Per-layer arrays to save
                L = len(layers)
                per_layer = {
                    'mean_qimg': torch.full((L,), float('nan')),
                    'sum_qimg': torch.full((L,), float('nan')),
                    'mean_img_rest': torch.full((L,), float('nan')),
                    'sum_img_rest': torch.full((L,), float('nan')),
                    'mean_latent': torch.full((L,), float('nan')),
                    'sum_latent': torch.full((L,), float('nan')),
                    'mean_other_prev': torch.full((L,), float('nan')),
                    'sum_other_prev': torch.full((L,), float('nan')),
                }

                for li, att in enumerate(layers):
                    # Compute metrics
                    m_qimg, s_qimg = _mean_and_sum(att, qimg_mask, restrict_prev=False)
                    m_img_r, s_img_r = _mean_and_sum(att, img_rest_mask, restrict_prev=False)
                    m_lat, s_lat = _mean_and_sum(att, latent_mask, restrict_prev=False)
                    m_oth, s_oth = _mean_and_sum(att, other_mask, restrict_prev=True)

                    if m_qimg is not None and torch.isfinite(m_qimg):
                        per_layer['mean_qimg'][li] = m_qimg.detach().float()
                    if s_qimg is not None and torch.isfinite(s_qimg):
                        per_layer['sum_qimg'][li] = s_qimg.detach().float()
                    if m_img_r is not None and torch.isfinite(m_img_r):
                        per_layer['mean_img_rest'][li] = m_img_r.detach().float()
                    if s_img_r is not None and torch.isfinite(s_img_r):
                        per_layer['sum_img_rest'][li] = s_img_r.detach().float()
                    if m_lat is not None and torch.isfinite(m_lat):
                        per_layer['mean_latent'][li] = m_lat.detach().float()
                    if s_lat is not None and torch.isfinite(s_lat):
                        per_layer['sum_latent'][li] = s_lat.detach().float()
                    if m_oth is not None and torch.isfinite(m_oth):
                        per_layer['mean_other_prev'][li] = m_oth.detach().float()
                    if s_oth is not None and torch.isfinite(s_oth):
                        per_layer['sum_other_prev'][li] = s_oth.detach().float()

                # Update rolling accumulators with last layer (for quick scalar logging, similar to previous behavior)
                last_idx = len(layers) - 1
                def _nan_to_none(x: torch.Tensor):
                    return None if (x.numel()==0 or not torch.isfinite(x)) else x
                m_qimg = _nan_to_none(per_layer['mean_qimg'][last_idx])
                m_img_r = _nan_to_none(per_layer['mean_img_rest'][last_idx])
                m_lat = _nan_to_none(per_layer['mean_latent'][last_idx])
                m_oth = _nan_to_none(per_layer['mean_other_prev'][last_idx])
                s_qimg = _nan_to_none(per_layer['sum_qimg'][last_idx])
                s_img_r = _nan_to_none(per_layer['sum_img_rest'][last_idx])
                s_lat = _nan_to_none(per_layer['sum_latent'][last_idx])
                s_oth = _nan_to_none(per_layer['sum_other_prev'][last_idx])

                if m_qimg is not None:
                    self._obs_to_qimg_att_mean_cum += float(m_qimg.item())
                    self._obs_to_qimg_att_mean_steps += 1
                if s_qimg is not None:
                    self._obs_to_qimg_att_sum_cum += float(s_qimg.item())
                    self._obs_to_qimg_att_sum_steps += 1
                if m_img_r is not None:
                    self._obs_to_img_rest_att_mean_cum += float(m_img_r.item())
                    self._obs_to_img_rest_att_mean_steps += 1
                if s_img_r is not None:
                    self._obs_to_img_rest_att_sum_cum += float(s_img_r.item())
                    self._obs_to_img_rest_att_sum_steps += 1
                if m_lat is not None:
                    self._obs_to_latent_att_cum += float(m_lat.item())
                    self._obs_to_latent_att_steps += 1
                if s_lat is not None:
                    self._obs_to_latent_att_sum_cum += float(s_lat.item())
                    self._obs_to_latent_att_sum_steps += 1
                if m_oth is not None:
                    self._obs_to_other_prev_att_mean_cum += float(m_oth.item())
                    self._obs_to_other_prev_att_mean_steps += 1
                if s_oth is not None:
                    self._obs_to_other_prev_att_sum_cum += float(s_oth.item())
                    self._obs_to_other_prev_att_sum_steps += 1

                # Stash per-layer arrays for saving at log step
                self._last_attn_layer_stats = {
                    'global_step': int(getattr(self.state, 'global_step', 0) or 0),
                    'per_layer': {k: v.cpu().numpy() for k, v in per_layer.items()},
                }
                # Restore impl and training mode
                try:
                    model.config._attn_implementation = prev_attn_impl
                except Exception:
                    pass
                if was_training:
                    try:
                        model.train()
                    except Exception:
                        pass

        except Exception:
            pass


    def on_epoch_end(self):
        # 注意：HF Trainer 不会自动调用子类自定义的 on_epoch_end；实际汇总通过回调实现。
        return super().on_epoch_end()

    def log(self, logs: dict, start_time: float | None = None):
        # Merge our rolling averages into the standard logs once per logging call
        merged = dict(logs)
        if self.observation_token_acc_step > 0:
            merged["observation_token_acc"] = round(self.observation_token_acc/ max(1, self.observation_token_acc_step), 6)
            self.observation_token_acc = 0.
            self.observation_token_acc_step = 0

        if getattr(self.args, 'attn_analysis', False):
            # new metrics
            if self._obs_to_qimg_att_mean_steps > 0:
                merged["obs_to_qimg_att_mean"] = round(self._obs_to_qimg_att_mean_cum / max(1, self._obs_to_qimg_att_mean_steps), 8)
            if self._obs_to_qimg_att_sum_steps > 0:
                merged["obs_to_qimg_att_sum"] = round(self._obs_to_qimg_att_sum_cum / max(1, self._obs_to_qimg_att_sum_steps), 8)
            if self._obs_to_img_rest_att_mean_steps > 0:
                merged["obs_to_img_rest_att_mean"] = round(self._obs_to_img_rest_att_mean_cum / max(1, self._obs_to_img_rest_att_mean_steps), 8)
            if self._obs_to_img_rest_att_sum_steps > 0:
                merged["obs_to_img_rest_att_sum"] = round(self._obs_to_img_rest_att_sum_cum / max(1, self._obs_to_img_rest_att_sum_steps), 8)
            if self._obs_to_latent_att_steps > 0:
                merged["obs_to_latent_att_mean"] = round(self._obs_to_latent_att_cum / max(1, self._obs_to_latent_att_steps), 8)
            if self._obs_to_latent_att_sum_steps > 0:
                merged["obs_to_latent_att_sum"] = round(self._obs_to_latent_att_sum_cum / max(1, self._obs_to_latent_att_sum_steps), 8)
            if self._obs_to_other_prev_att_mean_steps > 0:
                merged["obs_to_other_prev_att_mean"] = round(self._obs_to_other_prev_att_mean_cum / max(1, self._obs_to_other_prev_att_mean_steps), 8)
            if self._obs_to_other_prev_att_sum_steps > 0:
                merged["obs_to_other_prev_att_sum"] = round(self._obs_to_other_prev_att_sum_cum / max(1, self._obs_to_other_prev_att_sum_steps), 8)

            # Save per-layer stats for this log step (without pushing to W&B merged)
            if self._last_attn_layer_stats is not None and isinstance(self._last_attn_layer_stats, dict):
                step = int(self._last_attn_layer_stats.get('global_step', 0) or 0)
                per_layer = self._last_attn_layer_stats.get('per_layer', {})
                try:
                    np.savez_compressed(
                        os.path.join(self._attn_save_dir, f"attn_layers_step_{step:08d}.npz"),
                        **per_layer,
                        step=step,
                    )
                except Exception:
                    pass
                # do not reset here; keep last for potential later use
            self._obs_to_img_att_cum = 0.0
            self._obs_to_img_att_steps = 0
            self._obs_to_latent_att_cum = 0.0
            self._obs_to_latent_att_steps = 0
            self._obs_to_qimg_att_mean_cum = 0.0
            self._obs_to_qimg_att_mean_steps = 0
            self._obs_to_qimg_att_sum_cum = 0.0
            self._obs_to_qimg_att_sum_steps = 0
            self._obs_to_img_rest_att_mean_cum = 0.0
            self._obs_to_img_rest_att_mean_steps = 0
            self._obs_to_img_rest_att_sum_cum = 0.0
            self._obs_to_img_rest_att_sum_steps = 0
            self._obs_to_latent_att_sum_cum = 0.0
            self._obs_to_latent_att_sum_steps = 0
            self._obs_to_other_prev_att_mean_cum = 0.0
            self._obs_to_other_prev_att_mean_steps = 0
            self._obs_to_other_prev_att_sum_cum = 0.0
            self._obs_to_other_prev_att_sum_steps = 0

        if self._emph_steps > 0:
            merged['emphasize_latent_attn_loss'] = round(self._emph_loss_cum / max(1, self._emph_steps), 8)
            self._emph_loss_cum = 0.0
            self._emph_steps = 0

        if self.args.use_align_vision_latent_loss_projector:
            if self.align_vision_latent_loss_steps and self.align_vision_latent_loss_steps > 0:
                merged['align_vision_latent_loss_projector'] = round(self.align_vision_latent_loss_cum / max(1, self.align_vision_latent_loss_steps), 8)
            self.align_vision_latent_loss_cum = 0.0
            self.align_vision_latent_loss_steps = 0

        if self.args.use_align_vision_latent_loss_pooling:
            if self.align_vision_latent_loss_steps and self.align_vision_latent_loss_steps > 0:
                merged['align_vision_latent_loss_pooling'] = round(self.align_vision_latent_loss_cum / max(1, self.align_vision_latent_loss_steps), 8)
            self.align_vision_latent_loss_cum = 0.0
            self.align_vision_latent_loss_steps = 0

        # Call parent to keep default behavior (console/TB/W&B/etc.)
        return super().log(merged, start_time)

class CustomTrainerAVT_V2_Stage2(SFTTrainer):
    def __init__(self, *args, **kwargs): 
        self.exp_name =kwargs.pop('exp_name')
        super().__init__(*args, **kwargs)
        self.weight = self.args.alignment_weight
        self.ce_emphasize_factor: float = float(getattr(self.args, 'ce_emphasize_factor', 1.0))
        # Where to read precomputed teacher latents
        base_save = getattr(self.args, 'output_dir', './checkpoints')
        self.teacher_latent_dir = getattr(self.args, 'teacher_latent_dir', None)
        if not self.teacher_latent_dir:
            # fall back to user-specified save_model_path-like; use output_dir parent by default
            self.teacher_latent_dir = os.path.join(base_save if base_save else './checkpoints', 'teacher_latents')
        # 仅 rank‑0 进程写文件，防止多卡重复
        self.is_main_process = (
            not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )

        self.observation_token_acc = 0.
        self.observation_token_acc_step = 0
        self._al_loss_cum = 0.0       # cumulative alignment loss since last log
        self._al_steps = 0            # number of micro-steps accumulated
        self.student_ce_loss_cum = 0.0        # cumulative student CE loss
        self.student_ce_loss_steps = 0



    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss for AVT v2 stage2 with optional cached teacher latents.
        """
        # Try to load precomputed teacher latents
        teacher_latents = None
        batch_metadata = inputs['metadata']

        latents_list = []
        for metadata in batch_metadata:
            dataset_name = metadata['dataset_name']
            sample_id = metadata['sample_id']
            metadata_info = f"{self.args.alignment_layer}_{dataset_name}_{sample_id}"
            path = os.path.join(self.teacher_latent_dir, f"latent_{metadata_info}.pt")
            if not os.path.isfile(path):
                latents_list = []
                raise RuntimeError(f"Missing teacher latent file: {path}")
            data = torch.load(path, map_location='cpu')
            latents_list.append(data['latent'].detach())
        if batch_metadata is not None and len(latents_list) == len(batch_metadata):
            teacher_latents = latents_list

        # Student alignment forward
        inputs['latent_mode'] = True
        inputs['input_ids'] = inputs['student_input_ids']
        inputs['attention_mask'] = inputs['student_attention_mask']
        inputs['pixel_values'] = inputs['student_pixel_values']
        inputs['image_grid_thw'] = inputs['student_image_grid_thw']
        if 'labels' in inputs:
            inputs.pop('labels')
        inputs['alignment_poss'] = inputs['student_alignment_poss']
        inputs['teacher_hidden_states_for_alignment'] = teacher_latents
        model.gradient_checkpointing_disable()
        inputs['loss_type'] = ['alignment']
        inputs['output_latent_embeds'] = False
        (_, student_outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        alignment_loss = student_outputs.loss_dict['alignment']

        # Student CE forward
        inputs['latent_mode'] = False
        inputs['labels'] = inputs['student_labels']
        inputs['ce_patch_pos'] = student_outputs.ce_patch_pos
        inputs['ce_patch_vec'] = student_outputs.ce_patch_vec
        inputs['ce_emphasize_factor'] = self.ce_emphasize_factor
        inputs['ce_emphasize_poss'] = inputs['observation_poss']
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        inputs['loss_type'] = ['ce']
        inputs['compute_emphasize_acc'] = True
        if 'student_attention_mask_4d' in inputs:
            inputs['attention_mask_4d'] = inputs.pop('student_attention_mask_4d')
        (student_ce_loss, student_outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        if getattr(student_outputs, 'mean_emphasize_acc', None) is not None:
            self.observation_token_acc += getattr(student_outputs, 'mean_emphasize_acc')
            self.observation_token_acc_step += 1
        
        alignment_loss = alignment_loss.to(student_ce_loss.device, dtype=student_ce_loss.dtype)
        if isinstance(self.weight, torch.Tensor):
            self.weight = self.weight.to(student_ce_loss.device, dtype=student_ce_loss.dtype)
        else:
            self.weight = student_ce_loss.new_tensor(float(self.weight))
        loss = student_ce_loss + self.weight * alignment_loss

        outputs_student_loss = student_ce_loss.item()

        # Periodic light GC on main process
        del student_outputs
        step = int(getattr(self.state, 'global_step', 0) or 0)
        if self.is_main_process and step > 0 and (step % 20 == 0):
            try:
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass

        # Logging
        self._al_loss_cum += float(alignment_loss.detach().item())
        self._al_steps += 1
        self.student_ce_loss_cum += outputs_student_loss
        self.student_ce_loss_steps += 1

        '''if self.is_main_process:
            with open(self.loss_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.state.global_step,
                    self.state.epoch,
                    float(loss.detach().item()),
                    outputs_student_loss,
                    float(alignment_loss.detach().item()),
                ])'''
        return (loss, None) if return_outputs else loss
    
    def log(self, logs: dict, start_time: float | None = None):
        # Merge our rolling averages into the standard logs once per logging call
        merged = dict(logs)
        if self._al_steps > 0:
            merged["alignment_loss"] = round(self._al_loss_cum / max(1, self._al_steps), 6)
            self._al_loss_cum = 0.0
            self._al_steps = 0
        if self.student_ce_loss_steps > 0:
            merged["student_ce_loss"] = round(self.student_ce_loss_cum / max(1, self.student_ce_loss_steps), 6)
            self.student_ce_loss_cum = 0.0
            self.student_ce_loss_steps = 0
        if self.observation_token_acc_step > 0:
            merged["observation_token_acc"] = round(self.observation_token_acc/ max(1, self.observation_token_acc_step), 6)
            self.observation_token_acc = 0.
            self.observation_token_acc_step = 0

        # Call parent to keep default behavior (console/TB/W&B/etc.)
        return super().log(merged, start_time)

class CustomTrainerAVT_V3(SFTTrainer):
    def __init__(self, *args, **kwargs): 
        self.exp_name =kwargs.pop('exp_name')
        super().__init__(*args, **kwargs)
        self.weight = self.args.alignment_weight
        self.ce_emphasize_factor: float = float(getattr(self.args, 'ce_emphasize_factor', 1.0))
        # Where to read precomputed teacher latents
        base_save = getattr(self.args, 'output_dir', './checkpoints')
        self.teacher_latent_dir = getattr(self.args, 'teacher_latent_dir', None)
        if not self.teacher_latent_dir:
            # fall back to user-specified save_model_path-like; use output_dir parent by default
            self.teacher_latent_dir = os.path.join(base_save if base_save else './checkpoints', 'teacher_latents')
        # 仅 rank‑0 进程写文件，防止多卡重复
        self.is_main_process = (
            not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )
        self.align_loss_type = "pooling" if self.args.use_align_vision_latent_loss_pooling else "projector"
        self.observation_token_acc = 0.
        self.observation_token_acc_step = 0
        self.align_vision_latent_loss_cum = 0.
        self.align_vision_latent_loss_steps = 0
        self.student_ce_loss_cum = 0.0        # cumulative student CE loss
        self.student_ce_loss_steps = 0



    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Prepare teacher forward inputs (for latent extraction)
        inputs['latent_mode'] = False
        inputs['input_ids'] = inputs['teacher_input_ids']
        inputs['attention_mask'] = inputs['teacher_attention_mask']
        inputs['pixel_values'] = inputs['teacher_pixel_values']
        inputs['image_grid_thw'] = inputs['teacher_image_grid_thw']
        inputs['labels'] = None
        inputs['alignment_poss'] = inputs['teacher_alignment_poss']
        model.gradient_checkpointing_disable()
        inputs['latent_size'] = self.args.latent_size
        inputs['loss_type'] = [f'align_vision_latent_{self.align_loss_type}']
        inputs['segs'] = inputs['teacher_segs']
        inputs['output_helper_img_embeds'] = True
        with torch.no_grad():
            teacher_outputs = model(**inputs)

        # Student alignment forward
        inputs['latent_mode'] = True
        inputs['input_ids'] = inputs['student_input_ids']
        inputs['attention_mask'] = inputs['student_attention_mask']
        inputs['pixel_values'] = inputs['student_pixel_values']
        inputs['image_grid_thw'] = inputs['student_image_grid_thw']
        if 'labels' in inputs:
            inputs.pop('labels')
        model.gradient_checkpointing_disable()
        inputs['output_latent_embeds'] = False
        inputs['output_helper_img_embeds'] = False
        inputs['segs'] = None
        inputs['loss_type'] = []
        student_outputs_latent = model(**inputs)

        # Student CE forward
        inputs['latent_mode'] = False
        inputs['labels'] = inputs['student_labels']
        inputs['align_vision_latent_pre_result'] = teacher_outputs.align_vision_latent_pre_result
        inputs['ce_patch_pos'] = student_outputs_latent.ce_patch_pos
        inputs['ce_patch_vec'] = student_outputs_latent.ce_patch_vec
        inputs['ce_emphasize_factor'] = self.ce_emphasize_factor
        inputs['ce_emphasize_poss'] = inputs['observation_poss']
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        inputs['loss_type'] = ['ce']
        inputs['loss_type'].append(f'align_vision_latent_{self.align_loss_type}')

        inputs['compute_emphasize_acc'] = True
        if 'student_attention_mask_4d' in inputs:
            inputs['attention_mask_4d'] = inputs.pop('student_attention_mask_4d')
        (student_ce_loss, student_outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        if getattr(student_outputs, 'mean_emphasize_acc', None) is not None:
            self.observation_token_acc += getattr(student_outputs, 'mean_emphasize_acc')
            self.observation_token_acc_step += 1
        
        align_vision_latent_loss = student_outputs.loss_dict[f'align_vision_latent_{self.align_loss_type}']
        original_loss = student_ce_loss + self.args.align_vision_latent_loss_weight * align_vision_latent_loss
        outputs_student_loss = student_ce_loss.item()

        if self.args.emphasize_latent_weight != 1.0:
            latent_only_loss = compute_latents_only_loss(student_outputs_latent.ce_patch_vec, original_loss)
            loss = self.args.emphasize_latent_weight * latent_only_loss + 1.0 * original_loss
        else:
            loss = original_loss


        # Periodic light GC on main process
        del student_outputs, teacher_outputs
        step = int(getattr(self.state, 'global_step', 0) or 0)
        if self.is_main_process and step > 0 and (step % 20 == 0):
            try:
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass

        # Logging
        self.student_ce_loss_cum += outputs_student_loss
        self.student_ce_loss_steps += 1
        self.align_vision_latent_loss_cum += align_vision_latent_loss.item()
        self.align_vision_latent_loss_steps += 1

        return (loss, None) if return_outputs else loss
    


    def log(self, logs: dict, start_time: float | None = None):
        # Merge our rolling averages into the standard logs once per logging call
        merged = dict(logs)
        if self.align_vision_latent_loss_steps and self.align_vision_latent_loss_steps > 0:
            merged[f'align_vision_latent_loss_{self.align_loss_type}'] = round(self.align_vision_latent_loss_cum / max(1, self.align_vision_latent_loss_steps), 8)
            self.align_vision_latent_loss_cum = 0.0
            self.align_vision_latent_loss_steps = 0
        if self.student_ce_loss_steps > 0:
            merged["student_ce_loss"] = round(self.student_ce_loss_cum / max(1, self.student_ce_loss_steps), 6)
            self.student_ce_loss_cum = 0.0
            self.student_ce_loss_steps = 0
        if self.observation_token_acc_step > 0:
            merged["observation_token_acc"] = round(self.observation_token_acc/ max(1, self.observation_token_acc_step), 6)
            self.observation_token_acc = 0.
            self.observation_token_acc_step = 0

        # Call parent to keep default behavior (console/TB/W&B/etc.)
        return super().log(merged, start_time)

class CustomTrainerAVT_V3_1(SFTTrainer):
    def __init__(self, *args, **kwargs): 
        self.exp_name =kwargs.pop('exp_name')
        super().__init__(*args, **kwargs)
        self.weight = self.args.alignment_weight
        self.ce_emphasize_factor: float = float(getattr(self.args, 'ce_emphasize_factor', 1.0))
        # Where to read precomputed teacher latents
        base_save = getattr(self.args, 'output_dir', './checkpoints')
        self.teacher_latent_dir = getattr(self.args, 'teacher_latent_dir', None)
        if not self.teacher_latent_dir:
            # fall back to user-specified save_model_path-like; use output_dir parent by default
            self.teacher_latent_dir = os.path.join(base_save if base_save else './checkpoints', 'teacher_latents')
        # 仅 rank‑0 进程写文件，防止多卡重复
        self.is_main_process = (
            not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )
        self.align_loss_type = "pooling" if self.args.use_align_vision_latent_loss_pooling else "projector"
        self.observation_token_acc = 0.
        self.observation_token_acc_step = 0
        self.align_vision_latent_loss_cum = 0.
        self.align_vision_latent_loss_steps = 0
        self.student_ce_loss_cum = 0.0        # cumulative student CE loss
        self.student_ce_loss_steps = 0


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # -----------------------------
        # 1) Teacher forward (no-grad)
        # -----------------------------
        inputs['latent_mode'] = False
        inputs['input_ids'] = inputs['teacher_input_ids']
        inputs['attention_mask'] = inputs['teacher_attention_mask']
        inputs['pixel_values'] = inputs['teacher_pixel_values']
        inputs['image_grid_thw'] = inputs['teacher_image_grid_thw']
        inputs['labels'] = None
        inputs['alignment_poss'] = inputs['teacher_alignment_poss']
        model.gradient_checkpointing_disable()
        inputs['latent_size'] = self.args.latent_size
        inputs['loss_type'] = [f'align_vision_latent_{self.align_loss_type}']
        inputs['segs'] = inputs['teacher_segs']
        inputs['output_helper_img_embeds'] = True
        with torch.no_grad():
            teacher_outputs = model(**inputs)

        # -----------------------------
        # 2) Student latent forward
        # -----------------------------
        inputs['latent_mode'] = True
        inputs['input_ids'] = inputs['student_input_ids']
        inputs['attention_mask'] = inputs['student_attention_mask']
        inputs['pixel_values'] = inputs['student_pixel_values']
        inputs['image_grid_thw'] = inputs['student_image_grid_thw']
        if 'labels' in inputs:
            inputs.pop('labels')
        model.gradient_checkpointing_disable()
        inputs['output_latent_embeds'] = False
        inputs['output_helper_img_embeds'] = False
        inputs['segs'] = None
        inputs['loss_type'] = []
        student_outputs_latent = model(**inputs)

        # Make sure ce_patch_vec requires grad and is the SAME tensor used below
        # (Usually it already requires grad; retain_grad() is useful for debugging.)
        for v in student_outputs_latent.ce_patch_vec:
                if hasattr(v, "retain_grad"):
                    v.retain_grad()
        # -----------------------------
        # 3) Student CE forward (for building L3 only)
        # -----------------------------
        inputs['latent_mode'] = False
        inputs['labels'] = inputs['student_labels']
        inputs['align_vision_latent_pre_result'] = teacher_outputs.align_vision_latent_pre_result
        inputs['ce_patch_pos'] = student_outputs_latent.ce_patch_pos
        inputs['ce_patch_vec'] = student_outputs_latent.ce_patch_vec
        inputs['ce_emphasize_factor'] = self.ce_emphasize_factor
        inputs['ce_emphasize_poss'] = inputs['observation_poss']
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        inputs['loss_type'] = ['ce']
        inputs['loss_type'].append(f'align_vision_latent_{self.align_loss_type}')
        inputs['compute_emphasize_acc'] = True
        if 'student_attention_mask_4d' in inputs:
            inputs['attention_mask_4d'] = inputs.pop('student_attention_mask_4d')

        # Call the parent to get the CE/align losses computed by the model (3rd forward)
        student_ce_loss, student_outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        if getattr(student_outputs, 'mean_emphasize_acc', None) is not None:
            self.observation_token_acc += getattr(student_outputs, 'mean_emphasize_acc')
            self.observation_token_acc_step += 1

        align_vision_latent_loss = student_outputs.loss_dict[f'align_vision_latent_{self.align_loss_type}']

        # === (A) Build the conceptual loss L3 on the 3rd forward graph ===
        L3 = student_ce_loss + self.args.align_vision_latent_loss_weight * align_vision_latent_loss

        # === (B) Take grads of L3 w.r.t. ce_patch_vec ONLY ===
        # This collects dL3/d(ce_patch_vec) without accumulating parameter grads from the 3rd forward.
        def _flatten_tensors(x):
            # Flatten nested [list/tuple of Tensors] into a flat list of Tensors
            if isinstance(x, (list, tuple)):
                out = []
                for y in x:
                    out.extend(_flatten_tensors(y))
                return out
            return [x]

        ce_vec_list = _flatten_tensors(student_outputs_latent.ce_patch_vec)
        grads = torch.autograd.grad(
            outputs=L3,
            inputs=ce_vec_list,
            retain_graph=False,   # we won't reuse the 3rd graph
            create_graph=False,   # stop higher-order graph
            allow_unused=True     # in case some ce vectors are not used
        )

        # Replace None with zeros for unused elements
        safe_grads = []
        for v, g in zip(ce_vec_list, grads):
            if g is None:
                # Create a zero tensor on the same device/dtype/shape
                g = torch.zeros_like(v)
            safe_grads.append(g.detach())  # detach to stop any 3rd-forward param path

        # === (C) Build a proxy loss that only depends on ce_patch_vec (2nd forward graph) ===
        # grad(x^T g) = g, so backprop on this proxy loss will push exactly safe_grads into ce_patch_vec,
        # and then flow back ONLY through the 2nd forward graph/parameters.
        proxy_loss = torch.stack([(v * g).sum() for v, g in zip(ce_vec_list, safe_grads)]).sum()

        # ---------- Logging (keep your original meters) ----------
        outputs_student_loss = student_ce_loss.item()
        self.student_ce_loss_cum += outputs_student_loss
        self.student_ce_loss_steps += 1
        self.align_vision_latent_loss_cum += align_vision_latent_loss.item()
        self.align_vision_latent_loss_steps += 1

        # Periodic light GC on main process
        del student_outputs, teacher_outputs
        step = int(getattr(self.state, 'global_step', 0) or 0)
        if self.is_main_process and step > 0 and (step % 20 == 0):
            try:
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass

        # === Return ONLY the proxy loss ===
        return (proxy_loss, None) if return_outputs else proxy_loss


    def log(self, logs: dict, start_time: float | None = None):
        # Merge our rolling averages into the standard logs once per logging call
        merged = dict(logs)
        if self.align_vision_latent_loss_steps and self.align_vision_latent_loss_steps > 0:
            merged[f'align_vision_latent_loss_{self.align_loss_type}'] = round(self.align_vision_latent_loss_cum / max(1, self.align_vision_latent_loss_steps), 8)
            self.align_vision_latent_loss_cum = 0.0
            self.align_vision_latent_loss_steps = 0
        if self.student_ce_loss_steps > 0:
            merged["student_ce_loss"] = round(self.student_ce_loss_cum / max(1, self.student_ce_loss_steps), 6)
            self.student_ce_loss_cum = 0.0
            self.student_ce_loss_steps = 0
        if self.observation_token_acc_step > 0:
            merged["observation_token_acc"] = round(self.observation_token_acc/ max(1, self.observation_token_acc_step), 6)
            self.observation_token_acc = 0.
            self.observation_token_acc_step = 0

        # Call parent to keep default behavior (console/TB/W&B/etc.)
        return super().log(merged, start_time)

class CustomTrainerAVT_V4(SFTTrainer):
    def __init__(self, *args, **kwargs): 
        self.exp_name =kwargs.pop('exp_name')
        super().__init__(*args, **kwargs)
        self.alignment_weight = self.args.alignment_weight
        self.no_ce = self.args.no_ce
        self.ce_emphasize_factor: float = float(getattr(self.args, 'ce_emphasize_factor', 1.0))
        # Where to read precomputed teacher latents
        base_save = getattr(self.args, 'output_dir', './checkpoints')
        self.teacher_reps_dir = getattr(self.args, 'teacher_reps_dir', None)
        # 仅 rank‑0 进程写文件，防止多卡重复
        self.is_main_process = (
            not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )
        
        self.observation_token_acc = 0.
        self.observation_token_acc_step = 0
        self.alignment_loss_cum = 0.
        self.alignment_loss_steps = 0
        self.student_ce_loss_cum = 0.0        # cumulative student CE loss
        self.student_ce_loss_steps = 0
        self.teacher_ce_loss_cum = 0.0        # cumulative teacher CE loss
        self.teacher_ce_loss_steps = 0


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Prepare teacher forward inputs (for latent extraction)
        if self.teacher_reps_dir is None:
            inputs['latent_mode'] = False
            inputs['input_ids'] = inputs['teacher_input_ids']
            inputs['attention_mask'] = inputs['teacher_attention_mask']
            inputs['pixel_values'] = inputs['teacher_pixel_values']
            inputs['image_grid_thw'] = inputs['teacher_image_grid_thw']
            inputs['labels'] = None #inputs['labels'] = inputs['teacher_labels']
            inputs['alignment_poss'] = inputs['teacher_observation_poss']
            model.gradient_checkpointing_disable()
            inputs['latent_size'] = self.args.latent_size
            #inputs['loss_type'] = ['ce']
            inputs['output_hidden_states'] = True
            inputs['ce_emphasize_factor'] = self.ce_emphasize_factor
            inputs['ce_emphasize_poss'] = inputs['teacher_observation_poss']
            with torch.no_grad():
                teacher_outputs = model(**inputs)
                teacher_reps = teacher_outputs.hidden_states
            #teacher_ce_loss = teacher_outputs.loss_dict.get('ce', None)
            teacher_ce_loss = None
        else:       
            # Try to load precomputed teacher latents
            teacher_reps = None
            batch_metadata = inputs['metadata']
            teacher_ce_loss = None
            latents_list = []
            for metadata in batch_metadata:
                dataset_name = metadata['dataset_name']
                sample_id = metadata['sample_id']
                metadata_info = f"{self.args.alignment_layer}_{dataset_name}_{sample_id}"
                path = os.path.join(self.teacher_reps_dir, f"rep_{metadata_info}.pt")
                if not os.path.isfile(path):
                    latents_list = []
                    raise RuntimeError(f"Missing teacher latent file: {path}")
                data = torch.load(path, map_location='cpu')
                latents_list.append(data['latent'].detach())
            if batch_metadata is not None and len(latents_list) == len(batch_metadata):
                teacher_reps = latents_list



        # Student alignment forward
        inputs['latent_mode'] = True
        inputs['input_ids'] = inputs['student_input_ids']
        inputs['attention_mask'] = inputs['student_attention_mask']
        inputs['pixel_values'] = inputs['student_pixel_values']
        inputs['image_grid_thw'] = inputs['student_image_grid_thw']
        inputs['teacher_hidden_states_for_alignment'] = teacher_reps
        inputs['alignment_poss'] = inputs['student_observation_poss']
        inputs['output_hidden_states'] = False
        if 'labels' in inputs:
            inputs.pop('labels')
        model.gradient_checkpointing_disable()
        inputs['loss_type'] = []
        student_outputs_latent = model(**inputs)
        

        # Student CE forward
        inputs['latent_mode'] = False
        inputs['labels'] = inputs['student_labels']
        inputs['ce_patch_pos'] = student_outputs_latent.ce_patch_pos
        inputs['ce_patch_vec'] = student_outputs_latent.ce_patch_vec
        inputs['ce_emphasize_factor'] = self.ce_emphasize_factor
        inputs['ce_emphasize_poss'] = inputs['student_observation_poss']
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        inputs['loss_type'] = ['alignment']
        if not self.no_ce:
            inputs['loss_type'].append('ce')

        inputs['compute_emphasize_acc'] = True
        if 'student_attention_mask_4d' in inputs:
            inputs['attention_mask_4d'] = inputs.pop('student_attention_mask_4d')
        (student_ce_loss, student_outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        alignment_loss = student_outputs.loss_dict['alignment']

        if getattr(student_outputs, 'mean_emphasize_acc', None) is not None:
            self.observation_token_acc += getattr(student_outputs, 'mean_emphasize_acc')
            self.observation_token_acc_step += 1

        if not self.no_ce:
            record_student_loss = student_ce_loss.item()
            if teacher_ce_loss is not None:
                record_teacher_loss = teacher_ce_loss.item()
            else:
                record_teacher_loss = 0.0
        else:
            student_ce_loss = 0.0
            record_student_loss = 0.0
            record_teacher_loss = 0.0

        if self.args.emphasize_latent_weight != 1.0:
            latent_only_loss = compute_latents_only_loss(student_outputs_latent.ce_patch_vec, self.alignment_weight *alignment_loss)
            loss = self.args.emphasize_latent_weight * latent_only_loss + student_ce_loss
        else:
            loss = student_ce_loss + self.alignment_weight * alignment_loss

        if teacher_ce_loss is not None:
            loss += 1.0 * teacher_ce_loss

        # Periodic light GC on main process
        del student_outputs
        step = int(getattr(self.state, 'global_step', 0) or 0)
        if self.is_main_process and step > 0 and (step % 20 == 0):
            try:
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass

        # Logging
        self.student_ce_loss_cum += record_student_loss
        self.student_ce_loss_steps += 1
        self.alignment_loss_cum += alignment_loss.item()
        self.alignment_loss_steps += 1
        if teacher_ce_loss is not None:
            self.teacher_ce_loss_cum += record_teacher_loss
            self.teacher_ce_loss_steps += 1

        return (loss, None) if return_outputs else loss
    


    def log(self, logs: dict, start_time: float | None = None):
        # Merge our rolling averages into the standard logs once per logging call
        merged = dict(logs)
        if self.student_ce_loss_cum > 0:
            merged["student_ce_loss"] = round(self.student_ce_loss_cum / max(1, self.student_ce_loss_steps), 6)
            self.student_ce_loss_cum = 0.0
            self.student_ce_loss_steps = 0
        if self.teacher_ce_loss_cum > 0:
            merged["teacher_ce_loss"] = round(self.teacher_ce_loss_cum / max(1, self.teacher_ce_loss_steps), 6)
            self.teacher_ce_loss_cum = 0.0
            self.teacher_ce_loss_steps = 0
        if self.alignment_loss_cum > 0:
            merged[f'alignment_loss'] = round(self.alignment_loss_cum / max(1, self.alignment_loss_steps), 6)
            self.alignment_loss_cum = 0.0
            self.alignment_loss_steps = 0
        if self.observation_token_acc_step > 0:
            merged["observation_token_acc"] = round(self.observation_token_acc/ max(1, self.observation_token_acc_step), 6)
            self.observation_token_acc = 0.
            self.observation_token_acc_step = 0


        # Call parent to keep default behavior (console/TB/W&B/etc.)
        return super().log(merged, start_time)



import weakref

class RepSummaryCallback(TrainerCallback):
    """Epoch 结束时汇总表示相似度。使用 weakref 保存 trainer 引用，避免循环引用。"""
    def __init__(self, trainer):
        self._trainer_ref = weakref.ref(trainer)

    def on_epoch_end(self, args, state, control, **kwargs):
        trainer = self._trainer_ref()
        if trainer is None:
            return control
        analyzer = getattr(trainer, 'rep_analyzer', None)
        if analyzer is not None:
            ep = int(state.epoch)-1 if state.epoch is not None else 0
            # Gather samples from all ranks and merge before finalizing
            local_samples = []
            if ep in analyzer.per_epoch_records:
                local_samples = analyzer.per_epoch_records[ep].get('samples', [])

            is_dist = hasattr(torch, 'distributed') and torch.distributed.is_available() and torch.distributed.is_initialized()
            if is_dist:
                rank = torch.distributed.get_rank()
                try:
                    world_size = torch.distributed.get_world_size()
                    gathered = [None] * world_size
                    # collect python objects (list of sample dicts) from all ranks
                    torch.distributed.all_gather_object(gathered, local_samples)
                    merged_samples = []
                    for s in gathered:
                        if s:
                            merged_samples.extend(s)
                    #print(f"rank {rank}, merged_samples {merged_samples}")
                    analyzer.per_epoch_records[ep] = {'samples': merged_samples}
                except Exception as e:
                    logging.warning(f"[SFT Analysis] all_gather_object failed on epoch {ep}: {e}; fallback to rank-local summary.")

            # Only main process writes file, but ensure it has merged samples
            if (not is_dist) or torch.distributed.get_rank() == 0:
                analyzer.finalize_epoch(ep)
                logging.info(f"[SFT Analysis] Epoch {ep} summary written.")
        return control