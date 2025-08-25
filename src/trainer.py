from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback
import logging
import torch
import os, csv, torch, datetime
import gc
import numpy as np
from .utils import SFTRepAnalyzer
import math
from time import time

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
        self._stu_ce_cum = 0.0        # cumulative student CE loss
        self._stu_ce_steps = 0
                
    def alignment_loss(self, student_reps_all_layers, teacher_reps_all_layers):
        total_loss = 0.
        layerwise_sim_record = []
        bsz = len(student_reps_all_layers[0])
        for student_reps_l, teacher_reps_l in zip(student_reps_all_layers, teacher_reps_all_layers):
            layer_loss = 0.
            layerwise_sim = 0.
            for student_rep_l_b, teacher_rep_l_b in zip(student_reps_l, teacher_reps_l):
                if student_rep_l_b.shape[0] == 0 or teacher_rep_l_b.shape[0] == 0:
                    continue
                # 避免零范数导致 NaN，增加 eps
                sim = torch.nn.functional.cosine_similarity(student_rep_l_b, teacher_rep_l_b, eps=1e-6).mean()
                layer_loss += 1 - sim
                layerwise_sim += sim.item()
            total_loss += layer_loss/ bsz
            layerwise_sim_record.append(layerwise_sim / bsz)
        total_loss = total_loss / len(student_reps_all_layers)
        '''if torch.isnan(total_loss):
            #print("student_reps_all_layers =", student_reps_all_layers)
            #print("teacher_reps_all_layers =", teacher_reps_all_layers)
            print("student_poss =",student_poss)
            print("teacher_poss =",teacher_poss)'''
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
        self._stu_ce_cum += float(student_ce_loss.detach().item())
        self._stu_ce_steps += 1

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
        if self._stu_ce_steps > 0:
            merged["student_ce_loss"] = round(self._stu_ce_cum / max(1, self._stu_ce_steps), 6)

        # Reset accumulators after logging so the next window starts fresh
        self._al_loss_cum = 0.0
        self._al_steps = 0
        self._stu_ce_cum = 0.0
        self._stu_ce_steps = 0

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

        # Helper: compute total training steps if needed (may be updated later, so compute on demand as well)
        def _estimate_total_steps() -> int:
            # Prefer TrainerState.max_steps if available and > 0
            try:
                if hasattr(self.state, 'max_steps') and self.state.max_steps and self.state.max_steps > 0:
                    return int(self.state.max_steps)
            except Exception:
                pass
            # Fallback to args.max_steps if provided (>0)
            try:
                if hasattr(self.args, 'max_steps') and self.args.max_steps and self.args.max_steps > 0:
                    return int(self.args.max_steps)
            except Exception:
                pass
            return 0

        self._ce_emphasize_total_steps_hint = _estimate_total_steps()
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
                    "loss_teacher_ce"
                ])

        

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
        inputs['pixel_values'] = inputs['user_assistant_pixel_values']
        inputs['image_grid_thw'] = inputs['user_assistant_image_grid_thw']
        inputs['labels'] = inputs['teacher_labels']
        # Collect available position categories dynamically (supports non_observation_poss)
        poss_dict = {}
        for k in ['boxed_start_poss','observation_poss','non_observation_poss']:
            if k in inputs:
                poss_dict[k] = inputs[k]
        inputs['ce_emphasize_poss'] = inputs['observation_poss']
        # Dynamic warmup factor passed to model.forward
        inputs['ce_emphasize_factor'] = self._current_ce_emphasize_factor()
        inputs['loss_type'] = ['ce']
        (teacher_ce_loss, teacher_outputs) = super().compute_loss(
                model, 
                inputs,
                return_outputs=True, num_items_in_batch=num_items_in_batch
            )

        

        # Representation analysis update
        if self.rep_analyzer is not None and 'sample_id' in inputs:
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
                            )

        outputs_teacher_loss = teacher_ce_loss.item()

        del teacher_outputs
        gc.collect()
        torch.cuda.empty_cache()
        
        # --------  写本地文件  --------
        if self.is_main_process:
            with open(self.loss_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.state.global_step,
                    self.state.epoch,
                    outputs_teacher_loss
                ])
        # --------------------------------------------
        
        
        return (teacher_ce_loss, None) if return_outputs else teacher_ce_loss

    def on_epoch_end(self):
        # 注意：HF Trainer 不会自动调用子类自定义的 on_epoch_end；实际汇总通过回调实现。
        return super().on_epoch_end()

class CustomTrainerAVT_V2_Stage1(SFTTrainer):
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

        # Helper: compute total training steps if needed (may be updated later, so compute on demand as well)
        def _estimate_total_steps() -> int:
            # Prefer TrainerState.max_steps if available and > 0
            try:
                if hasattr(self.state, 'max_steps') and self.state.max_steps and self.state.max_steps > 0:
                    return int(self.state.max_steps)
            except Exception:
                pass
            # Fallback to args.max_steps if provided (>0)
            try:
                if hasattr(self.args, 'max_steps') and self.args.max_steps and self.args.max_steps > 0:
                    return int(self.args.max_steps)
            except Exception:
                pass
            return 0

        self._ce_emphasize_total_steps_hint = _estimate_total_steps()
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

        '''# 日志文件路径
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
                    "loss_teacher_ce"
                ])'''

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
        '''if self.is_main_process:
            torch.cuda.synchronize()
            start_time = time()'''
        inputs['stage'] = 'avt_v2_stage1'
        inputs['latent_mode'] = True
        inputs['loss_type'] = []
        #inputs['enable_ce_checkpoint'] = False
        outputs = model(**inputs, return_dict=True, output_hidden_states=False)
        '''if self.is_main_process:
            time_1 = time()
            print(f"latent forward time {time_1 - start_time}")'''
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        #inputs['enable_ce_checkpoint'] = True
        inputs['latent_mode'] = False
        inputs['ce_patch_pos'] = outputs.ce_patch_pos
        inputs['ce_patch_vec'] = outputs.ce_patch_vec
        inputs['ce_emphasize_poss'] = inputs['observation_poss']
        # Dynamic warmup factor passed to model.forward
        inputs['ce_emphasize_factor'] = self._current_ce_emphasize_factor()
        inputs['loss_type'] = ['ce']
        '''(teacher_ce_loss, teacher_outputs) = super().compute_loss(
                model, 
                inputs,
                return_outputs=True, num_items_in_batch=num_items_in_batch
            )'''
        teacher_ce_loss = super().compute_loss(
                model, 
                inputs,
                return_outputs=False, num_items_in_batch=num_items_in_batch
            )
        '''if self.is_main_process:
            torch.cuda.synchronize()
            time_2 = time()
            print(f"ce forward time {time_2 - time_1}")'''
        #outputs_teacher_loss = teacher_ce_loss.item()

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
        
        '''# --------  写本地文件  --------
        if self.is_main_process:
            with open(self.loss_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.state.global_step,
                    self.state.epoch,
                    outputs_teacher_loss
                ])
        # --------------------------------------------'''

        return (teacher_ce_loss, None) if return_outputs else teacher_ce_loss

    def on_epoch_end(self):
        # 注意：HF Trainer 不会自动调用子类自定义的 on_epoch_end；实际汇总通过回调实现。
        return super().on_epoch_end()

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

        # 日志文件路径
        '''log_dir = self.args.logging_dir or "./logs"
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
                ])'''

        self._al_loss_cum = 0.0       # cumulative alignment loss since last log
        self._al_steps = 0            # number of micro-steps accumulated
        self._stu_ce_cum = 0.0        # cumulative student CE loss
        self._stu_ce_steps = 0
                
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss for AVT v2 stage2 with optional cached teacher latents.
        """
        # Prepare teacher forward inputs (for latent extraction)
        inputs['stage'] = 'avt_v2_stage2'
        inputs['latent_mode'] = True
        inputs['input_ids'] = inputs['teacher_input_ids']
        inputs['attention_mask'] = inputs['teacher_attention_mask']
        inputs['pixel_values'] = inputs['teacher_pixel_values']
        inputs['image_grid_thw'] = inputs['teacher_image_grid_thw']
        inputs['labels'] = None
        inputs['alignment_poss'] = inputs['teacher_alignment_poss']
        model.gradient_checkpointing_disable()
        inputs['loss_type'] = []
        inputs['output_latent_embeds'] = True

        # Try to load precomputed teacher latents
        teacher_outputs = None
        use_cached = False
        batch_metadata = inputs.get('metadata', None)
        if batch_metadata is not None and self.teacher_latent_dir and os.path.isdir(self.teacher_latent_dir):
            try:
                latents_list = []
                for metadata in batch_metadata:
                    dataset_name = metadata['dataset_name']
                    sample_id = metadata['sample_id']
                    metadata_info = f"{dataset_name}_{sample_id}"
                    path = os.path.join(self.teacher_latent_dir, f"latent_{metadata_info}.pt")
                    if not os.path.isfile(path):
                        latents_list = []
                        break
                    data = torch.load(path, map_location='cpu')
                    latents_list.append(data['latent'])
                if batch_metadata is not None and len(latents_list) == len(batch_metadata):
                    latents = torch.stack([t if t.dim() == 2 else t.squeeze(0) for t in latents_list], dim=0).to(model.device)
                    class _Obj:
                        pass
                    teacher_outputs = _Obj()
                    teacher_outputs.latent_embeds = latents
                    use_cached = True
            except Exception:
                use_cached = False
                teacher_outputs = None

        # Fallback to online teacher forward
        if not use_cached:
            with torch.no_grad():
                teacher_outputs = model(**inputs, return_dict=True, output_hidden_states=True)

        # Student alignment forward
        inputs['latent_mode'] = True
        inputs['input_ids'] = inputs['student_input_ids']
        inputs['attention_mask'] = inputs['student_attention_mask']
        inputs['pixel_values'] = inputs['student_pixel_values']
        inputs['image_grid_thw'] = inputs['student_image_grid_thw']
        if 'labels' in inputs:
            inputs.pop('labels')
        inputs['alignment_poss'] = inputs['student_alignment_poss']
        inputs['teacher_hidden_states_for_alignment'] = teacher_outputs.latent_embeds
        model.gradient_checkpointing_disable()
        inputs['loss_type'] = ['alignment']
        inputs['output_latent_embeds'] = False
        (alignment_loss, student_outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        # Student CE forward
        inputs['latent_mode'] = False
        inputs['labels'] = inputs['student_labels']
        inputs['ce_patch_pos'] = student_outputs.ce_patch_pos
        inputs['ce_patch_vec'] = student_outputs.ce_patch_vec
        inputs['ce_emphasize_factor'] = self.ce_emphasize_factor
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        inputs['loss_type'] = ['ce']
        (student_ce_loss, student_outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        
        alignment_loss = alignment_loss.to(student_ce_loss.device, dtype=student_ce_loss.dtype)
        if isinstance(self.weight, torch.Tensor):
            self.weight = self.weight.to(student_ce_loss.device, dtype=student_ce_loss.dtype)
        else:
            self.weight = student_ce_loss.new_tensor(float(self.weight))
        loss = student_ce_loss + self.weight * alignment_loss

        outputs_student_loss = student_ce_loss.item()

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
        self._al_loss_cum += float(alignment_loss.detach().item())
        self._al_steps += 1
        self._stu_ce_cum += outputs_student_loss
        self._stu_ce_steps += 1

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
        if self._stu_ce_steps > 0:
            merged["student_ce_loss"] = round(self._stu_ce_cum / max(1, self._stu_ce_steps), 6)

        # Reset accumulators after logging so the next window starts fresh
        self._al_loss_cum = 0.0
        self._al_steps = 0
        self._stu_ce_cum = 0.0
        self._stu_ce_steps = 0

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