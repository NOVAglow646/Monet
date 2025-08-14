from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback
import logging
import torch
import os, csv, torch, datetime
import gc
import numpy as np
from .utils import SFTRepAnalyzer
import math

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
        self.weight = 1.0
        # 仅 rank‑0 进程写文件，防止多卡重复
        self.is_main_process = (
            not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )

        # 日志文件路径
        log_dir = self.args.logging_dir or "./logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        self.loss_log_path = os.path.join(log_dir, f"loss_history_w{self.weight}_{self.exp_name}_{timestamp}.csv")

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
        (_, teacher_outputs) = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
            
        inputs['latent_mode'] = True
        inputs['input_ids'] = inputs['student_input_ids']
        inputs['attention_mask'] = inputs['student_attention_mask']
        inputs['pixel_values'] = inputs['user_pixel_values']
        inputs['image_grid_thw'] = inputs['user_image_grid_thw']
        inputs['labels'] = inputs['student_labels']
        inputs['alignment_poss'] = inputs['student_alignment_poss']
        inputs['image_out_mask'] = inputs['student_image_out_mask']
        inputs['teacher_hidden_states_for_alignment'] = teacher_outputs.hidden_states

        (alignment_loss, student_outputs) = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
 
        inputs['latent_mode'] = False
        inputs['ce_patch_pos'] = student_outputs.ce_patch_pos
        inputs['ce_patch_vec'] = student_outputs.ce_patch_vec

        (student_ce_loss, student_outputs) = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        loss = student_ce_loss + self.weight *alignment_loss

        outputs_student_loss = student_ce_loss.item()

        del student_outputs, teacher_outputs
        gc.collect()
        torch.cuda.empty_cache()
        
        # --------  写本地文件  --------
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
    
    
    
    
class CustomTrainerSFT(SFTTrainer):
    def __init__(self, *args, **kwargs):
        self.exp_name =kwargs.pop('exp_name')
        # accept processing_class (preferred) and fall back to tokenizer for backward compat
        if 'processing_class' not in kwargs and 'tokenizer' in kwargs:
            kwargs['processing_class'] = kwargs.pop('tokenizer')
        super().__init__(*args, **kwargs)
        self.weight = 1.0
        # observation_ce_factor warmup configuration
        # Target factor (backward compatible with existing arg name)
        self._obs_ce_target: float = float(getattr(self.args, 'observation_ce_factor', 1.0))
        # Optional absolute warmup steps takes precedence over ratio
        self._obs_ce_warmup_steps: int = int(getattr(self.args, 'observation_ce_warmup_steps', 0) or 0)

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

        self._obs_ce_total_steps_hint = _estimate_total_steps()
        # Representation analysis
        self.rep_analyzer = None
        args_cfg = self.args
        if getattr(args_cfg, 'sft_analysis_enable', False):
            self.rep_analyzer = SFTRepAnalyzer(
                save_dir=args_cfg.sft_analysis_save_dir,
                categories=args_cfg.sft_analysis_categories,
                dataset_names=self.args.dataset_names
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
        self.loss_log_path = os.path.join(log_dir, f"loss_history_w{self.weight}_{self.exp_name}_{timestamp}.csv")

        # 如果文件不存在，就写表头
        if self.is_main_process and not os.path.exists(self.loss_log_path):
            with open(self.loss_log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "global_step","epoch",
                    "loss_teacher_ce"
                ])

    def _current_observation_ce_factor(self) -> float:
        """Linear warmup from 1.0 -> target over N steps or ratio of total steps.

        Priority: observation_ce_warmup_steps (absolute) > observation_ce_warmup_ratio > no warmup.
        After warmup, clamp to target. If target <= 1.0, return target directly.
        """
        target = float(self._obs_ce_target)
        if target == 1.0:
            return 1.0

        # Decide warmup steps
        warmup_steps = int(self._obs_ce_warmup_steps or 0)

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
        inputs['sft_analysis_poss'] = poss_dict
        if 'sft_analysis_poss' in inputs and isinstance(inputs['sft_analysis_poss'], dict):
            inputs['alignment_poss'] = inputs['sft_analysis_poss'].get('observation_poss', None)
        # Dynamic warmup factor passed to model.forward
        inputs['observation_ce_factor'] = self._current_observation_ce_factor()
        #print("Observation CE Factor:", inputs['observation_ce_factor'])
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
                            # positions dict from inputs['sft_analysis_poss'] per sample index b
                            poss_all = inputs.get('sft_analysis_poss', {})
                            # expect each value is List[List[int]] of length B
                            pos_dict = {}
                            for cat, batch_poss in poss_all.items():
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