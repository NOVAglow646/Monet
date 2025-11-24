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
    '''
    Compute a loss (`loss_for_latents`) that backpropagates only through the latent embeddings `latents`.
    '''
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
        safe_grads.append(g.detach())  # detach to stop any 3rd-forward param pathg

    proxy_loss = torch.stack([(v * g).sum() for v, g in zip(ce_vec_list, safe_grads)]).sum()
    return proxy_loss

def load_offline_tensor(tensor_dir, batch_metadata, alignment_layer="all_layers", rep_type="rep", v5_s1_align_poss="obs"):
    '''
    Load precomputed teacher representations (observation tokens for the alignment in SFT stage 2 or the latent embeddings for SFT stage 3)
    '''
    teacher_reps = None
    latents_list = []
    for metadata in batch_metadata:
        dataset_name = metadata['dataset_name']
        sample_id = metadata['sample_id']
        metadata_info = f"{alignment_layer}_{dataset_name}_{sample_id}"
        if v5_s1_align_poss == 'obs':
            metadata_str = f"{rep_type}_{metadata_info}.pt"
        elif v5_s1_align_poss == 'latent_end':
            metadata_str = f"{rep_type}_latent_end_{metadata_info}.pt"
        path = os.path.join(tensor_dir, metadata_str)
        if not os.path.isfile(path):
            latents_list = []
            raise RuntimeError(f"Missing teacher latent file: {path}")
        data = torch.load(path, map_location='cpu')
        latents_list.append(data['latent'].detach())
    if batch_metadata is not None and len(latents_list) == len(batch_metadata):
        teacher_reps = latents_list
    return teacher_reps


class CustomTrainerSFT_STAGE1(SFTTrainer):
    def __init__(self, *args, **kwargs):
        self.exp_name =kwargs.pop('exp_name')
        # accept processing_class (preferred) and fall back to tokenizer for backward compat
        if 'processing_class' not in kwargs and 'tokenizer' in kwargs:
            kwargs['processing_class'] = kwargs.pop('tokenizer')
        super().__init__(*args, **kwargs)
        self.observation_token_acc = 0.
        self.observation_token_acc_step = 0
        self.teacher_ce_cum = 0.0        # cumulative student CE loss
        self.teacher_ce_steps = 0

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

        del teacher_outputs
        gc.collect()
        torch.cuda.empty_cache()
        
        return (teacher_ce_loss, None) if return_outputs else teacher_ce_loss

    def on_epoch_end(self):
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



class CustomTrainerSFT_STAGE2(SFTTrainer):
    def __init__(self, *args, **kwargs):
        self.exp_name = kwargs.pop('exp_name')
        # accept processing_class (preferred) and fall back to tokenizer for backward compat
        if 'processing_class' not in kwargs and 'tokenizer' in kwargs:
            kwargs['processing_class'] = kwargs.pop('tokenizer')
        super().__init__(*args, **kwargs)

        self.ce_emphasize_factor = self.args.ce_emphasize_factor
        self.teacher_ce_loss_cum = 0.0        # cumulative teacher CE loss
        self.teacher_ce_loss_steps = 0
        self.observation_token_acc = 0.
        self.observation_token_acc_step = 0
        self.alignment_loss_cum = 0.
        self.alignment_loss_steps = 0


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        # ------------------------------------------------------------------
        # Latent forward to get ce_patch_pos (positions of latent embeddings) and ce_patch_vec (latent embeddings).
        # Multiple forward is needed since we need to autoregressively generate latents.
        # ------------------------------------------------------------------
        inputs['latent_mode'] = True
        inputs['loss_type'] = []
        model.gradient_checkpointing_disable() # since we set use_cache=True in latent forward, we must disable grad checkpointing
        outputs = model(**inputs, return_dict=True, output_hidden_states=False)
        
        # ------------------------------------------------------------------
        # Insert the collected latent embeddings into the latent positions, and forward once.
        # ------------------------------------------------------------------
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        inputs['latent_mode'] = False
        inputs['ce_patch_pos'] = outputs.ce_patch_pos
        inputs['ce_patch_vec'] = outputs.ce_patch_vec
        inputs['ce_emphasize_poss'] = inputs['observation_poss']
        inputs['ce_emphasize_factor'] = self.ce_emphasize_factor
        inputs['loss_type'] = ['ce']
        if self.args.alignment_weight != 0:
            inputs['loss_type'].append('alignment')

        inputs['compute_emphasize_acc'] = True
        # Ensure training forward does NOT request attentions (prevents checkpoint recompute mismatch)
        inputs.pop('output_attentions', None)
        inputs.pop('attn_analysis', None)

        if self.args.alignment_weight != 0:
            # Load precomputed teacher representations of observation tokens for alignment loss
            teacher_reps = load_offline_tensor(self.args.teacher_reps_dir, batch_metadata=inputs['metadata'], 
            alignment_layer=self.args.alignment_layer, v5_s1_align_poss=self.args.v5_s1_align_poss)
            if self.args.v5_s1_align_poss == 'obs':
                inputs['alignment_poss'] = inputs['observation_poss']
            elif self.args.v5_s1_align_poss == 'latent_end':
                inputs['alignment_poss'] = inputs['latent_end_poss']
            inputs['teacher_hidden_states_for_alignment'] = teacher_reps

        teacher_ce_loss, teacher_output = super().compute_loss(
                model, 
                inputs,
                return_outputs=True, num_items_in_batch=num_items_in_batch
            )
        
        alignment_loss = teacher_output.loss_dict.get('alignment', torch.tensor(0.0))
        if self.args.emphasize_latent_weight != 0.0 and alignment_loss.item() != 0.0: # latent-only backpropagation for alignment loss
            latent_only_loss = compute_latents_only_loss(outputs.ce_patch_vec, self.args.alignment_weight * alignment_loss)
            loss = self.args.emphasize_latent_weight * latent_only_loss + teacher_ce_loss
        else:
            loss = teacher_ce_loss + self.args.alignment_weight * alignment_loss

        if getattr(teacher_output, 'mean_emphasize_acc', None) is not None:
            self.observation_token_acc += getattr(teacher_output, 'mean_emphasize_acc')
            self.observation_token_acc_step += 1

        self.teacher_ce_loss_cum += teacher_ce_loss.item()
        self.teacher_ce_loss_steps += 1
        self.alignment_loss_cum += alignment_loss.item()
        self.alignment_loss_steps += 1

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

        return (loss, None) if return_outputs else loss


    def on_epoch_end(self):
        return super().on_epoch_end()

    def log(self, logs: dict, start_time: float | None = None):
        # Merge our rolling averages into the standard logs once per logging call
        merged = dict(logs)
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

class CustomTrainerSFT_STAGE3(SFTTrainer):
    def __init__(self, *args, **kwargs): 
        self.exp_name =kwargs.pop('exp_name')
        super().__init__(*args, **kwargs)
        self.alignment_weight = self.args.alignment_weight
        self.ce_emphasize_factor: float = float(getattr(self.args, 'ce_emphasize_factor', 1.0))
        # Where to read precomputed teacher latents
        self.teacher_latent_dir = getattr(self.args, 'teacher_latent_dir', None)
        if not self.teacher_latent_dir:
            raise ValueError("teacher_latent_dir must be specified for SFT Stage 3")

        self.observation_token_acc = 0.
        self.observation_token_acc_step = 0
        self.al_loss_cum = 0.0       # cumulative alignment loss since last log
        self.al_steps = 0            # number of micro-steps accumulated
        self.student_ce_loss_cum = 0.0        # cumulative student CE loss
        self.student_ce_loss_steps = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss for SFT stage3 with optional cached teacher latents.
        """
        # Load precomputed teacher latents
        teacher_latents = load_offline_tensor(self.teacher_latent_dir, batch_metadata=inputs['metadata'], alignment_layer=self.args.alignment_layer, rep_type="latent")

        # ------------------------------------------------------------------
        # Latent forward to get ce_patch_pos (positions of latent embeddings) and ce_patch_vec (latent embeddings)
        # ------------------------------------------------------------------
        inputs['latent_mode'] = True
        inputs['input_ids'] = inputs['student_input_ids']
        inputs['attention_mask'] = inputs['student_attention_mask']
        inputs['pixel_values'] = inputs['student_pixel_values']
        inputs['image_grid_thw'] = inputs['student_image_grid_thw']
        if 'labels' in inputs:
            inputs.pop('labels')
        inputs['alignment_poss'] = inputs['student_alignment_poss']
        inputs['teacher_hidden_states_for_alignment'] = teacher_latents
        model.gradient_checkpointing_disable() # since we set use_cache=True in latent forward, we must disable grad checkpointing
        inputs['loss_type'] = []
        inputs['output_hidden_states'] = False
        student_outputs_latent = model(**inputs)
        

        # Student CE forward
        inputs['latent_mode'] = False
        inputs['labels'] = inputs['student_labels']
        inputs['ce_patch_pos'] = student_outputs_latent.ce_patch_pos
        inputs['ce_patch_vec'] = student_outputs_latent.ce_patch_vec
        inputs['ce_emphasize_factor'] = self.ce_emphasize_factor
        inputs['ce_emphasize_poss'] = inputs['observation_poss']
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        inputs['loss_type'] = ['ce', 'alignment']
        inputs['compute_emphasize_acc'] = True
        if 'student_attention_mask_4d' in inputs:
            inputs['attention_mask_4d'] = inputs.pop('student_attention_mask_4d')
        (student_ce_loss, student_outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        if getattr(student_outputs, 'mean_emphasize_acc', None) is not None:
            self.observation_token_acc += getattr(student_outputs, 'mean_emphasize_acc')
            self.observation_token_acc_step += 1
        alignment_loss = student_outputs.loss_dict['alignment']
        loss = student_ce_loss + self.alignment_weight * alignment_loss
        outputs_student_loss = student_ce_loss.item()

        del student_outputs
        step = int(getattr(self.state, 'global_step', 0) or 0)
        if step > 0 and (step % 20 == 0):
            try:
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass

        # Logging
        self.al_loss_cum += float(alignment_loss.detach().item())
        self.al_steps += 1
        self.student_ce_loss_cum += outputs_student_loss
        self.student_ce_loss_steps += 1

        return (loss, None) if return_outputs else loss
    
    def log(self, logs: dict, start_time: float | None = None):
        # Merge our rolling averages into the standard logs once per logging call
        merged = dict(logs)
        if self.al_steps > 0:
            merged["alignment_loss"] = round(self.al_loss_cum / max(1, self.al_steps), 6)
            self.al_loss_cum = 0.0
            self.al_steps = 0
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

