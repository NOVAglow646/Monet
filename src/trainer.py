from trl import SFTTrainer, SFTConfig
import torch
import os, csv, torch, datetime
import gc
import numpy as np

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
                    "loss_student_ce", "loss_teacher_ce",
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
                sim = torch.nn.functional.cosine_similarity(student_rep_l_b, teacher_rep_l_b).mean()
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
        inputs['labels'] = inputs['teacher_labels']
        inputs['alignment_poss'] = inputs['teacher_alignment_poss']
        inputs['image_out_mask'] = inputs['teacher_image_out_mask']
        (teacher_ce_loss, teacher_outputs) = super().compute_loss(
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
        loss = teacher_ce_loss + student_ce_loss + self.weight *alignment_loss

        outputs_student_loss = student_ce_loss.item()
        outputs_teacher_loss = teacher_ce_loss.item()

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
                    outputs_teacher_loss,
                    alignment_loss.item() if isinstance(alignment_loss, torch.Tensor) else alignment_loss,
                ])
        # --------------------------------------------
        
        
        return (loss, None) if return_outputs else loss
    
    
    
    
class CustomTrainerSFT(SFTTrainer):
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
                    "loss_teacher_ce"
                ])


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
        (teacher_ce_loss, teacher_outputs) = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
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