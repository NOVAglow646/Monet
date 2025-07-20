from trl import SFTTrainer, SFTConfig
import torch

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
    def alignment_loss(self, student_reps_all_layers, teacher_reps_all_layers, student_poss, teacher_poss):
        total_loss = 0.
        for student_reps, teacher_reps in zip(student_reps_all_layers, teacher_reps_all_layers):
            layer_loss = 0.
            for batch_idx, (student_pos, teacher_pos) in enumerate(zip(student_poss, teacher_poss)):
                student_reps = student_reps[batch_idx, student_pos, :]
                teacher_reps = teacher_reps[batch_idx, teacher_pos, :].detach() # stop gradient
                loss = torch.nn.functional.cosine_similarity(student_reps, teacher_reps).mean()
                layer_loss += 1 - loss
            total_loss += layer_loss/ len(student_poss)
        total_loss = total_loss / len(student_reps_all_layers)
        return total_loss


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        (ce_loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        loss = 0.1 * ce_loss + self.alignment_loss(
            outputs.student_hidden_states,
            outputs.teacher_hidden_states,
            outputs.student_alignment_poss,
            outputs.teacher_alignment_poss
        )
        return (loss, outputs) if return_outputs else loss