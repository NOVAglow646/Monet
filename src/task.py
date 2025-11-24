from PIL import Image
from pathlib import Path
import os
from src.utils import *
from qwen_vl_utils import process_vision_info

def Monet_single_input_images_preprocess_function(sample, dataset_root="", allow_no_observation=False):
    """
    Preprocess function for Monet with single input images, interleaved CoT.
    """
    n_img_pad = 0
    n_img = 0
    conversations = sample["data"]
    seen_observation = False
    # Process image loading for all steps first
    for i, step in enumerate(conversations):
        new_step = step.copy()
        if step["role"] == "system":
            new_step["content"][0]["text"] = "You are a helpful assistant."
        # Track whether an assistant image has appeared before any observation text in this step
        seen_assistant_image = False if step["role"] == "assistant" else None
        for j, content in enumerate(new_step["content"]):        
            if content["type"] == "image":
                img_file_name = content.pop("image")
                if "kling_mm" in dataset_root:
                    img_file_name = img_file_name.replace("created_dataset/filtered_data/", "")
                content["image"] = os.path.join(dataset_root, img_file_name)
                if j>0 and new_step["content"][j-1]["type"] == "text" and step["role"] == "assistant":
                    if "<abs_vis_token></abs_vis_token>" not in new_step["content"][j-1]["text"]:
                        #print("[Preprocess] No <abs_vis_token> before assistant image. Discard this sample")
                        return None
                # Mark that an assistant image has been seen in this step
                if step["role"] == "assistant":
                    n_img += 1
                    seen_assistant_image = True
            elif content["type"] == "text":
                
                if step["role"] == "assistant":
                    n_img_pad += content['text'].count('<abs_vis_token></abs_vis_token>')
                    # Validate that any observation text must be preceded by an assistant image within the same step
                    if "<observation>" in content.get("text", "") and not seen_assistant_image:
                        content['text'] = content['text'].replace("<observation>", "").replace("</observation>", "")
                    if "<observation>" in content.get("text", ""):
                        seen_observation = True

                elif step["role"] == "user":
                    img_key = "image"
                    if 'Zebra_CoT_visual_search' not in new_step["content"][0][img_key] and 'Zebra_CoT_count' not in new_step["content"][0][img_key]: # keep boxed instructions for Zebra_CoT_visual_search
                        content["text"] = content["text"].replace("\nPut your final answer within \\boxed{}.", "")

            new_step["content"][j] = content
        conversations[i] = new_step
    sample["data"] = conversations
    
    if n_img != n_img_pad:
        print(f"n_img ({n_img}) != num of <abs_vis_token></abs_vis_token> ({n_img_pad}), discard this sample")
        return None

    if not seen_observation and not allow_no_observation:
        #print("[Preprocess] No observation found in assistant responses. Discard this sample")
        return None

    return sample

def Monet_single_input_images_preprocess_function_question_only(sample, dataset_root="", cur_max=-1, id=0, rank=-1):
    """
    Preprocess function for Monet with single input images, question only.
    """
    conversations = []

    # Process image loading for all steps first
    for i, step in enumerate(sample[:2]):
        new_step = step.copy()
        seen_assistant_image = False if step["role"] == "assistant" else None
        for j, content in enumerate(new_step["content"]):        
            if content["type"] == "image":
                content["image"] = os.path.join(dataset_root,content.pop("image")) 
                if j>0 and new_step["content"][j-1]["type"] == "text" and step["role"] == "assistant":
                    if "<abs_vis_token></abs_vis_token>" not in new_step["content"][j-1]["text"]:
                        return None, cur_max
                if step["role"] == "assistant":
                    seen_assistant_image = True
            elif content["type"] == "text" and step["role"] == "assistant":
                if "<observation>" in content.get("text", "") and not seen_assistant_image:
                    return None, cur_max
            
            new_step["content"][j] = content
        conversations.append(new_step)

    return conversations, cur_max


task_preporcess_config = {
    'mm-reasoning': Monet_single_input_images_preprocess_function
}

    """
    Simulate the action sequence on the provided map.
    """
    action_sequence = parse_action_sequence(path_str)

    start = None
    for r, row in enumerate(map_desc):
        for c, val in enumerate(row):
            if val == 1:
                start = (r, c)
                break
        if start is not None:
            break

    if start is None:
        raise ValueError("The map description does not contain a start position (cell value 1).")
    
    current_position = start
    for action in action_sequence:
        if action not in ACTION_MAP:
            return {
                "success": False,
                "status": "Invalid action",
                "invalid": True,
            }
            raise ValueError(f"Invalid action: {action}. Valid actions are {list(ACTION_MAP.keys())}.")
        
        dr, dc = ACTION_MAP[action]
        new_r = current_position[0] + dr
        new_c = current_position[1] + dc

        # Check boundaries; ignore the move if it goes out-of-bounds.
        if new_r < 0 or new_r >= len(map_desc) or new_c < 0 or new_c >= len(map_desc[0]):
            continue
        
        current_position = (new_r, new_c)
        
        # If the new cell is a hole (-1), immediately return failure.
        if map_desc[new_r][new_c] == -1:
            return {
                "success": False,
                "status": "Fell in hole",
                "invalid": False,
            }

    # Check if the final position is the goal (2).
    final_r, final_c = current_position
    if map_desc[final_r][final_c] == 2:
        return {
            "success": True,
            "status": "Reached goal",
            "invalid": False,
        }
    else:
        return {
            "success": False,
            "status": "Did not reach goal",
            "invalid": False,
        }