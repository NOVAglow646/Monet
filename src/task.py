from PIL import Image
from pathlib import Path
import os
from src.utils import *
from qwen_vl_utils import process_vision_info
def single_input_image_preprocess_function(sample):
    # Load images
    image = Image.open(sample["image_input"][0]).convert("RGB") 
    image_output = Image.open(sample["image_output"]).convert("RGB")

    # Format conversations
    conversations = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": sample["text_input"]},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "image", "image": image_output},
                {"type": "text", "text": sample["text_output"]},
                ],
        }
    ]

    return conversations

def multiple_input_images_preprocess_function(sample):

    # Multiple input images
    user_content = []
    for image in sample['image_input']:
        user_content.append({"type": "image", "image": Image.open(image).convert("RGB") })
    user_content.append({"type": "text", "text": sample["text_input"]})

    image_output = Image.open(sample["image_output"]).convert("RGB")

    conversations = [
        {
            "role": "user", 
            "content": user_content
        }, 
        {
            "role": "assistant", 
            "content": [
                {"type": "image", "image": image_output}, 
                {"type": "text", "text": sample["text_output"]}
                ],
        },
    ]

    return conversations

def OLD_abstract_visual_token_single_input_images_preprocess_function(sample, add_reflection=False):
    conversations = sample
    dataset_root = Path('./new')
    # Process image loading for all steps first
    for i, step in enumerate(conversations):
        new_step = step.copy()
        if step["role"] == "system":
            new_step["content"][0]["text"] = "You are a helpful assistant."
        for j, content in enumerate(new_step["content"]):
            if content["type"] == "image":
                content["image"] = Image.open((Path(dataset_root)/content.pop("image_file_name")).resolve()).convert("RGB")
            new_step["content"][j] = content
        conversations[i] = new_step
    
    # Merge all assistant steps into one
    merged_conversations = []
    assistant_contents = []
    
    first_assistant_step = True
    reflection_str = "Wait, the answer seems incorrect, I need to reconsider manipulating the visual cues. "
    
    for step in conversations:
        if step["role"] == "assistant":
            # Collect all assistant content
            if not first_assistant_step and add_reflection:
                step["content"][0]["text"] = reflection_str + step["content"][0]["text"]
            assistant_contents.extend(step["content"])
            first_assistant_step = False
        else:
            # If we have collected assistant content, merge and add it
            if assistant_contents:
                merged_assistant_step = {
                    "role": "assistant",
                    "content": assistant_contents.copy()
                }

                merged_conversations.append(merged_assistant_step)
                assistant_contents = []
            
            # Add the non-assistant step
            merged_conversations.append(step)
    
    # Handle case where conversation ends with assistant steps
    if assistant_contents:
        merged_assistant_step = {
            "role": "assistant", 
            "content": assistant_contents.copy()
        }
        
        merged_conversations.append(merged_assistant_step)
    
    return merged_conversations

def avt_single_input_images_preprocess_function(sample, dataset_root=""):
    """
    Preprocess function for AVT with single input images.
    """
    conversations = sample["data"]

    # Process image loading for all steps first
    for i, step in enumerate(conversations):
        new_step = step.copy()
        if step["role"] == "system":
            new_step["content"][0]["text"] = "You are a helpful assistant."
        # Track whether an assistant image has appeared before any observation text in this step
        seen_assistant_image = False if step["role"] == "assistant" else None
        for j, content in enumerate(new_step["content"]):        
            if content["type"] == "image":
                #if "image_file_name" not in content:
                #    print(dataset_root)
                img_file_name = content.pop("image_file_name")
                if "kling_mm" in dataset_root:
                    img_file_name = img_file_name.replace("created_dataset/filtered_data/", "")
                content["image"] = os.path.join(dataset_root, img_file_name)
                #content["image"] = content['image_file_name'].replace("created_dataset/filtered_data","/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Dataset/abstract_visual")
                if j>0 and new_step["content"][j-1]["type"] == "text" and step["role"] == "assistant":
                    if "<abs_vis_token></abs_vis_token>" not in new_step["content"][j-1]["text"]:
                        return None
                # Mark that an assistant image has been seen in this step
                if step["role"] == "assistant":
                    seen_assistant_image = True
            elif content["type"] == "text" and step["role"] == "assistant":
                # Validate that any observation text must be preceded by an assistant image within the same step
                if "<observation>" in content.get("text", "") and not seen_assistant_image:
                    return None
            
            new_step["content"][j] = content
        conversations[i] = new_step
    sample["data"] = conversations
    
    return sample

def avt_single_input_images_preprocess_function_question_only(sample, dataset_root="", processor=None, max_seq_len=4096, cur_max=-1, id=0, rank=-1):
    """
    Preprocess function for AVT with single input images.
    """
    conversations = []

    # Process image loading for all steps first
    for i, step in enumerate(sample[:2]):
        new_step = step.copy()
        #if step["role"] == "system":
        #    new_step["content"][0]["text"] += "Here is an example:\n\nWhat is the standing man wearing in this image? \nPut your final answer within \\boxed{}.\n\nI need to locate the standing man in the provided image and observe what he is wearing. To clearly identify what the man is wearing, I will generate a zoomed-in view of his face. \n<abs_vis_token></abs_vis_token> <observation>The zoomed-in image clearly shows the man is wearing a red hat and sunglasses.</observation> Based on the visual evidence, <observation>the man is wearing sunglasses.</observation>"
        seen_assistant_image = False if step["role"] == "assistant" else None
        for j, content in enumerate(new_step["content"]):        
            if content["type"] == "image":
                content["image"] = os.path.join(dataset_root,content.pop("image_file_name")) 
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

def abstract_visual_token_multiple_input_images_preprocess_function(sample):

    # Multiple input images
    user_content = []
    for image in sample['image_input']:
        user_content.append({"type": "image", "image": Image.open(image).convert("RGB") })
    user_content.append({"type": "text", "text": sample["text_input"]})

    image_output = Image.open(sample["image_output"]).convert("RGB")

    conversations = [
        {
            "role": "system", 
            "content": sample["sys_prompt"]
        },
        {
            "role": "user", 
            "content": user_content
        }, 
        {
            "role": "assistant", 
            "content": [
                {"type": "image", "image": image_output}, 
                {"type": "text", "text": sample["text_output"]}
                ],
        },
    ]

    return conversations



task_preporcess_config = {
    'vsp-spatial-reasoning': single_input_image_preprocess_function,
    'vsp-spatial-planning': single_input_image_preprocess_function,
    'blink-jigsaw': multiple_input_images_preprocess_function,
    'sat': multiple_input_images_preprocess_function,
    'mm-reasoning': avt_single_input_images_preprocess_function
}