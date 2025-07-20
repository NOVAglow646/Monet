from PIL import Image
from pathlib import Path

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


def avt_single_input_images_preprocess_function(sample):
    """
    Preprocess function for AVT with single input images.
    """
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
    
    return conversations



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