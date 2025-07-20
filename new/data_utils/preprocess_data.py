import os
import json
from collections import defaultdict
from typing import List, Dict, Any
from datasets import Dataset
from AAA_vllm_toolkit.load_and_gen_vllm import *
from AAA_vllm_toolkit.load_and_gen_hf import *
import re



dataset_name = 'PixelReasoner-SFT-Data'
dataset_name = 'CoMDataset'
dataset_path = "/data1/qxwang/datasets/multimodal/PixelReasoner-SFT-Data"
dataset_path = '/data1/qxwang/datasets/multimodal/CoMDataset'
save_path = os.path.join(dataset_path, 'com_math_processed.jsonl')

sys_prompt = (
    "You are a helpful assistant. You can generate abstract visual tokens that represent a cropped image region or images with auxiliary information like lines, bounding boxes, etc. "
    "When you decide to generate abstract visual tokens, put them in <abs_vis_token>...</abs_vis_token>."
)

def load_jsonl_dataset(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
        data = data[:]
    return data

def load_json_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def load_dataset(file_path):
    if file_path.endswith('.jsonl'):
        return load_jsonl_dataset(file_path)
    elif file_path.endswith('.json'):
        return load_json_dataset(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")



def preprocess_data(dataset_name, dataset_path, model_path = '/data0/huggingface/Qwen/Qwen2.5-14B-Instruct'):
    processed_data = []
    if dataset_name == 'PixelReasoner-SFT-Data':
        def find_bbox(text: str):
            start = text.find('"bbox_2d":')
            end = text.find(',"target_image"')
            if start != -1 and end != -1:
                bbox_str = text[start + len('"bbox_2d":'):end].strip()
                return json.loads(bbox_str)
            return None
        
        def remove_tool_call_inst(text: str):
            start = text.find("Determine if it is")
            end = text.find("by `crop_image`. ")
            if start != -1 and end != -1:
                return text[:start] + text[end + len("by `crop_image`. "):]
            return text

        def replace_tool_call(text: str):
            start = text.find("<tool_call>")
            end = text.find("</tool_call>")
            if start != -1 and end != -1:
                return text[:start] + "<abs_vis_token></abs_vis_token>" + text[end + len("</tool_call>"):]
            return text
        
        unprocessed_data = load_dataset(os.path.join(dataset_path, 'release.json'))

        for item in unprocessed_data:
            data = {}
            data["sys_prompt"] = sys_prompt
            data["qid"] = item['qid']
            data["response_steps"] = []
            if 'image' not in item['message_list'][1]['content'][1]:
                continue
            for i, step in enumerate(item['message_list']):
                if i==1: # question
                    data['question'] = remove_tool_call_inst(step['content'][0]['text'])
                    data["image"] = step['content'][1]['image']
                if step['role'] == 'assistant':
                    step_response_str = step['content'][0]['text']
                    step_dict = {}
                    step_dict["manipulation"] = {}
                    bbox = find_bbox(step_response_str)
                    if bbox is not None:
                        step_dict["manipulation"]["type"] = "crop"
                        step_dict["manipulation"]["parameters"] = bbox
                    step_dict["response_str"] = replace_tool_call(step_response_str)
                    data["response_steps"].append(step_dict)
                
            processed_data.append(data)
    if dataset_name == 'CoMDataset':
        def get_line_coordinates(text: str):
            start = text.find("[")
            end = text.find("]")
            if start != -1 and end != -1:
                coordinates_str = text[start:end + 1]
                try:
                    coordinates = json.loads(coordinates_str)
                    return coordinates
                except json.JSONDecodeError:
                    return None

        def replace_res_txt(text: str, variable: dict) -> str:
            pattern = re.compile(r'`((?:res|txt)_\d+)`')

            def repl(match: re.Match) -> str:
                key = match.group(1)            # 拿到 res_12 / txt_3
                return str(variable.get(key, match.group(0)))  # 有值就替换；否则原样返回

            return pattern.sub(repl, text)

        def llm_process(inst_prompt, usr_prompts, llm_dir):
            llm, sampling_params = vllm_llm_init(llm_dir)
            tokenizer = load_tokenizer(llm_dir)
            inputs = vllm_llm_process_batch_data(sys_prompt=inst_prompt, usr_prompts=usr_prompts, tokenizer=tokenizer)
            outputs = vllm_llm_generate(inputs, sampling_params, llm)
            return outputs
        
        unprocessed_data = load_dataset(os.path.join(dataset_path, 'com.jsonl'))

        
        inst_prompt = (
            "You should complete the following requirements:\n"
            "1. Remove tool callings like 'Based on GROUNDING(...)', 'use LINE([64, 245, 209, 266], 1, 2)->img_1.', 'Leveraging OCR(texts in image `img_1`)', etc. while keeping the original meaning of the text.\n"
            "2. Remove all coordinates like '[623,114,33,27]' or bounding box information like 'bbx_1`, `bbx_2', 'bbx_3', etc. in the text.\n"
        )

        demos_com_math=(
            "Here are some examples:\n\n"
            "Input: Use the GROUNDING(bar) to outline each bar column with the coordinates `bbx_2`, `bbx_3`, `bbx_4`, `bbx_5`, `bbx_6`, `bbx_7`, `bbx_8`, `bbx_9`, and `bbx_10`."
            "Your output: Outline each bar column.\n\n"
            "Input: Then, based on the result of the comparison, and GROUNDING(all data points below the Line), find the positions of all data points below the Line in the figure: [623,114,33,27; 532,120,27,24; 432,117,22,24]."
            "Your output: Then, based on the result of the comparison, find the positions of all data points below the Line in the figure.\n\n"
            "Input: Use GROUNDING(all countries) to find the specific positions of the Line charts corresponding to all countries in the picture: South Korea is located at `bbx_2`, Japan at `bbx_3`, Norway at `bbx_4`, Australia at `bbx_5`, and Belgium at `bbx_6`."
            "Your output: Find the specific positions of the Line charts corresponding to all countries in the picture.\n\n"
            "Input: Then, use Grounding to locate the data points of the Line graphs for the years 2003 and 2016, which are positioned at [101,371,44,46] and [681,53,51,58], respectively, within the area of [104,60,612,355] in the image."
            "Your output: Then, locate the data points of the Line graphs for the years 2003 and 2016.\n\n"
        )

        demos_com=(
            "Here are some examples:\n\n"
            "Input: Leveraging CROP_AND_ZOOMIN(region `bbx_1`) to crop and zoom in the region defined by `bbx_1`, and the result is `img_1`."
            "Your output: Crop and zoom in the region defined by `bbx_1`, and the result is `img_1`.\n\n"
            "Input: Leveraging OCR(texts in region `bbx_2`) to interpret the texts in region `bbx_2`, which is `txt_1`."
            "Your output: Interpret the texts in region `bbx_2`, which is `txt_1`.\n\n"
            "Input: Leveraging CALCULATE(the type of tea based on `txt_1`) to determine the type of tea based on `txt_1`, resulting `res_1`."
            "Your output: Determine the type of tea based on `txt_1`, resulting `res_1`.\n\n"
        )
        query_prompt = ("Now it's your turn. Input: Next, use GROUNDING(green) to find the three points larger than 40 in the diagram,"
                    " their positions are `bbx_2`; `bbx_3`; `bbx_4`."
        )

        all_steps = []

        for item in unprocessed_data:
            data = {}
            data["sys_prompt"] = sys_prompt
            metadata = item['metadata'][0]
            data["image_size"] = item['img_size']
            data["image"] = item["image_path"]
            data["qid"] = metadata["pid"]
            data["question"] = metadata['question']
            data["answer"] = metadata['answer']
            data["response_steps"] = []

            for i, step in enumerate(metadata['final_com'].values()):
                step_dict = {}
                step_dict["manipulation"] = {}
                if step['func'] is not None:
                    if 'line' in step['func']:
                        step_dict["manipulation"]["type"] = "line"
                        step_dict["manipulation"]["parameters"] = get_line_coordinates(step['param'])
                    elif 'grounding' in step['func']:
                        step_dict["manipulation"]["type"] = "grounding"
                        step_dict["manipulation"]["parameters"] = step['return']
                    elif 'crop_and_zoom_in' in step['func']:
                        step_dict["manipulation"]["type"] = "crop"
                        step_dict["manipulation"]["parameters"] = step['onbox']
                    elif 'calculate' in step['func']:
                        step_dict["manipulation"]["type"] = "calculate"
                        step_dict["manipulation"]["parameters"] = step['param']
                        step['desc'] = replace_res_txt(step['desc'], step['variables'])
                all_steps.append(step['desc'])    
                data["response_steps"].append(step_dict)
            processed_data.append(data)

        usr_prompts = [inst_prompt + demos_com_math + "Now it's your turn. Input: " + step for step in all_steps]
        
        outputs = llm_process(inst_prompt, usr_prompts, model_path)
        outputs_iter = iter(outputs)
        
        for data in processed_data:
            steps = data["response_steps"]
            new_steps = []
            for step in steps:
                processed_step = next(outputs_iter).outputs[0].text
                if "type" in step["manipulation"]:
                    processed_step += " <abs_vis_token></abs_vis_token>"
                step["response_str"] = processed_step
                new_steps.append(step)
            data["response_steps"] = new_steps
                    
    return processed_data

def save_processed_data(processed_data: List[Dict[str, Any]], output_path: str):
    with open(output_path, 'w') as f:
        json.dump(processed_data, f)
        
processed_data = preprocess_data(dataset_name, dataset_path)
save_processed_data(processed_data, save_path)