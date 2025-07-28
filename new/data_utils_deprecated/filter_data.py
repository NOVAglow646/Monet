from datasets import Dataset
import json
from PIL import Image
import os
from AAA_vllm_toolkit.load_and_gen_hf import *
from AAA_vllm_toolkit.load_and_gen_vllm import *
from AAA_vllm_toolkit.extract_and_check import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="CoF", choices=["PixelReasoner", "CoM", "CoF"],
                    help="The name of the dataset to filter.")
parser.add_argument("--devices", type=str, default="0")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

sys_prompt = "You are a helpful assistant. You can generate abstract visual tokens that represent a cropped image region or images with auxiliary information like lines, bounding boxes, etc. When you decide to generate abstract visual tokens, put them in <abs_vis_token>...</abs_vis_token>."

def load_jsonl_dataset(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
        data = data[:]
    return Dataset.from_list(data)

def load_json_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


from PIL import Image, ImageDraw, ImageFont
import os

def choice2str(gt_choices, letter):
    if gt_choices is None:
        return letter
    return gt_choices[ord(letter) - ord('A')]

def add_boxed_instruction(question: str):
    return question + " Put your final answer within \\boxed{}. If you cannot see relevant visual information to infer the answer from the image, just output \\boxed{None} and don't guess the answer based on your knowledge."

def draw_bboxes(
    img,
    bboxes: list,
    labels: list | None = None,
    colors: list | None = None,
    save_path: str | None = None,
    line_width: int = 3,
    font_path: str | None = None,
    font_size: int = 16,
):
    """
    在图片上绘制 bounding boxes。

    参数
    ----
    bboxes : list[tuple[int, int, int, int]]
        每个 bounding box 以 (xmin, ymin, xmax, ymax) 像素坐标表示。
    labels : list[str] | None
        与 bboxes 对应的文本标签（例如类名或置信度）。可省略。
    colors : list[str] | None
        每个框的颜色（任意 Pillow 认识的颜色字符串或 RGB 元组）。长度与 bboxes 相同，缺省则自动循环常用色。
    save_path : str | None
        保存路径；若为 None 则只返回 Image 对象，不落盘。
    line_width : int
        框线宽度。
    font_path : str | None
        字体文件路径；缺省时 Pillow 会用默认位图字体，可能不支持中文。
    font_size : int
        标签字号。
    """
    draw = ImageDraw.Draw(img)

    # 颜色准备
    default_palette = ["red", "lime", "blue", "yellow", "cyan", "magenta", "orange"]
    if colors is None:
        colors = [default_palette[i % len(default_palette)] for i in range(len(bboxes))]

    # 字体准备
    if labels:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            # 某些 Pillow 版本的默认字体不支持中文；如需中文请显式指定 font_path
            font = ImageFont.load_default()

    # 主循环
    for i, bbox in enumerate(bboxes):
        if len(bbox) != 4:
            continue
        xmin, ymin, xmax, ymax = bbox
        color = colors[i]
        try:
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=line_width)
        except Exception as e:
            print(f"Error drawing rectangle {i}: {e}")
            return None

        if labels and i < len(labels):
            text = labels[i]
            text_size = draw.textlength(text, font=font)
            text_height = font.getbbox(text)[3] - font.getbbox(text)[1]
            # 文字背景框
            draw.rectangle(
                [xmin, ymin - text_height - 4, xmin + text_size + 6, ymin],
                fill=color,
            )
            # 文字
            draw.text(
                (xmin + 3, ymin - text_height - 2),
                text,
                fill="black",
                font=font,
            )

    if save_path:
        img.save(save_path)
        print(f"结果已保存到: {save_path}")
    return img

def draw_line(
    img,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: str | tuple[int, int, int] = "red",
    width: int = 3,
    save_path: str | None = None,
    canvas_size: tuple[int, int] | None = None,
    bg_color: str | tuple[int, int, int] = "white",
) -> Image.Image:
    """
    给定两点坐标，在图片上画线。

    参数
    ----
    pt1, pt2 : (x, y)
        线段两端点坐标，像素单位。
    color : str | RGB
        线条颜色；可用 Pillow 支持的任何颜色名称或 (R,G,B) 元组。
    width : int
        线宽（像素）。
    save_path : str | None
        若指定则保存到该路径，否则只返回 Image 对象。
    canvas_size : (w, h) | None
        当 img_path 为 None 时必须指定，用于创建空白画布的尺寸。
    bg_color : str | RGB
        新建画布时的背景色。
    """
    draw = ImageDraw.Draw(img)
    draw.line([pt1, pt2], fill=color, width=width)

    if save_path:
        img.save(save_path)
        print(f"已保存到 {save_path}")

    return img

def valid_img_size(img: Image.Image):
    if img is None:
        return False
    w,h = img.size 
    if w == 0 or h == 0:
        return False
    return True

def insert_alignment_tokens(text: str, llm, sampling_params):
    sys_prompt = (
        "You are a helpful assistant. You need to decide what are the observations obtained by the visual manipulations. Put these observations in <observation>...</observation>"
        "and keep other texts unchanged. If there are no observations attained from the visual manipulations, just output the original text.\n"
        "Here are some examples:\n\n"
        "Input: The cropped part doesn't contain the target object, I will zoom in again.<abs_vis_token></abs_vis_token>\n"
        "Your output: <observation> The cropped part doesn't contain the target object, I will zoom in again.</observation><abs_vis_token></abs_vis_token>\n\n"
        "Input: The cropped section explains that the process involves several stages, with the first stage being autonomous state estimation from the onboard sensors.\n\n\\boxed{Autonomous state estimation from the onboard sensors}\n"
        "Your output: <observation> The cropped section explains that the process involves several stages, with the first stage being autonomous state estimation from the onboard sensors.</observation>\n\n\\boxed{Autonomous state estimation from the onboard sensors}\n\n"
        "Input: Then, draw a line segment from the leftmost side of the bush-troop. <abs_vis_token></abs_vis_token>\n"
        "Output: Then, draw a line segment from the leftmost side of the bush-troop. <observation></observation><abs_vis_token></abs_vis_token>\n\n"
        "Input: The length of the nail is about 2 inches, so the answer is 2.\n"
        "Output: <observation> The length of the nail is about 2 inches,</observation> so the answer is 2.\n\n"
        "Input: By comparing Cape Verde and the 1400 Line, it is known that the Line representing Cape Verde is above the 1400 Line.\n"
        "Output: By comparing Cape Verde and the 1400 Line, <observation>it is known that the Line representing Cape Verde is above the 1400 Line.</observation>\n\n"
        "Input: In img1, there are 6 data points from Georgia above the Line (y=15), which means there are 6 years of data greater than 15, so the answer is: 6. \n\nAfter removing tool callings and coordinates:\nThere are 6 data points from Georgia above the line (y=15), which means there are 6 years of data greater than 15, so the answer is: 6.\n"
        "Output: <observation> In img1, there are 6 data points from Georgia above the Line (y=15),</observation> which means there are 6 years of data greater than 15. So the answer is: 6.\n\n"
    )
    formated_text = "Now, it's your turn.\n\nInput: " + text + "\nOutput: "
    inputs = vllm_llm_process_batch_data(sys_prompt=sys_prompt, usr_prompts=[formated_text], tokenizer=llm.get_tokenizer())
    output = vllm_llm_generate(inputs, sampling_params, llm)
    return output[0].outputs[0].text.strip()


def get_filtered_data(sample, dataset_root, models, sampling_params, tokenizer, processor, valid_id, dataset_name):
    helper_images = []
    gt = None
    gt_choices = None
    saved_img_root = f"./created_dataset/filtered_data/{dataset_name}/images"
    os.makedirs(saved_img_root, exist_ok=True)
    mllm_sampling_params = sampling_params['mllm']
    llm_sampling_params = sampling_params['llm']
    
    manipulation_cot = []
    
    llm = models["llm"]
    llm.wake_up()
    
    cot_to_save = [{
        "role": "system",
        "content": [
            {"type": "text", "text": sys_prompt}
        ]
    }]
    
    if 'PixelReasoner' in dataset_root:
        def get_pure_question(text: str):
            s = text.find("Question:")
            e = text.find("\nThink in the mind first,")
            if s == -1 or e == -1:
                return text.strip()
            return text[s + len("Question:"):e].strip()

        def extract_choices(text: str):
            s = text.find("\nchoices:\n")
            e = text.find("\n\nGuidelines")
            if s == -1 or e == -1:
                return None
            choices_str = text[s + len("\nchoices:\n"):e]
            return [line.split(':')[1].strip() for line in choices_str.strip().splitlines() if line.strip()]

        def remove_choices(question: str):
            s = question.find("\nchoices:\n")
            if s == -1:
                return question.strip()
            return question[:s].strip()

        #question = replace_guideline(get_pure_question(sample["question"]))
        gt_choices = extract_choices(sample["question"])
        question = remove_choices(sample["question"])
        question = add_boxed_instruction(question)
        sample["question"] = question
        qid = sample["qid"]
        input_image = Image.open(os.path.join(dataset_root, sample["image"])).convert("RGB")
        conversations = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": input_image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        cot_to_save.append({
            "role": "user",
            "content": [
                {"type": "image", "image": input_image, "image_file_name": f"{saved_img_root}/{valid_id}_0.jpg"},
                {"type": "text", "text": question},
            ]
        })
        width, height = input_image.size
        # Format conversations
        #print(sample["question"])
        steps = sample["response_steps"]
        for i, step in enumerate(steps):
            if i>0:
                step_str = insert_alignment_tokens(step["response_str"], llm, llm_sampling_params)
            else:
                step_str = step["response_str"]
            step_content = [
                        {"type": "text", "text": step_str},
                    ]
            #print(step["response_str"])
            if step["manipulation"]:
                if step["manipulation"]["type"] == "crop":
                    if "<abs_vis_token>" in step["response_str"]:
                        bbox_norm = step["manipulation"]["parameters"]
                        #print(bbox_norm)
                        x_min = int(bbox_norm[0] * width)
                        y_min = int(bbox_norm[1] * height)
                        x_max = int(bbox_norm[2] * width)
                        y_max = int(bbox_norm[3] * height)
                        helper_image = input_image.crop((x_min, y_min, x_max, y_max))
                        if not valid_img_size(helper_image):
                            print(f"Invalid image size for helper image, return None")
                            return None
                        helper_images.append(helper_image)
                        step_content.append(
                            {"type": "image", "image": helper_image}
                        )

            manipulation_cot.append({
                "role": "assistant",
                "content": step_content
            })
            gt = extract_boxed_answer(step["response_str"])
        gt = choice2str(gt_choices, gt) if gt_choices else gt
        
    elif "CoM" in dataset_root:
        def remove_choices(question: str):
            s = question.find("\nChoices:")
            if s == -1:
                return question.strip()
            return question[:s].strip()

        def extract_choices(question: str):
            s = question.find("\nChoices:\n")
            choices_str = question[s + len("\nChoices:\n"):].strip()
            pattern = r"\([A-H]\)\s*(.*?)\s*(?=\([A-H]\)|$)"
            matches = re.findall(pattern, choices_str, re.DOTALL)
            return [m.strip() for m in matches]
        
        def add_instruction(question: str):
            if "Choices" in question:
                gt_choices = extract_choices(question)
                return add_boxed_instruction(remove_choices(question)), gt_choices
            else:
                return add_boxed_instruction(question), None
            
        
        qid = sample["qid"]
        input_image = Image.open(os.path.join(dataset_root, sample["image"])).convert("RGB")
        question = sample["question"]
        question, gt_choices = add_instruction(question)
        conversations = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": input_image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        cot_to_save.append({
            "role": "user",
            "content": [
                {"type": "image", "image": input_image, "image_file_name": f"{saved_img_root}/{valid_id}_0.jpg"},
                {"type": "text", "text": question},
            ]
        })
        width, height = input_image.size
        gt = sample["answer"]
        steps = sample["response_steps"]
        
        for i, step in enumerate(steps):
            if i>0:
                step_str = insert_alignment_tokens(step["response_str"], llm, llm_sampling_params)
            else:
                step_str = step["response_str"]
                
            step_content = [
                {"type": "text", "text": step_str},
            ]
            if step["manipulation"]:
                #print(i, step["manipulation"]["type"])
                if step["manipulation"]["type"] == "crop_and_zoomin":
                    bbox = step["manipulation"]["parameters"]
                    #print(bbox)
                    x_min, y_min, x_max, y_max = bbox
                    helper_image = input_image.crop((x_min, y_min, x_max, y_max))

                elif step["manipulation"]["type"] == "grounding":
                    bboxes = step["manipulation"]["parameters"]
                    if bboxes is None:
                        return None
                    if isinstance(bboxes, dict):
                        return None
                    if not isinstance(bboxes[0], list):
                        bboxes = [bboxes]
                    helper_image = draw_bboxes(input_image.copy(), bboxes=bboxes)
    
                elif step["manipulation"]["type"] == "line":
                    pts = step["manipulation"]["parameters"]
                    if pts is None:
                        return None
                    if pts[2] - pts[0] < pts[3] - pts[1]:
                        pts[2] = pts[0]
                    else:
                        pts[3] = pts[1]
                    helper_image = draw_line(input_image.copy(), (pts[0], pts[1]), (pts[2], pts[3]))
                else:
                    return None    
                    
                if not valid_img_size(helper_image):
                    print(f"Invalid image size for helper image, return None")
                    return None
                helper_images.append(helper_image)
                step_content.append(
                    {"type": "image", "image": helper_image}
                )

            manipulation_cot.append({
                "role": "assistant",
                "content": step_content
            })
    
    elif 'CoF' in dataset_root:
        img_id = 1
        def rmv_tool_call_instruction(question: str):
            s = question.find("\nThink in the mind first,")
            if s == -1:
                return question.strip()
            return question[:s].strip().replace("<image> ", "")
        
        def rmv_choices(question: str):
            s = question.find("\n(A)")
            if s == -1:
                return question.strip()
            return question[:s].strip()
        
        def extract_choices(question: str):
            s = question.find("?\n")
            e = question.find("\nAnswer with the option")
            choices_str = question[s + 2:e].strip()
            pattern = r"\([A-D]\)\s*(.*?)\s*(?=\([A-D]\)|$)"
            matches = re.findall(pattern, choices_str, re.DOTALL)
            return [m.strip() for m in matches]
        
        
        def get_bbox_and_rmv_tool(text: str):
            text = text.replace("<think> ", "").replace("</think>", "").replace("<answer>", "\\boxed{").replace("</answer>", "}")
            s = text.find("<tool_call>")
            e = text.find("</tool_call>")
            if s == -1 or e == -1:
                return None, text
            tool_call_str = text[s+len("<tool_call>"):e]
            bbox = json.loads(tool_call_str)["arguments"]["bbox_2d"]
            return bbox, text[:s] + "<abs_vis_token></abs_vis_token>"

        if len(sample['images']) < 2: # the problem is correctly answered with the original image in the gt cot
            return None
        
        input_image_path = sample["images"][0]
        qid = int(input_image_path.split('/')[0])
        steps = sample['messages']
        input_image = Image.open(os.path.join(dataset_root, 'images', input_image_path)).convert("RGB")
        question = rmv_tool_call_instruction(steps[1]["content"])
        question = rmv_choices(question)
        question = add_boxed_instruction(question)
        gt_choices = extract_choices(question)
        conversations = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": input_image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        cot_to_save.append({
            "role": "user",
            "content": [
                {"type": "image", "image": input_image, "image_file_name": f"{saved_img_root}/{valid_id}_0.jpg"},
                {"type": "text", "text": question},
            ]
        })
        
        for i, step in enumerate(steps[2:]):
            if step["role"] == "user":
                continue
            gt = extract_html_answer(step["content"])
            bbox, resp_wo_tool = get_bbox_and_rmv_tool(step["content"])
            
            if i>0:
                step_str = insert_alignment_tokens(resp_wo_tool, llm, llm_sampling_params)
            else:
                step_str = resp_wo_tool
            
            step_content = [
                {"type": "text", "text": step_str},
            ]
            
            if bbox is not None:
                helper_image = Image.open(os.path.join(dataset_root, 'images', sample["images"][img_id])).convert("RGB")
                if not valid_img_size(helper_image):
                    print(f"Invalid image size for helper image, return None")
                    return None
                helper_images.append(helper_image)
                step_content.append(
                    {"type": "image", "image": helper_image}
                )
                img_id += 1


            
            manipulation_cot.append({
                "role": "assistant",
                "content": step_content
            })
         
        if gt in ['A','B','C','D', 'E', 'F', 'G', 'H']:
            gt = choice2str(gt_choices, gt)
    llm.sleep()
       
    def ask_llm(conv, mllm, sampling_params):
        """简单包装 vllm 推理接口，返回字符串答案"""
        inputs=vllm_mllm_process_batch_from_messages([conv], processor)
        total_input_len = count_qwen_vl_tokens(inputs, tokenizer, processor)
        if total_input_len[0] > max_model_len:
            print(f"[Exit] Too many input tokens: {total_input_len} > {max_model_len}. Skipping this sample.")
            return None, None
        outputs = mllm.generate(inputs, sampling_params=sampling_params, use_tqdm=False)
        # vllm 默认批量；这里只取单条
        raw_resp = outputs[0].outputs[0].text.strip()
        return extract_boxed_answer(raw_resp), raw_resp

    # 2.1 首轮仅原始图 + 问题
    print(f"[Q {valid_id}]: {question}")
    policy_mllm = models["policy_mllm"]
    policy_mllm.wake_up()
    extracted_pred, raw_pred = ask_llm(conversations, policy_mllm, mllm_sampling_params)
    policy_mllm.sleep()
    if raw_pred is None or extracted_pred is None:
        print(f"===== [Exit] No valid answered can be extracted. Skipping this sample.")
        return None
    print(f"[A {valid_id}]: {raw_pred}")
    if batch_judge([extracted_pred], [gt], [gt_choices])[0]:
        # 起始就答对，无需 helper，直接丢弃此样本
        print("===== [Zero-shot Correct] correct answer without helper images. Discarding this sample.")
        return None


    # 2.2 add helper images, remove the original input image
    img_id = 1
    conversations[0]['content'] = conversations[0]['content'][1:] # remove the original input image
    conversations.append(
        {
            "role": "assistant",
            "content": []
        }
    )
    
    cot_to_save.append({
        "role": "assistant",
        "content": []
    })
    
    strong_mllm = models["strong_mllm"]
    strong_mllm.wake_up()
    for i, step in enumerate(manipulation_cot):
        manipulation_str = step["content"][0]["text"]
        if len(step["content"]) == 1:# text-only step
            cot_to_save[-1]["content"].append({"type": "text", "text": manipulation_str},)
            continue
        
        # step with manipulation
        img = step["content"][1]["image"]
        

        conversations[1]["content"].extend([
                {"type": "text", "text": manipulation_str},
                {"type": "image", "image": img},
            ])

        
        extracted_pred, raw_pred = ask_llm(conversations, strong_mllm, mllm_sampling_params)
        if raw_pred is None:
            return None
        img_file_name = f"{saved_img_root}/{valid_id}_{img_id}.jpg"
        cot_to_save[-1]["content"].extend([
                {"type": "text", "text": manipulation_str},
                {"type": "image", "image": img, "image_file_name": img_file_name},
            ])
        print(f"[A {valid_id}.{img_id}]: {raw_pred}")
        
        if batch_judge([extracted_pred], [gt], [gt_choices])[0]: # wrong -> correct given this helper image
            strong_mllm.sleep()
            #input_image.save(f"{saved_img_root}/{valid_id}_0.jpg")

            cot_to_save_wo_pil_img = []
            for j, step in enumerate(cot_to_save):
                new_step = {
                    "role": step["role"],
                    "content": []
                }
                for k, content in enumerate(step["content"]):
                    if content["type"] == "image":
                        img = content["image"]
                        img.save(content["image_file_name"])
                        new_content = {"type": "image", "image_file_name": content["image_file_name"]}
                    else:
                        new_content = content
         
                    new_step["content"].append(new_content)

                if j == len(cot_to_save) - 1: # response, last step
                    llm.wake_up()
                    new_content = {
                        "type": "text",
                        "text": insert_alignment_tokens(raw_pred, llm, llm_sampling_params)
                    }
                    new_step["content"].append(new_content)
                    llm.sleep()
                cot_to_save_wo_pil_img.append(new_step)

            print(f"==== [Correct] Helper image {img_id} helped, save this sample.")

            return cot_to_save_wo_pil_img
            
        else:
            print(f"==== [Wrong] Helper image {img_id} did not help")

        img_id += 1
    # 全部 helper 用完仍答错
    strong_mllm.sleep()
    print(f"==== [Exit] All helper images exhausted, no correct answer found for QID {qid}.")
    return None        

    

strong_mllm_dir = "/data1/qxwang/checkpoints/Qwen2.5-VL-7B-Instruct" # "/data1/qxwang/checkpoints/Qwen2.5-VL-32B-Instruct" #
strong_mllm, mllm_sampling_params = vllm_mllm_init(strong_mllm_dir)
strong_mllm.sleep()

policy_mllm_dir = "/data1/qxwang/checkpoints/Qwen2.5-VL-7B-Instruct"
policy_mllm, mllm_sampling_params = vllm_mllm_init(policy_mllm_dir)
policy_mllm.sleep()
tokenizer  = AutoTokenizer.from_pretrained(policy_mllm_dir, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(policy_mllm_dir)

llm_dir = "/data1/qxwang/checkpoints/Qwen2.5-7B-Instruct" # "/data0/huggingface/Qwen/Qwen2.5-32B-Instruct" # 
llm, llm_sampling_params = vllm_llm_init(llm_dir)
llm.sleep()

models = {
    "strong_mllm": strong_mllm,
    "policy_mllm": policy_mllm,
    "llm": llm,
}

sampling_params = {
    "mllm": mllm_sampling_params,
    "llm": llm_sampling_params,
}

dataset_name = args.dataset_name
dataset_mapping = {
    "PixelReasoner": {
        "dataset_path": "/data1/qxwang/datasets/multimodal/PixelReasoner-SFT-Data/processed_data.json",
        "dataset_root": "/data1/qxwang/datasets/multimodal/PixelReasoner-SFT-Data"
    },
    "CoM": {
        "dataset_path": "/data1/qxwang/datasets/multimodal/CoMDataset/com_math_processed.jsonl",
        "dataset_root": "/data1/qxwang/datasets/multimodal/CoMDataset"
    },
    "CoF": {
        "dataset_path": "/data1/qxwang/datasets/multimodal/CoF-SFT-Data-5.4k/cof_sft_data.json",
        "dataset_root": "/data1/qxwang/datasets/multimodal/CoF-SFT-Data-5.4k"
    }
}

dataset_path = dataset_mapping[dataset_name]["dataset_path"]
dataset_root = dataset_mapping[dataset_name]["dataset_root"]
train_dataset = load_json_dataset(dataset_path)


filtered_train_dataset = []
valid_id = 114
sample_start_id = 671

os.makedirs(f"./created_dataset/filtered_data/{dataset_name}", exist_ok=True)
filtered_file_path = f"./created_dataset/filtered_data/{dataset_name}/filtered_train.json"

# Initialize empty file or clear existing content

if valid_id == 0 and sample_start_id == 0:
    with open(filtered_file_path, "w", encoding="utf-8") as f:
        f.write("[\n")

for sample in tqdm(train_dataset[sample_start_id:], desc=f"Filtering {dataset_name} training data", total=len(train_dataset)):
    processed = get_filtered_data(sample, dataset_root, models, sampling_params, tokenizer, processor, valid_id, dataset_name)
    if processed is not None:
        with open(filtered_file_path, "a", encoding="utf-8") as f:
            if valid_id > 0:
                f.write(",\n")
            json.dump(processed, f, ensure_ascii=False, indent=4)
        valid_id += 1


if valid_id == 0 and sample_start_id == 0:
    with open(filtered_file_path, "a", encoding="utf-8") as f:
        f.write("\n]")