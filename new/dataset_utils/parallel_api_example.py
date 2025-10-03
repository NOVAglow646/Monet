import os
from glob import glob
import cv2
import json
import base64
import time
import math
import traceback
import numpy as np
import sys
from tqdm import tqdm
from PIL import Image
from copy import deepcopy
import concurrent.futures as cf
from tqdm import tqdm
import os, time, traceback
import numpy as np

MAX_RETRY = 15


# os.environ['GOOGLE_APPLICATION_CREDENTIALS']='/pfs/gaohuan03/gemini_exp/mmu-gemini-2test-52d3c3234a01.json'
os.environ[
    "GOOGLE_APPLICATION_CREDENTIALS"
] = "/pfs/gaohuan03/gemini_exp/mmu-gemini-caption-1-5pro-86ec97219196.json"
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models

gemini_generation_config = {"max_output_tokens": 9000, "temperature": 0.0, "top_p": 1.0}
safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.OFF,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.OFF,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.OFF,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.OFF,
}
vertexai.init(project="mmu-gemini-caption-1-5pro", location="us-central1")
gemini_model = GenerativeModel("gemini-2.5-pro")  # NOTICE


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def save_jsonl(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def sample_video_frames(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_sample = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    sampled_frames = []
    frame_ids = set(frames_to_sample)
    current_frame = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame in frame_ids:
            _, buffer = cv2.imencode(".jpg", frame)
            sampled_frames.append(base64.b64encode(buffer).decode("utf-8"))
        current_frame += 1
        if current_frame > max(frame_ids):
            break
    cap.release()
    return sampled_frames


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


def process_one(item_0):
    """单条样本的并行处理函数（在子进程中运行）"""
    item = deepcopy(item_0)
    try:
        SCORE = 1
        img = item["generated_image"]
        if img == "" or img is None or not os.path.exists(img):
            SCORE = 0
            item["score"] = SCORE
            return item

        image_base64 = encode_image(img)
        contents = [Part.from_data(mime_type="image/png", data=image_base64)]
        for qa in item["question_list"]:
            response_text = ""

            origin_q = qa["question"]  # 原始问题
            gt_answer = qa["answer"]  # 标准答案

            for attempt in range(MAX_RETRY):
                try:

                    prompt = (
                        "Please answer the following question based on the image:\n"
                        + "Question: "
                        + origin_q
                        + "\n\nYou should only reply yes or no, and do not provide any other extra content."
                    )
                    contents.append(prompt)
                    print(f"\nPrompt:\n{prompt}")

                    responses = gemini_model.generate_content(
                        contents,
                        generation_config=gemini_generation_config,
                        safety_settings=safety_settings,
                    )
                    response_text = getattr(responses, "text", "") or ""
                    if response_text == "" or response_text is None:
                        continue
                    else:
                        break
                except Exception:
                    # 打印堆栈便于定位；做简单指数退避
                    traceback.print_exc()
                    continue
            response_text = response_text.lower()
            print(f"\nResponse:\n{response_text}")
            if "yes" in response_text:
                response_text = "yes"
            if "no" in response_text:
                response_text = "no"
            if response_text == gt_answer:
                continue
            else:
                SCORE = 0
                break

        # 打印与返回
        item["score"] = SCORE
        return item

    except Exception as e:
        print("Error:", repr(e))
        import sys

        sys.exit(1)


if __name__ == "__main__":
    task_json_list = {
        "Gemini_Banana": "/mmu_vcg_ssd/shiyang06/Project/RealUnify/res/Nano_Banana/step/res/under.json",
    }
    RES_JSON_DIR = (
        # "/mmu_vcg_ssd/shiyang06/Project/RealUnify/res/AAA-IMAGE-EVAL/STEP/GEN"
        "/mmu_vcg_ssd/shiyang06/Project/RealUnify/res/ORACLE_CASE/gen-to-und/gemini_banana"
    )
    for model_name in task_json_list:

        json_file = task_json_list[model_name]

        if json_file.endswith(".jsonl"):
            data = load_jsonl(json_file)
        elif json_file.endswith(".json"):
            data = load_json(json_file)

        # ===== 可选：像你之前那样“先切块再并行” =====
        # 注：对进程池逐条 submit 就已能均衡分配任务；手动切块不是必须。
        # 如果你确实想先切块，可用 np.array_split 确保不遗漏样本：
        # num_workers = 50
        # chunks = [list(c) for c in np.array_split(data, num_workers) if len(c) > 0]
        # 然后把每个元素仍然逐条提交给进程池（见下），无需再手写多层进程。

        num_workers = 50  # 按需调整；I/O / API 调用为主，50 通常可接受
        new_data = []

        with cf.ProcessPoolExecutor(max_workers=num_workers) as ex:
            # 逐条提交任务；进程池会自动做均衡调度
            futures = [ex.submit(process_one, item) for item in data]
            for fut in tqdm(cf.as_completed(futures), total=len(futures)):
                new_data.append(fut.result())

        res_json_file = os.path.join(RES_JSON_DIR, f"{model_name}_res.json")

        save_json(new_data, res_json_file)

        print("Total samples: ", len(new_data))
