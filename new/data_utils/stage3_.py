#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 3: Alignment (全步拼接) + 分步还原 + 打包训练集
=====================================================

与上一版 Stage3 差异（按用户最新要求）：
1. **Alignment 输入** = 所有 helper step 文本 (0..K) + strong_mllm 首次答对步原始输出，按顺序拼接，并加 *显式标记*；整体送入 alignment LLM。
2. Alignment LLM 在上下文整体基础上插入 `<observation>...</observation>`；**要求原样保留每段标记**，我们据此解析回每一步文本。
3. **最终保存的 assistant content：**
   - step0..K：使用 alignment 后的分步文本（分别替换原 helper 文本）。
   - 每步后跟其图像（如有）。
   - 最后一项只放 alignment 后的 *strong_mllm 正确回答段*（不再重复之前所有文本）。

若 alignment 输出缺失或标记不完整，将回退到原始未对齐文本（保持稳健）。
"""

from __future__ import annotations
import os
import json
import re
import argparse
from typing import List, Dict, Any, Optional, Tuple

from transformers import AutoTokenizer

# --- 用户工具包 -------------------------------------------------------------
from AAA_vllm_toolkit.load_and_gen_vllm import (
    vllm_llm_init,
    vllm_llm_process_batch_data,
    vllm_generate,
)
from AAA_vllm_toolkit.deepseek_api import build_ds_client, get_ds_response

# ---------------------------------------------------------------------------
# 标记模板（在 alignment 输入/输出中包裹各段，便于解析）
# ---------------------------------------------------------------------------
STEP_START = "<STEP_{i}>"
STEP_END   = "<END_STEP_{i}>"
FINAL_START = "<FINAL_STEP>"
FINAL_END   = "<END_FINAL_STEP>"

# ---------------------------------------------------------------------------
# Alignment 系统提示（告知 LLM 保留标记、仅在段内插入 observation）
# ---------------------------------------------------------------------------
ALIGN_SYS_PROMPT = (
    "You are a helpful assistant. You need to decide what are the observations obtained by the visual manipulations (denoted by <abs_vis_token></abs_vis_token>). "
    "Put these observations in <observation>...</observation> and keep other texts unchanged. "
    "Segments are delimited by explicit markers like <STEP_0> ... <END_STEP_0> and <FINAL_STEP> ... <END_FINAL_STEP>.\n"
    "Rules:\n"
    "1. Do NOT remove, reorder, or rename any markers. Always output ALL markers exactly as given.\n"
    "2. Only observations after <abs_vis_token></abs_vis_token> should be wrapped in <observation>...</observation>.\nIf there's no <abs_vis_token></abs_vis_token> ahead, the description about the image doesn't count as an observation."
    "3. You should judge whether the content in \\boxed{} is obtained by observing the image or reasoning. If it is directly obtained by observing the image, put the observation in <observation>...</observation>; If it is obtained by repeating a previous observation, don't wrap it in <observation>...</observation>\n"
    "4. Do NOT add any new content in the reasoning chain.\n"
)

examples_pool = {
    "CoF": (
        "Here are some examples:\n\n"
        "## Input: <STEP_0>To determine what the X-axis stands for, I need to analyze the X-axis in the image. However, the X-axis is not clearly visible in the initial view due to the image's resolution. To improve visibility, I need to explore step by step.<END_STEP_0>\n<STEP_1>\nI first locate the X-axis within the bounding box [1118, 1092, 1208, 1131]. I zoom in on this area to obtain a refined visual embedding. \n<abs_vis_token></abs_vis_token>\n<END_STEP_1>\n<FINAL_STEP>\\boxed{Visit}<END_FINAL_STEP>\n\n"
        "## Your output: <STEP_0>To determine what the X-axis stands for, I need to analyze the X-axis in the image. However, the X-axis is not clearly visible in the initial view due to the image's resolution. To improve visibility, I need to explore step by step.<END_STEP_0>\n<STEP_1>\nI first locate the X-axis within the bounding box [1118, 1092, 1208, 1131]. I zoom in on this area to obtain a refined visual embedding. \n<abs_vis_token></abs_vis_token>\n<END_STEP_1>\n<FINAL_STEP><observation>\\boxed{Visit}</observation><END_FINAL_STEP>\n\n\n"
        "## Input: <STEP_0>Now let me take a closer look at the image. <abs_vis_token></abs_vis_token>\n<END_STEP_0>\n<STEP_1>The cropped section explains that the process involves several stages, with the first stage being autonomous state estimation from the onboard sensors.\n\n<END_STEP_1>\n<FINAL_STEP>\\boxed{Autonomous state estimation from the onboard sensors}\n\n"
        "## Your output: <STEP_0>Now let me take a closer look at the image. <abs_vis_token></abs_vis_token>\n<END_STEP_0>\n<STEP_1><observation>The cropped section explains that the process involves several stages, with the first stage being autonomous state estimation from the onboard sensors.</observation>\n\n<END_STEP_1>\n<FINAL_STEP>\\boxed{Autonomous state estimation from the onboard sensors}\n\n\n"
        "## Input: <STEP_0>The image shows an emblem carved into a wall. The emblem features a crowned eagle with outstretched wings, holding a sphere in one claw. Below the eagle, there is a decorative object at the bottom of the emblem, which appears to have a distinct shape.\n\nNow I will zoom in to look clearer at the object at the bottom of the emblem.<abs_vis_token></abs_vis_token>\n<END_STEP_0>\n<STEP_1>The cropped part doesn't contain the target object, I will zoom in again.<END_STEP_1>\n<FINAL_STEP><abs_vis_token></abs_vis_token>\n\nThe object at the bottom of the emblem has a leaf-like shape. It appears to be a stylized or ornate leaf design, possibly resembling an oak leaf or another type of foliage.\n\n\\boxed{\\text{leaf}}<FINAL_STEP>\n\n"
        "## Your output: <STEP_0>The image shows an emblem carved into a wall. The emblem features a crowned eagle with outstretched wings, holding a sphere in one claw. Below the eagle, there is a decorative object at the bottom of the emblem, which appears to have a distinct shape.\n\nNow I will zoom in to look clearer at the object at the bottom of the emblem.<abs_vis_token></abs_vis_token>\n<END_STEP_0>\n<STEP_1><observation>The cropped part doesn't contain the target object,</observation> I will zoom in again.<END_STEP_1>\n<FINAL_STEP><abs_vis_token></abs_vis_token>\n\n<observation>The object at the bottom of the emblem has a leaf-like shape. It appears to be a stylized or ornate leaf design, possibly resembling an oak leaf or another type of foliage.</observation>\n\n\\boxed{\\text{leaf}}<END_FINAL_STEP>\n\n\n"
    ),
    "CoM_w_MathVista": (
        "Here are some examples:\n\n"
        "## Input: <STEP_0>Draw a straight line and obtain the new image after drawing the line. <abs_vis_token></abs_vis_token><END_STEP_0>\n<STEP_1>In the new image, find the year with PDI=1.5 for the North Atlantic cyclone.<END_STEP_1>\n<FINAL_STEP>The value of PDI=1.5 corresponds to the index point two positions after 1975, so the year for PDI=1.5 is 1977, hence the answer is 1977.<END_FINAL_STEP>\n\n"
        "## Your output: <STEP_0>Draw a straight line and obtain the new image after drawing the line. <abs_vis_token></abs_vis_token><END_STEP_0>\n<STEP_1>In the new image, find the year with PDI=1.5 for the North Atlantic cyclone.<END_STEP_1>\n<FINAL_STEP><observation>The value of PDI=1.5 corresponds to the index point two positions after 1975, so the year for PDI=1.5 is 1977, hence the answer is 1977.</observation><END_FINAL_STEP>\n\n\n"
        "## Input: <STEP_0>Draw a straight line and obtain the new image after the line is drawn. <abs_vis_token></abs_vis_token><END_STEP_0>\n<FINAL_STEP>To determine the year when the blue bar crossed 400 marks, let's analyze the data in the image:\n\n1. The x-axis represents years, and the y-axis represents the number of residents in millions.\n2. We need to identify the first year where the height of the blue bar exceeds 400.\n\nLooking at the data:\n- In 2055, the value is 407.41 million.\n- In 2060, the value is 416.8 million.\n\nThe blue bar crosses 400 marks between 2055 and 2060. Since the question asks for the specific year, we can conclude that it crosses 400 in **2055**.\n\n\\boxed{2055}<END_FINAL_STEP>\n\n"
        "## Your output: <STEP_0>Draw a straight line and obtain the new image after the line is drawn. <abs_vis_token></abs_vis_token><END_STEP_0>\n<FINAL_STEP>To determine the year when the blue bar crossed 400 marks, let's analyze the data in the image:\n\n<observation>1. The x-axis represents years, and the y-axis represents the number of residents in millions.</observation>\n2. We need to identify the first year where the height of the blue bar exceeds 400.\n\nLooking at the data:\n<observation>- In 2055, the value is 407.41 million.\n- In 2060, the value is 416.8 million.\n\nThe blue bar crosses 400 marks between 2055 and 2060.</observation> Since the question asks for the specific year, we can conclude that it crosses 400 in **2055**.\n\n\\boxed{2055}<END_FINAL_STEP>\n\n\n"
        "## Input: <STEP_0>Draw a line to obtain the new image after the line is drawn. <abs_vis_token></abs_vis_token><END_STEP_0>\n<FINAL_STEP>To determine the sum of real users below 1 million, we need to identify the services with user counts less than 1 million and then sum their values.\n\nFrom the bar chart:\n\n- HBOGo.pl: 0.63 million\n- Ipla TV: 0.59 million\n- Chili.com: 0.25 million\n- NCplusgo.pl: 0.21 million\n\nNow, let's sum these values:\n\n\\[\n0.63 + 0.59 + 0.25 + 0.21 = 1.68\n\\]\n\nThus, the sum of real users below 1 million is:\n\n\\boxed{1.68}\n\n"
        "## Your output: <STEP_0>Draw a line to obtain the new image after the line is drawn. <abs_vis_token></abs_vis_token><END_STEP_0>\n<FINAL_STEP>To determine the sum of real users below 1 million, we need to identify the services with user counts less than 1 million and then sum their values.\n\nFrom the bar chart:\n\n<observation>- HBOGo.pl: 0.63 million\n- Ipla TV: 0.59 million\n- Chili.com: 0.25 million\n- NCplusgo.pl: 0.21 million</observation>\n\nNow, let's sum these values:\n\n\\[\n0.63 + 0.59 + 0.25 + 0.21 = 1.68\n\\]\n\nThus, the sum of real users below 1 million is:\n\n\\boxed{1.68}\n\n\n"
        ),
    "CoM_wo_MathVista": (
        "Here are some examples:\n\n"
        "## Input: <STEP_0>Draw a straight line and obtain the new image after drawing the line. <abs_vis_token></abs_vis_token><END_STEP_0>\n<STEP_1>In the new image, find the year with PDI=1.5 for the North Atlantic cyclone.<END_STEP_1>\n<FINAL_STEP>The value of PDI=1.5 corresponds to the index point two positions after 1975, so the year for PDI=1.5 is 1977, hence the answer is 1977.<END_FINAL_STEP>\n\n"
        "## Your output: <STEP_0>Draw a straight line and obtain the new image after drawing the line. <abs_vis_token></abs_vis_token><END_STEP_0>\n<STEP_1>In the new image, find the year with PDI=1.5 for the North Atlantic cyclone.<END_STEP_1>\n<FINAL_STEP><observation>The value of PDI=1.5 corresponds to the index point two positions after 1975, so the year for PDI=1.5 is 1977, hence the answer is 1977.</observation><END_FINAL_STEP>\n\n\n"
        "## Input: <STEP_0>Draw a straight line and obtain the new image after the line is drawn. <abs_vis_token></abs_vis_token><END_STEP_0>\n<FINAL_STEP>To determine the year when the blue bar crossed 400 marks, let's analyze the data in the image:\n\n1. The x-axis represents years, and the y-axis represents the number of residents in millions.\n2. We need to identify the first year where the height of the blue bar exceeds 400.\n\nLooking at the data:\n- In 2055, the value is 407.41 million.\n- In 2060, the value is 416.8 million.\n\nThe blue bar crosses 400 marks between 2055 and 2060. Since the question asks for the specific year, we can conclude that it crosses 400 in **2055**.\n\n\\boxed{2055}<END_FINAL_STEP>\n\n"
        "## Your output: <STEP_0>Draw a straight line and obtain the new image after the line is drawn. <abs_vis_token></abs_vis_token><END_STEP_0>\n<FINAL_STEP>To determine the year when the blue bar crossed 400 marks, let's analyze the data in the image:\n\n<observation>1. The x-axis represents years, and the y-axis represents the number of residents in millions.</observation>\n2. We need to identify the first year where the height of the blue bar exceeds 400.\n\nLooking at the data:\n<observation>- In 2055, the value is 407.41 million.\n- In 2060, the value is 416.8 million.\n\nThe blue bar crosses 400 marks between 2055 and 2060.</observation> Since the question asks for the specific year, we can conclude that it crosses 400 in **2055**.\n\n\\boxed{2055}<END_FINAL_STEP>\n\n\n"
        "## Input: <STEP_0>Draw a line to obtain the new image after the line is drawn. <abs_vis_token></abs_vis_token><END_STEP_0>\n<FINAL_STEP>To determine the sum of real users below 1 million, we need to identify the services with user counts less than 1 million and then sum their values.\n\nFrom the bar chart:\n\n- HBOGo.pl: 0.63 million\n- Ipla TV: 0.59 million\n- Chili.com: 0.25 million\n- NCplusgo.pl: 0.21 million\n\nNow, let's sum these values:\n\n\\[\n0.63 + 0.59 + 0.25 + 0.21 = 1.68\n\\]\n\nThus, the sum of real users below 1 million is:\n\n\\boxed{1.68}\n\n"
        "## Your output: <STEP_0>Draw a line to obtain the new image after the line is drawn. <abs_vis_token></abs_vis_token><END_STEP_0>\n<FINAL_STEP>To determine the sum of real users below 1 million, we need to identify the services with user counts less than 1 million and then sum their values.\n\nFrom the bar chart:\n\n<observation>- HBOGo.pl: 0.63 million\n- Ipla TV: 0.59 million\n- Chili.com: 0.25 million\n- NCplusgo.pl: 0.21 million</observation>\n\nNow, let's sum these values:\n\n\\[\n0.63 + 0.59 + 0.25 + 0.21 = 1.68\n\\]\n\nThus, the sum of real users below 1 million is:\n\n\\boxed{1.68}\n\n\n"
        ),
    "ReFocus": (
        "Here are some examples:\n\n"
        "## Input: <STEP_0>This is a horizontal bar chart image. I need to find the number of aircraft due to enter the 'lease end' phase in 2018. The relevant y value is 'AerCap', as it has the highest number of lease returns and extensions.<abs_vis_token></abs_vis_token><END_STEP_0>\n<FINAL_STEP>To determine the number of aircraft due to enter the 'lease end' phase in 2018 for AerCap, we need to analyze the horizontal bar chart provided:\n\n1. The chart shows the number of lease returns and extensions for various leasing companies.\n2. The y-axis lists the leasing companies, with \"AerCap\" being the first entry.\n3. The x-axis represents the number of lease returns and extensions.\n4. The bar for AerCap extends to the far right, indicating the highest number of lease returns and extensions among the listed companies.\n5. The length of the AerCap bar corresponds to a value of approximately 190 on the x-axis.\n\nGiven this information, the number of aircraft due to enter the 'lease end' phase in 2018 for AerCap is approximately 190.\n\n\\boxed{190}<END_FINAL_STEP>\n\n"
        "## Your output: <STEP_0>This is a horizontal bar chart image. I need to find the number of aircraft due to enter the 'lease end' phase in 2018. The relevant y value is 'AerCap', as it has the highest number of lease returns and extensions.<abs_vis_token></abs_vis_token><END_STEP_0>\n<FINAL_STEP>To determine the number of aircraft due to enter the 'lease end' phase in 2018 for AerCap, we need to analyze the horizontal bar chart provided:\n\n<observation>1. The chart shows the number of lease returns and extensions for various leasing companies.</observation>\n<observation>2. The y-axis lists the leasing companies, with \"AerCap\" being the first entry.</observation>\n<observation>3. The x-axis represents the number of lease returns and extensions.</observation>\n<observation>4. The bar for AerCap extends to the far right, indicating the highest number of lease returns and extensions among the listed companies.</observation>\n5. <observation>The length of the AerCap bar corresponds to a value of approximately 190 on the x-axis.</observation>\n\nGiven this information, the number of aircraft due to enter the 'lease end' phase in 2018 for AerCap is approximately 190.\n\n\\boxed{190}<END_FINAL_STEP>\n\n\n"
        "## Input: <STEP_0>This is a horizontal bar chart image. I need to focus on the part when the y-axis value equals 'Consumer Electronics' to find out the number of retail chains in Austria in 2020.<abs_vis_token></abs_vis_token><END_STEP_0>\n<FINAL_STEP>To determine the number of retail chains in Austria in the year 2020 for the category \"Consumer Electronics,\" let's follow these steps:\n\n1. **Identify the relevant category**: The question asks specifically about \"Consumer Electronics.\"\n\n2. **Locate the category in the chart**: In the chart, \"Consumer Electronics\" is listed as one of the categories on the y-axis.\n\n3. **Read the value for \"Consumer Electronics\"**: The bar corresponding to \"Consumer Electronics\" shows a value of 21.\n\n4. **Verify the year**: The chart does not explicitly state the year, but the question specifies the year 2020. Since the chart is from Statista 2021 and is likely reporting data for the previous year, it is reasonable to assume the data is for 2020.\n\n5. **Final Answer**: The number of retail chains in Austria for \"Consumer Electronics\" in 2020 is 21.\n\n\\boxed{21}<END_FINAL_STEP>\n\n"
        "## Your output: <STEP_0>This is a horizontal bar chart image. I need to focus on the part when the y-axis value equals 'Consumer Electronics' to find out the number of retail chains in Austria in 2020.<abs_vis_token></abs_vis_token><END_STEP_0>\n<FINAL_STEP>To determine the number of retail chains in Austria in the year 2020 for the category \"Consumer Electronics,\" let's follow these steps:\n\n1. **Identify the relevant category**: The question asks specifically about \"Consumer Electronics.\"\n\n2. **Locate the category in the chart**: In the chart, \"Consumer Electronics\" is listed as one of the categories on the y-axis.\n\n3. **Read the value for \"Consumer Electronics\"**: The bar corresponding to \"Consumer Electronics\" shows a value of 21.\n\n4. **Verify the year**: The chart does not explicitly state the year, but the question specifies the year 2020. Since the chart is from Statista 2021 and is likely reporting data for the previous year, it is reasonable to assume the data is for 2020.\n\n<observation>5. **Final Answer**: The number of retail chains in Austria for \"Consumer Electronics\" in 2020 is 21.</observation>\n\n\\boxed{21}<END_FINAL_STEP>\n\n\n"
        "## Input: <STEP_0>This is a vertical bar chart image, and I need to focus on the part where the x-axis value is '35 to 54' to find out how many veterans were employed.<abs_vis_token></abs_vis_token><END_STEP_0>\n<FINAL_STEP>To determine the number of veterans employed in the United States in 2019 for the age group \"35 to 54,\" we need to focus on the blue section of the bar corresponding to this age group.\n\nFrom the chart:\n- The blue section (representing \"Employed\") for the age group \"35 to 54\" is labeled as **3,290,974**.\n\nThus, the number of veterans employed in the United States in 2019 for the age group \"35 to 54\" is:\n\n\\boxed{3290974}<END_FINAL_STEP>\m\n\n"
        "## Your output: <STEP_0>This is a vertical bar chart image, and I need to focus on the part where the x-axis value is '35 to 54' to find out how many veterans were employed.<abs_vis_token></abs_vis_token><END_STEP_0>\n<FINAL_STEP>To determine the number of veterans employed in the United States in 2019 for the age group \"35 to 54,\" we need to focus on the blue section of the bar corresponding to this age group.\n\n<observation>From the chart:\n- The blue section (representing \"Employed\") for the age group \"35 to 54\" is labeled as **3,290,974**.</observation>\n\nThus, the number of veterans employed in the United States in 2019 for the age group \"35 to 54\" is:\n\n\\boxed{3290974}<END_FINAL_STEP>\m\n\n"
        )
}
# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def save_json(data: List[Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# Alignment 输入构建 & 解析
# ---------------------------------------------------------------------------

def build_alignment_text_for_sample(rec: Dict[str,Any]) -> Tuple[str,int]:
    """Return (full_text, num_steps=K+1). K = first_correct_helper_idx."""
    k = rec["first_correct_helper_idx"]
    helpers = rec.get("helpers", [])
    segments = []
    for i in range(k+1):
        txt = helpers[i].get("text", "")
        segments.append(STEP_START.format(i=i) + txt + STEP_END.format(i=i))
    # strong output at step k
    strong_txt = rec["strong_steps"][k].get("pred_raw", "")
    segments.append(FINAL_START + strong_txt + FINAL_END)
    full = "\n".join(segments)
    return full, k+1  # num segments incl FINAL

# regex precompile (DOTALL)
_STEP_RE_TPL = r"<STEP_{i}>\s*(.*?)\s*<END_STEP_{i}>"
_FINAL_RE = re.compile(r"<FINAL_STEP>\s*(.*?)\s*<END_FINAL_STEP>", re.S)


def parse_aligned_text(text: str, k: int) -> Tuple[List[str], str]:
    """Parse alignment output. Return (aligned_steps[0..k], aligned_final)."""
    aligned_steps = []
    for i in range(k+1):  # steps 0..k
        pat = re.compile(_STEP_RE_TPL.format(i=i), re.S)
        m = pat.search(text)
        if m:
            aligned_steps.append(m.group(1).strip())
        else:
            aligned_steps.append("")  # fallback later
    m = _FINAL_RE.search(text)
    aligned_final = m.group(1).strip() if m else ""
    return aligned_steps, aligned_final

# ---------------------------------------------------------------------------
# Alignment 批生成
# ---------------------------------------------------------------------------

def build_align_prompts(inputs: List[str]) -> List[str]:
    return ["Now it's your turn. ## Input: " + t + "\n\n## Your output:" for t in inputs]


def batch_align_concat(dataset_name: str, full_texts: List[str], llm, sampling_params, tokenizer, batch_size: int=512, ds_client=None) -> List[str]:
    out_texts: List[str] = []
    n = len(full_texts)
    for i in range(0, n, batch_size):
        chunk = full_texts[i:i+batch_size]
        prompts = build_align_prompts(chunk)
        if llm is not None:
            inputs = vllm_llm_process_batch_data(sys_prompt=ALIGN_SYS_PROMPT+examples_pool[dataset_name], usr_prompts=prompts, tokenizer=tokenizer)
            outs = vllm_llm_generate(inputs, sampling_params, llm)
            for o in outs:
                out_texts.append(o.outputs[0].text.strip() if o.outputs else "")
        elif ds_client is not None:
            responses = get_ds_response(ds_client, ALIGN_SYS_PROMPT+examples_pool[dataset_name], prompts, temperature=0.3)
            for response in responses:
                out_texts.append(response.strip())
        else:
            raise ValueError("Either llm or ds_client must be provided for alignment.")
    return out_texts

# ---------------------------------------------------------------------------
# cot_to_save 构造（使用解析后的每步文本）
# ---------------------------------------------------------------------------

def make_final_cot(sample: Dict[str,Any], aligned_steps: List[str], aligned_final: str) -> List[Dict[str,Any]]:
    k = sample["first_correct_helper_idx"]
    helpers = sample.get("helpers", [])
    # system
    cot = [{"role":"system","content":[{"type":"text","text":"You are a helpful assistant. You can generate abstract visual tokens that represent a cropped image region or images with auxiliary information like lines, bounding boxes, etc. When you decide to generate abstract visual tokens, put them in <abs_vis_token>...</abs_vis_token>."}]}]
    # user
    uc = []
    if sample.get("image_main"):
        uc.append({"type":"image","image_file_name":sample["image_main"]})
    uc.append({"type":"text","text":sample["question"]})
    cot.append({"role":"user","content":uc})
    # assistant (aligned per-step)
    ac = []
    for i in range(k+1):
        if i < len(helpers):
            txt_al = aligned_steps[i].strip() if i < len(aligned_steps) else helpers[i].get("text","")
            if not txt_al:
                txt_al = helpers[i].get("text","")  # fallback
            ac.append({"type":"text","text":txt_al})
            ip = helpers[i].get("image_path")
            if ip:
                ac.append({"type":"image","image_file_name":ip})
    # final strong segment
    final_txt = aligned_final.strip()
    ac.append({"type":"text","text":final_txt})
    cot.append({"role":"assistant","content":ac})
    return cot

# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Stage3 align all steps -> per-step output")
    ap.add_argument("--stage2", required=True, help="Stage2 JSONL")
    ap.add_argument("--llm-path", help="alignment text-only 模型路径", default="")
    ap.add_argument("--devices", default="0,1,2,3", help="GPU IDs")
    ap.add_argument("--out-json", required=True, help="输出训练 JSON list")
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--align-batch", type=int, default=4096)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    recs = load_jsonl(args.stage2)
    if args.max_records is not None:
        recs = recs[:args.max_records]
    # 保留 strong 成功纠错样本
    good = []
    for r in recs:
        k = r.get("first_correct_helper_idx")
        if k is None: continue
        steps = r.get("strong_steps", [])
        if k >= len(steps): continue
        if not steps[k].get("correct", False): continue
        good.append(r)
    print(f"[Stage3] retained {len(good)} / {len(recs)} records")

    # 构造 full_texts + seg counts
    full_texts: List[str] = []
    seg_counts: List[int] = []
    for r in good:
        ft, nseg = build_alignment_text_for_sample(r)
        full_texts.append(ft)
        seg_counts.append(nseg)  # k+1 (helpers) + final

    # 模型
    llm = None
    tokenizer = None
    sampling_params = None
    ds_client = None
    
    if args.llm_path == "":
        ds_client = build_ds_client()
    else:
        print(f"[Stage3] loading alignment LLM: {args.llm_path}")
        llm, sampling_params = vllm_llm_init(args.llm_path)
        tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)

    # 批 alignment
    dataset_name = args.stage2.split("/")[-2]
    aligned_texts = batch_align_concat(dataset_name, full_texts, llm, sampling_params, tokenizer, batch_size=args.align_batch, ds_client=ds_client)
    assert len(aligned_texts) == len(good)

    # 解析 & 打包
    finals = []
    for rec, txt_al, nseg in zip(good, aligned_texts, seg_counts):
        k = rec["first_correct_helper_idx"]
        step_texts, final_txt = parse_aligned_text(txt_al, k)
        cot = make_final_cot(rec, step_texts, final_txt)
        finals.append(cot)

    save_json(finals, args.out_json)
    print(f"[Stage3] wrote {len(finals)} records -> {args.out_json}")

if __name__ == "__main__":
    main()
