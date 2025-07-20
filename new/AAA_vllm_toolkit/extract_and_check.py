from AAA_vllm_toolkit.eval_utils.math_equal import math_equal
from AAA_vllm_toolkit.eval_utils.checker import check_one_answer
from AAA_vllm_toolkit.eval_utils.util import equiv, strip_string, choice_answer_clean
from math_evaluation import is_equiv
import re
from typing import Any, Dict, Type, Optional, List, Tuple, Union
from vllm import LLM, SamplingParams
from mathruler.grader import extract_boxed_content as mathruler_extract_boxed_content
from mathruler.grader import grade_answer as mathruler_grade_answer
from tqdm import tqdm

def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True

def extract_boxed_answer(text, debug=False):
    if text is None:
        return "Invalid prediction."
    start = text.rfind(r"\boxed{")
    if start == -1:
        return "Invalid prediction."
    end = None
    stack = []
    answer = text[start:]
    for i, c in enumerate(answer):
        if c == "{":
            stack.append(i)
        elif c == "}":
            start = stack.pop()  # \boxed start{
            if len(stack) == 0:
                end = i  # \boxed end}
                break
    if end is None and debug:
        print("Extract boxed answer: brack not closing, return None", answer)
        return "Invalid prediction."
    return answer[start + 1 : end]

'''def extract_no_boxed_answer(text, debug=False):
    if text is None:
        return None
    answer_indicators = ["answer is:", "answer:", "："]
    start = -1
    for answer_indicator in answer_indicators:
        if answer_indicator in text.lower():
            start = text.lower().rfind(answer_indicator)
            if start == -1:
                continue
            else:
                start = start + len(answer_indicator)
                break
    if start == -1:
        return None
    end = text.find("</answer>")
    if end !=1:
        return text[start:end]
    return text[start:]'''

def extract_html_answer(text: str):
    if text is None:
        return None
    start = text.find("<answer>")
    if start == -1:
        return None
    end = text.find("</answer>")
    if end == -1:
        return None
    return text[start + len("<answer>"):end].strip()


# 预编译正则，忽略大小写并允许跨行
_ANSWER_REGEX = re.compile(
    r"""                    # 用括号分组便于提取
    (?:                     # ===== 指示词 =====
        answer\s*(?:is)?\s*[:：] |  # Answer is: / Answer:
        ：                       # 纯中文冒号
    )
    \s*                      # 可有可无的空格
    (.*?)                    # ===== 真正答案（非贪婪）=====
    (?=</answer>|$)          # 直到 </answer> 或文本结尾
    """,
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)

def extract_no_boxed_answer(
    text: Union[str, None],
    debug: bool = False,
) -> Optional[str]:
    """
    从 LLM 输出中提取不带 \\boxed{} 的答案。

    - 支持多种答案指示词 (“Answer is:”, “Answer:”, “：”)
    - 取 **最后一次**出现的指示词之后的内容（与 rfind 语义一致）
    - 若出现 </answer> 标签，以此为终止边界
    """
    if not text:
        return None

    last_match = None
    for m in _ANSWER_REGEX.finditer(text):
        last_match = m  # 迭代到最后一次匹配

    if last_match is None:
        if debug:
            print("[extract_no_boxed_answer] 未找到答案指示词")
        return None

    # 提取并清理
    answer = last_match.group(1).strip()
    return answer or None




def rule_equiv(gt, pred, grt_choices = None): # grt_choices is a list that lists the ground truth choices if it's a multi-choice question
    # In this function, I integrated multiple open-source evaluation tools
    # each with its own judgment logic and strengths in handling special cases such as LaTeX, units, etc.
    gt = str(gt)
    pred = str(pred)
    try:
        if gt.strip().lower() == pred.strip().lower():
            return True
        
        pred_choice = None
        if pred.lower() in ["a", "b", "c", "d", "e", "f"]:
            pred_choice = pred[0]
        
        if pred[:2].lower() in ['a:', 'b:', 'c:', 'd:', 'e:', 'f:', 'a.', 'b.', 'c.', 'd.', 'e.', 'f.']: # for preds like "A: 1.414"
            pred_choice = pred[0]

        if pred[:3].lower() in ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']: # for preds like "(a) 1.414"
            pred_choice = pred[1]

            

        # If gt is not in ['A', 'B', 'C', 'D', 'E'] but pred is in ['A', 'B', 'C', 'D', 'E']
        if gt.lower() not in ['a', 'b', 'c', 'd', 'e', 'f'] and pred_choice is not None:
            choices = ['a', 'b', 'c', 'd', 'e', 'f']
            ground_truth_choice = choices[grt_choices.index(gt)]
            if ground_truth_choice.lower() == pred_choice.lower():
                return True
            
        # If gt is in ['A', 'B', 'C', 'D', 'E'] but pred is not in ['A', 'B', 'C', 'D', 'E']
        if gt.lower() in ['a', 'b', 'c', 'd', 'e', 'f'] and pred_choice is None and grt_choices is not None:
            choices = ['a', 'b', 'c', 'd', 'e', 'f']
            pred_choice = choices[grt_choices.index(pred)]
            #print("pred_choice", pred_choice)
            if gt.lower() == pred_choice.lower():
                return True

        # Check if both gt and pred are words (no numbers) and pred is a substring of gt
        if not any(char.isdigit() for char in gt) and not any(char.isdigit() for char in pred):
            if pred.strip().lower() in gt.strip().lower():
                return True

        # Check if gt or pred contains "√{*}" and convert to "\sqrt{*}"
        sqrt_pattern = r"√\{(.*?)\}"
        gt = re.sub(sqrt_pattern, r"\\sqrt{\1}", gt)
        pred = re.sub(sqrt_pattern, r"\\sqrt{\1}", pred)
            

        # For college-math and omni-math, the pred and gt positions need to be changed.
        # Because we found that the quality of ground truth in a small subset of problems within benchmarks like college-math is relatively low.
        if any(
            func(x, y) for func in [math_equal, is_equiv, check_one_answer] for x, y in [(gt, pred), (pred, gt)]
        ):
            return True
        # special for college-math, etc.
        gt_strip, pred_strip = strip_string(gt), strip_string(pred)
        if any(
            func(x, y) for func in [math_equal, is_equiv, check_one_answer] for x, y in [(gt_strip, pred_strip), (pred_strip, gt_strip)]
        ):
            return True

        # for choice question
        if gt in ["A", "B", "C", "D", "E"] and pred not in ["A", "B", "C", "D", "E"]:
            pred = choice_answer_clean(pred)
            if math_equal(gt, pred):
                return True
        elif is_multi_choice(gt) and not is_multi_choice(pred):
            pred = "".join(
                [c for c in pred if c in ["A", "B", "C", "D", "E"]]
            )
            if math_equal(gt, pred):
                return True
    except Exception as e:
        #print("maroi_equiv error")
        #print(e)
        pass
    return False


def remove_text_box(text):
    if text is None or text == "None":
        return None
    start = text.find(r"\text{")
    if start == -1:
        return text
    end = None
    stack = []
    answer = text[start:]
    for i, c in enumerate(answer):
        if c == "{":
            stack.append(i)
        elif c == "}":
            start_text = stack.pop()
            if len(stack) == 0:
                end_text = i
                break
    in_text_string = text[start + start_text + 1 : start + end_text]

    if in_text_string.strip() == "and":
        ex_text = text[:start] + text[start + end_text + 1 :]
    else:
        ex_text = (
            text[:start]
            + text[start + start_text + 1 : start + end_text].strip()
            + text[start + end_text + 1 :]
        )
    return ex_text.strip()

def batch_extract_answer(response_strs: List[str], use_ds_api=False, ds_api_args=None) -> List[str]:
    preds = []
    for i, response_str in tqdm(enumerate(response_strs), total=len(response_strs), desc="Extracting answers"):
        pred = remove_text_box(mathruler_extract_boxed_content(response_str))
        if pred is None:
            pred = remove_text_box(extract_no_boxed_answer(response_str))
        if pred is None and use_ds_api:
            pred = extract_answer_api(question=ds_api_args["all_questions"][i], response=response_str, client=ds_api_args["client"], verbose=False)
        preds.append(pred)
    return preds

def choice2text(pred: str, gt_choices: List[str]) -> str:
        """
        Convert a choice answer to text.
        """
        if pred in ["A", "B", "C", "D", "E"]:
            choice_id = ord(pred) - ord("A")
            if choice_id < len(gt_choices) and choice_id >= 0:
                return gt_choices[choice_id]
            return pred
        return pred

def batch_judge(preds, gts, gt_choicess=None, questions=None, llm: LLM = None, use_ds_api=False, ds_api_args=None) -> List[int]:
    '''
    Use mathruler_grade_answer, rule_equiv, and judge_llm to judge the correctness of predictions.
    '''
    tp = 1
    temperature = 0.0
    max_tokens = 1
    n_generate_sample = 1
    top_k = 50
    top_p = 0.95
    best_of = 1
    seed = 0
    stop = ["\n"]
    
    sampling_params = SamplingParams(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        #best_of=best_of,
        max_tokens=max_tokens,
        n=n_generate_sample,
        stop=stop,
        skip_special_tokens=False,
        seed=seed if temperature == 0 else None, # vllm0.6.6.post1
    )

    prompts = []
    pred_and_gts = []

    if questions is None:
        questions = [""] * len(preds)
    for i, (question, pred, gt) in enumerate(zip(questions, preds, gts)):
        sys_prompt = "You are given a question, a prediction and a ground truth. Please judge whether the prediction is correct or not. If the prediction is correct, please output '1', otherwise output '0'. Don't give any explanation."
        prompt = f"Question: {question}\n\nPrediction: {pred}\n\nGround Truth: {gt}\n\nYour output:"
        prompts.append(
            f"{sys_prompt}\n\n{prompt}"
        )
        if gt_choicess is not None:
            pred_and_gts.append((pred, gt, gt_choicess[i]))
        else:
            pred_and_gts.append((pred, gt, None))
        #reasoning_steps.append(reasoning_step)
        
    #print(f"Total prompts: {len(prompts)}")

    results = []
    if llm:
        llm_judge_responses = llm.generate(prompts, sampling_params=sampling_params)
    else:
        llm_judge_responses = [None] * len(prompts)
        
    for i, (llm_judge_resp, (pred, gt, gt_choices)) in tqdm(enumerate(zip(llm_judge_responses, pred_and_gts)), total=len(prompts), desc="Judging predictions"):
        # 每个 resp.outputs[0].text 应为 '1' 或 '0'
        if pred is None:
            results.append(0)
            continue
        choice2txt_pred = choice2text(pred, gt_choices) if gt_choices is not None else pred
        choice2txt_gt = choice2text(gt, gt_choices) if gt_choices is not None else gt
        
        if pred in ['A','B','C', 'D', 'E'] and gt not in ['A','B','C', 'D', 'E'] or pred not in ['A','B','C', 'D', 'E'] and gt in ['A','B','C', 'D', 'E']:
            pred = choice2txt_pred
            gt = choice2txt_gt
            
        if mathruler_grade_answer(pred, gt):
            results.append(1)
            continue
        elif is_equiv(gt, pred):
            results.append(1)
            continue
        elif llm_judge_resp is not None:
            text = llm_judge_resp.outputs[0].text.strip()
            if text == "1":
                results.append(1)
            else:
                results.append(0)
        elif use_ds_api:
            judgment = match_answer(gt, pred, ds_api_args['client'], verbose=False)
            if verify_judgment(judgment):
                results.append(int(judgment))
            else:
                results.append(0)
        else:
            results.append(0)
        #print(f"## Ground Truth: {gt}  Prediction: {pred}   Judge: {correctness}")

    return results


def quick_batch_judge(preds, gts):
    results = []
    for pred, gt in tqdm(zip(preds, gts), total=len(preds), desc="Quick judging predictions using mathruler"):
        if pred is None:
            results.append(0)
            continue
        if mathruler_grade_answer(gt, pred):
            results.append(1)
        elif mathruler_grade_answer(gt.lower(), pred.lower()):
            results.append(1)
        else:
            results.append(0)
    return results

def llm_batch_judge(preds, gts, llm, questions):
    tp = 1
    temperature = 0.0
    max_tokens = 1
    n_generate_sample = 1
    top_k = 50
    top_p = 0.95
    best_of = 1
    seed = 0
    stop = ["\n"]
    
    sampling_params = SamplingParams(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        #best_of=best_of,
        max_tokens=max_tokens,
        n=n_generate_sample,
        stop=stop,
        skip_special_tokens=False,
        seed=seed if temperature == 0 else None, # vllm0.6.6.post1
    )

    prompts = []
    results = []
    for i, (question, pred, gt) in enumerate(zip(questions, preds, gts)):
        sys_prompt = "You are given a question, a prediction and a ground truth. Please judge whether the prediction is correct or not. If the prediction is correct, please output '1', otherwise output '0'. Don't give any explanation."
        prompt = f"Question: {question}\n\nPrediction: {pred}\n\nGround Truth: {gt}\n\nYour output:"
        prompts.append(
            f"{sys_prompt}\n\n{prompt}"
        )
    llm_judge_responses = llm.generate(prompts, sampling_params=sampling_params)

    for i, (llm_judge_resp, pred, gt) in tqdm(enumerate(zip(llm_judge_responses, preds, gts)), total=len(prompts), desc="Judging predictions"):
        judge_text = llm_judge_resp.outputs[0].text.strip()
        if judge_text == "1":
            results.append(1)
        elif judge_text == "0":
            results.append(0)
        else:
            results.append(1 if pred.lower() == gt.lower() else 0)
    return results
    

#############################################################################################
# Deepseek api extract and check
#############################################################################################
sys_prompt = """
Imagine you are an intelligent teacher. \
Thoroughly read the provided instruction to ensure a solid understanding of the information provided.
"""

demo_prompt_extract = """
Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.
If the question requires a full sentence with a correct word filled in, please provide the word only.
If there is no answer in the model response, please return 'No answer'.

Question: There is a single rectangle with multiple color layers in the image. What is the color of the boundary of the rectangle? The answer should be one of ‘red’, ‘yellow’, ‘green’, or ‘blue’.
Model response: The color of the boundary of the circle is red.
Extracted answer: red

Question: How many line segments are in the image? Answer should be a number.
Model response: There are 4 dashed line segments in the image.
Extracted answer: 4

Question: Choose the word in parentheses that correctly describes the image. Rewrite the sentence with the chosen word.
In the image, shape (A/B) has sides curved inward.
Model response: In the image, shape B has sides curved inward.
Extracted answer: B

Question: In this image, choose the path which is a single line segment between points A and B from the following options. Provide your answer as a single uppercase letter: (A) the purple path (B) the blue path (C) the green path (D) the red path
Model response: B
Extracted answer: B

Question: Choose the most appropriate color to fill in the box marked with ‘?’ in the image. The answer is one of ‘a’, ‘b’, ‘c’, or ‘d’.
Model response: The correct color to fill in the box marked with '?' is (a) blue.\n\nThe colors are following a gradient pattern from red, to a more purple hue, and finally to blue. The logical next color in the sequence would be blue, as it extends the progression seen in the previous squares.
Extracted answer: a
"""


demo_prompt_score = """
The [Standard Answer] is the correct answer to the question, and the [Model Answer] is the answer generated by a model for that question.
Thoroughly read both the [Standard Answer] and the [Model Answer]. Assess the consistency of the information provided in these two responses.
Although you do not know the specific question, you can still assess the consistency between the two responses by checking for logical conflicts if both responses are assumed to be correct.
If the [Model Answer] is consistent with the [Standard Answer], please answer '1'. Otherwise, answer '0'.
Don't output any other information, just the number '0' or '1'.
Below are the examples of the correct consistency judgment.

[Standard Answer] a
[Model Answer] a
Judgment: 1

[Standard Answer] 1
[Model Answer] 4
Judgment: 0

[Standard Answer] circle
[Model Answer] the circle
Judgment: 1

[Standard Answer] line segment B and C
[Model Answer] B, C
Judgment: 1

[Standard Answer] 7.07
[Model Answer] 7.07 \\, \\text{cm/s}^2
Judgment: 1

[Standard Answer] three
[Model Answer] 3
Judgment: 1

[Standard Answer] decrease
[Model Answer] The deer tick population would decrease.
Judgment: 1

Now, below are two answers to a question. What is your judgment?
"""

def get_evaluation_chat_response(sys_prompt, user_prompt, client, temperature=0.7):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1024,
        temperature=0.7,
        stream=False
    )
    return response.choices[0].message.content

# Create a test prompt for the model to extract the answer
def create_test_prompt(demo_prompt, question, response):
    demo_prompt = demo_prompt.strip()
    test_prompt = f"Question: {question}\nModel response: {response}\nExtracted Answer: "
    full_prompt = f"{demo_prompt}\n\n{test_prompt}"
    return full_prompt

# Extract the answer from the model response
def extract_answer_api(question, response, client, verbose=False):
    try:
        test_prompt = create_test_prompt(demo_prompt_extract, question, response)
        extraction = get_evaluation_chat_response(sys_prompt, test_prompt, client)
        # Only extract the content after 'Extracted Answer:'
        if 'Extracted answer:' in extraction:
            return extraction.split('Extracted answer:')[-1].strip()
        # If the model does not provide the answer in an instructed format, return the whole response
        else:
            return extraction.strip()
    except Exception as e:
        print(e, verbose)
        print(f"Error in extracting answer for '{response}'")
    return ""


# Check if the judgment is in the correct format
def verify_judgment(judgment):
    judgment = judgment.strip()
    if judgment == None or judgment not in ['0', '1']:
        return False
    return True

# Create a test prompt for the model to score the answer
def create_test_prompt(demo_prompt, answer, extraction):
    demo_prompt = demo_prompt.strip()
    test_prompt = f"[Standard Answer] {answer}\n[Model Answer] {extraction}\njudgment: "
    full_prompt = f"{demo_prompt}\n\n{test_prompt}"
    return full_prompt

# Match the standard answer with the extracted answer
def match_answer(answer, extraction, client, verbose=False):
    # general extraction
    try:
        test_prompt = create_test_prompt(demo_prompt_score, answer, extraction)
        judgment = get_evaluation_chat_response(sys_prompt, test_prompt, client)
        # sometimes gpt may return 'judgment: 1' or 'judgment: 0'
        return judgment.lower().replace("judgment:", "").strip()
    except Exception as e:
        print(e, verbose)
        print(f"Error in matching answer:\n[Standard Answer] {answer}\n[Model Answer] {extraction}")
    return ""