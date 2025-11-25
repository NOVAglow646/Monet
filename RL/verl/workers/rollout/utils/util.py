import pprint
import re
from word2number import w2n
from __future__ import annotations
from typing import List, Dict, Any, Optional, Type, Tuple, Union
#from math_evaluation import is_equiv
from verl.workers.rollout.utils.math_equal import math_equal
from verl.workers.rollout.utils.checker import check_one_answer
from verl.workers.rollout.utils.util import equiv, strip_string, choice_answer_clean
import re


def convert_word_number(text: str) -> str:
    try:
        text = str(w2n.word_to_num(text))
    except:
        pass
    return text

def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    return _string

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

unit_texts = [
    "east",
    "degree",
    "mph",
    "kmph",
    "ft",
    "m sqaure",
    " m east",
    "sq m",
    "deg",
    "mile",
    "q .",
    "monkey",
    "prime",
    "ratio",
    "profit of rs",
    "rd",
    "o",
    "gm",
    "p . m",
    "lb",
    "tile",
    "per",
    "dm",
    "lt",
    "gain",
    "ab",
    "way",
    "west",
    "a .",
    "b .",
    "c .",
    "d .",
    "e .",
    "f .",
    "g .",
    "h .",
    "t",
    "a",
    "h",
    "no change",
    "men",
    "soldier",
    "pie",
    "bc",
    "excess",
    "st",
    "inches",
    "noon",
    "percent",
    "by",
    "gal",
    "kmh",
    "c",
    "acre",
    "rise",
    "a . m",
    "th",
    "π r 2",
    "sq",
    "mark",
    "l",
    "toy",
    "coin",
    "sq . m",
    "gallon",
    "° f",
    "profit",
    "minw",
    "yr",
    "women",
    "feet",
    "am",
    "pm",
    "hr",
    "cu cm",
    "square",
    "v â € ™",
    "are",
    "rupee",
    "rounds",
    "cubic",
    "cc",
    "mtr",
    "s",
    "ohm",
    "number",
    "kmph",
    "day",
    "hour",
    "minute",
    "min",
    "second",
    "man",
    "woman",
    "sec",
    "cube",
    "mt",
    "sq inch",
    "mp",
    "∏ cm ³",
    "hectare",
    "more",
    "sec",
    "unit",
    "cu . m",
    "cm 2",
    "rs .",
    "rs",
    "kg",
    "g",
    "month",
    "km",
    "m",
    "cm",
    "mm",
    "apple",
    "liter",
    "loss",
    "yard",
    "pure",
    "year",
    "increase",
    "decrease",
    "d",
    "less",
    "Surface",
    "litre",
    "pi sq m",
    "s .",
    "metre",
    "meter",
    "inch",
]

def is_float(s):
    if s is None:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def extract_only_number(s):
    return "".join(filter(lambda x: x.isdigit() or x in [".", "-"], s))

def remove_angle_brackets(text):
    """for im_token"""
    if text is None:
        return None
    while text.find(r"<") != -1 and text.find(r">") != -1:
        start = text.find(r"<")
        # if start == -1:
        #     return text
        end = None
        stack = []
        answer = text[start:]
        end_text = len(answer)
        for i, c in enumerate(answer):
            if c == "<":
                stack.append(i)
            elif c == ">":
                start_text = stack.pop()  # <
                if len(stack) == 0:
                    end_text = i  # >
                    break
        text = text[:start] + text[start + end_text + 1 :]
    return text.strip()

def last_boxed_only(sample):
    q, a = sample
    a = last_boxed_only_string(a)
    if a == None:
        return None
    return (q, a)

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval

def only_until_first_boxed_from_tokens(string, tokens):
    idx = string.find("\\boxed")
    if idx < 0:
        idx = string.find("\\fbox")
        if idx < 0:
            return None
    
    cum_length = 0
    for i, t in enumerate(tokens):
        cum_length += len(t)
        if cum_length >= idx:
            break
    
    return tokens[:i]



def clean_numbers(sample):
    if not sample:
        return None
    new_sample = list()
    for s in sample:
        new_sample.append(_clean_numbers(s))

    return tuple(new_sample)

def _clean_numbers(string):
    """
    Clean Numbers in the given string

    >>> _clean_numbers(None, "Hello 123")
    'Hello 123'
    >>> _clean_numbers(None, "Hello 1234")
    'Hello 1,234'
    >>> _clean_numbers(None, "Hello 1234324asdasd")
    'Hello 1,234,324asdasd'
    """
    num_prev_digits = 0
    new_string = ""
    for i, c in enumerate(string):
        # isdigit() doesnt work here because of weird unicode chars.
        if c in {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0'}:
            num_prev_digits += 1
        else:
            if num_prev_digits > 3:
                # Some fixing
                string_number = new_string[-num_prev_digits:]
                new_string = new_string[:-num_prev_digits] + "{0:,}".format(int(string_number))
            num_prev_digits = 0
        new_string += c

    if num_prev_digits > 3:
        # Some fixing
        string_number = new_string[-num_prev_digits:]
        new_string = new_string[:-num_prev_digits] + "{0:,}".format(int(string_number))

    return new_string

def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string

def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


# def strip_string(string):
#     # linebreaks
#     string = string.replace("\n", "")

#     # remove inverse spaces
#     string = string.replace("\\!", "")

#     # replace \\ with \
#     string = string.replace("\\\\", "\\")

#     # replace tfrac and dfrac with frac
#     string = string.replace("tfrac", "frac")
#     string = string.replace("dfrac", "frac")

#     # remove \left and \right
#     string = string.replace("\\left", "")
#     string = string.replace("\\right", "")

#     # Remove circ (degrees)
#     string = string.replace("^{\\circ}", "")
#     string = string.replace("^\\circ", "")

#     # remove dollar signs
#     string = string.replace("\\$", "")

#     # remove units (on the right)
#     string = remove_right_units(string)

#     # remove percentage
#     string = string.replace("\\%", "")
#     string = string.replace("\%", "")  # noqa: W605

#     # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
#     string = string.replace(" .", " 0.")
#     string = string.replace("{.", "{0.")
#     # if empty, return empty string
#     if len(string) == 0:
#         return string
#     if string[0] == ".":
#         string = "0" + string

#     # to consider: get rid of e.g. "k = " or "q = " at beginning
#     if len(string.split("=")) == 2:
#         if len(string.split("=")[0]) <= 2:
#             string = string.split("=")[1]

#     # fix sqrt3 --> sqrt{3}
#     string = fix_sqrt(string)

#     # remove spaces
#     string = string.replace(" ", "")

#     # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
#     string = fix_fracs(string)

#     # manually change 0.5 --> \frac{1}{2}
#     if string == "0.5":
#         string = "\\frac{1}{2}"

#     # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
#     string = fix_a_slash_b(string)

#     return string

def strip_string(string, skip_unit=False):
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    # replace \\ with \
    string = string.replace("\\!", "")
    # string = string.replace("\\ ", "")
    # string = string.replace("\\\\", "\\")

    # matrix
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = (
        string.replace("\\neq", "\\ne")
        .replace("\\leq", "\\le")
        .replace("\\geq", "\\ge")
    )

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("\\{", "{")
    string = string.replace("\\}", "}")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    if not skip_unit:
        # Remove unit: texts
        for _ in range(2):
            for unit_text in unit_texts:
                # use regex, the prefix should be either the start of the string or a non-alphanumeric character
                # the suffix should be either the end of the string or a non-alphanumeric character
                _string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
                if _string != "":
                    string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = string.replace("\\(", "").replace("\\)", "")

    # convert word number to digit
    string = convert_word_number(string)

    # replace "\\text{...}" to "..."
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    for key in ["x=", "y=", "z=", "x\\in", "y\\in", "z\\in", "x\\to", "y\\to", "z\\to"]:
        string = string.replace(key, "")
    string = string.replace("\\emptyset", r"{}")
    string = string.replace("(-\\infty,\\infty)", "\\mathbb{R}")

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # cdot
    # string = string.replace("\\cdot", "")
    if (
        string.startswith("{")
        and string.endswith("}")
        and string.isalnum()
        or string.startswith("(")
        and string.endswith(")")
        and string.isalnum()
        or string.startswith("[")
        and string.endswith("]")
        and string.isalnum()
    ):
        string = string[1:-1]

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and
    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string.replace("'", "")
    string.replace('"', "")

    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r"(\d+)\.0*([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0*$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        #pdb.set_trace()
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2

class NotEqual:
    def __eq__(self, other):
        return False
    
    
direct_answer_trigger_for_fewshot = ("choice is", "answer is")
def choice_answer_clean(pred: str):
    pred = pred.strip("\n")

    # Determine if this is ICL, if so, use \n\n to split the first chunk.
    ICL = False
    for trigger in direct_answer_trigger_for_fewshot:
        if pred.count(trigger) > 1:
            ICL = True
    if ICL:
        pred = pred.split("\n\n")[0]

    # Split the trigger to find the answer.
    preds = re.split("|".join(direct_answer_trigger_for_fewshot), pred)
    if len(preds) > 1:
        answer_flag = True
        pred = preds[-1]
    else:
        answer_flag = False

    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")

    # Clean the answer based on the dataset
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]

    if len(pred) == 0:
        pred = ""
    else:
        if answer_flag:
            # choose the first element in list ...
            pred = pred[0]
        else:
            # choose the last e
            pred = pred[-1]

    # Remove the period at the end, again!
    pred = pred.rstrip(".").rstrip("/")

    return pred




def remove_text_box(text: str | None) -> str | None:
    if text is None:
        return None

    start = text.find(r"\text{")
    if start == -1:
        return text.strip()

    # ---------- 初始化哨兵 ----------
    start_text = end_text = None
    stack = []
    answer = text[start:]

    for i, c in enumerate(answer):
        if c == "{":
            stack.append(i)
        elif c == "}":
            if not stack:          # 括号不匹配，直接返回原串
                return text.strip()
            start_text = stack.pop()
            if not stack:          # 最外层 '}' 配对完成
                end_text = i
                break

    # ---------- 若没匹配成功，直接返回原串 ----------
    if start_text is None or end_text is None:
        return text.strip()

    in_text_string = text[start + start_text + 1 : start + end_text]
    if in_text_string.strip() == "and":
        ex_text = text[:start] + text[start + end_text + 1 :]
    else:
        ex_text = (
            text[:start]
            + in_text_string.strip()
            + text[start + end_text + 1 :]
        )
    return ex_text.strip()



def extract_boxed_answer(text, debug=False):
    if text is None:
        return None
    start = text.rfind(r"\boxed{")
    if start == -1:
        return text
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
        print("brack not closing", answer)
        return None
    return answer[start + 1 : end]

def extract_no_boxed_answer(text, debug=False):
    if text is None:
        return None
    start = -1
    answer_indicators = ["is"]
    for answer_indicator in answer_indicators:
        if answer_indicator in text.lower():
            start = text.lower().rfind(answer_indicator)
            if start == -1:
                continue
            else:
                start = start + len(answer_indicator)
                break
    
    end = text.find("</answer>")
    if start == -1:
        return 'None'
    if end !=1:
        return text[start:end]
    return text[start:]

INVALID_ANS = "[invalid]"



def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def remove_single_dollar(s):
    if not s:
        return s
    if isinstance(s, list):
        s = s[0]
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1]
    return s


def any_condition(conditions):
    return any(conditions)


def rstar_equiv(gt, pred, grt_choices = None): # grt_choices is a list that lists the ground truth choices if it's a multi-choice question
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
        if gt.lower() not in ['a', 'b', 'c', 'd', 'e', 'f'] and pred_choice is not None and grt_choices is not None:
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
            func(x, y) for func in [math_equal, check_one_answer] for x, y in [(gt, pred), (pred, gt)]
        ):
            return True
        # special for college-math, etc.
        gt_strip, pred_strip = strip_string(gt), strip_string(pred)
        if any(
            func(x, y) for func in [math_equal, check_one_answer] for x, y in [(gt_strip, pred_strip), (pred_strip, gt_strip)]
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
        

def math_equiv(grt: Union[str, list[str]], prd: str, grt_choice = None):
    prd = (prd)
    if isinstance(grt, list):
        for g in grt:
            if rstar_equiv(g, prd, grt_choice):
                return True
        return False
    else:
        return rstar_equiv(grt, prd, grt_choice)


def truncate_prompt(tokenizer, prompts, max_input_len):
    encoded_batch = tokenizer(
        prompts,
        truncation=True,
        max_length=max_input_len,
        padding=False, # whether to padding to the same_size
        return_tensors=None
    )
    input_ids_list = encoded_batch["input_ids"]
    return [tokenizer.decode(ids) for ids in input_ids_list]


def on_annotated_path(node_tag):
    if node_tag == '0':
        return True
    ids = node_tag.split('.')[1:]
    for i in ids:
        if i!='1':
            return False
    return True
    
def extract_and_check(response_str: str, gt: str) -> bool:
    # Extract the answer from the response string
    answer = remove_text_box(extract_boxed_answer(response_str))
    if answer is None:
        answer = remove_text_box(extract_no_boxed_answer(response_str))
    if answer is None:
        return False
    
    return rstar_equiv(gt, answer)