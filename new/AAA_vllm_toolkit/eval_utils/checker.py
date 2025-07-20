# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from .xwin_latex_answer_check import (
    latex_answer_check,
)
from .util import *

def extract_boxed_answer(text, debug=False):
    start = text.find(r"answer{")
    if start == -1:
        start = text.find(r"boxed{")
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

def remove_text_box(text):
    if text is None:
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
            start_text = stack.pop()  # \text start{
            if len(stack) == 0:
                end_text = i  # \text end}
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

def check_answer(ex_answer, gt_answer, is_remove_angle_brackets=False):
    if ex_answer and is_remove_angle_brackets:
        ex_answer = remove_angle_brackets(ex_answer)
    flag = equiv(ex_answer, gt_answer) or latex_answer_check(
        ex_answer, gt_answer, None, None, eval_policy="aggressive"
    )
    return flag

def check_one_answer(ex_answer, gt_answer):
    flag = False

    flag = check_answer(
        ex_answer, gt_answer, is_remove_angle_brackets=False
    )

    if not flag:
        ex_answer = remove_text_box(extract_boxed_answer(ex_answer))
        flag = check_answer(ex_answer, gt_answer)
        
    if not flag:
        def extract_answer_from_res(text):
            if ':' in text:
                return text.split(':')[-1].strip()
            elif 'is:' in text:
                return text.split('is:')[-1].strip()
            elif 'is' in text:
                return text.split('is')[-1].strip()
        ex_program_answer = extract_answer_from_res(ex_answer)
        if ex_program_answer is not None:
            flag = check_answer(ex_program_answer, gt_answer)               
    return flag