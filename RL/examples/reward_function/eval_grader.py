from mathruler.grader import extract_boxed_content, grade_answer
from verl.workers.rollout.utils.math_equal import math_equal
from math_evaluation import is_equiv
print(grade_answer(r"Minutes", "minutes"))
print(math_equal(r"Minutes", "minutes"))
print(is_equiv(r"Minutes", "minutes"))

