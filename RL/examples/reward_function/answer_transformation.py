import re

sqrt_pattern = r"√(?:\\*\{|\()([^{}()]+)(?:\\*\}|\))"

def answer_transformation_fn(
    ans: str,  # The answer to be transformed
) -> bool:
    
    # Check if gt or pred contains "√{*}" and convert to "\sqrt{*}"
    ans = re.sub(sqrt_pattern, r"\\sqrt{\1}", ans)
    return ans
