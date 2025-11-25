from typing import List, Optional, Dict, Tuple
from tools.custom_api import get_api_response
import traceback 
import time
import pdb
def judge_wrap_fn(pred: Optional[str], gt: Optional[str], question: Optional[str], repetition_penalty: bool=False) -> Tuple[str, str]:
    if not repetition_penalty:
        sys_prompt = (
            "You are a strict answer judge. Given the question, a model's predicted answer, and the ground-truth answer, "
            "determine if the prediction is correct. Consider semantic equivalence, case/format variations, "
            "and numeric equivalence if applicable. Only reply with 'yes' or 'no'."
        )
        user_prompt = (
            f"Question: {question if question is not None else ''}\n"
            f"Predicted Answer: {pred if pred is not None else ''}\n"
            f"Ground Truth Answer: {gt if gt is not None else ''}\n"
            "Does the predicted answer exactly or semantically match the ground-truth? Reply 'yes' or 'no'."
        )
    else:
        sys_prompt = (
            "You are a strict answer judge. Given the question, a model's predicted answer, and the ground-truth answer, you should:\n"
            "1. Determine if the prediction is correct. Consider semantic equivalence, case/format variations, "
            "and numeric equivalence if applicable. If the prediction is correct, reply with '1'.\n"
            "2. If the prediction is incorrect, then determine if the prediction contains repeatedly illogical contents. Here are two examples:\n" 
            "Example (1) 'First, observe the pattern in the top row of the image.  The pattern in the top row is  increasing by one row each time.  The pattern in the bottom row is  increasing by one column each time.  The pattern in the bottom row is  increasing by one column each time.  The pattern in the bottom row is  increasing by one column each time. ...'\n" 
            "Example (2) 'First, observe the pattern in the top row of the provided image.  The pattern in the top row is  \boxed{A}.  The pattern in the bottom row is  \boxed{D}.  The pattern in the middle row is  \boxed{B}.  The pattern in the bottom row is  \boxed{C}.  The pattern in the middle row is  \boxed{A}.  The pattern in the bottom row is  \boxed{D}.  The pattern in the middle row is  \boxed{B}. ...'\n"
            "If the prediction doesn't contain such contents, reply with '0'. Else, reply with '-1'.\n"
            "Remember, you are only allowed to output '1', '0', or '-1', do not output anything else." 
        )
        user_prompt = (
            f"Question: {question if question is not None else ''}\n"
            f"Predicted Answer: {pred if pred is not None else ''}\n"
            f"Ground Truth Answer: {gt if gt is not None else ''}\n"
            "Your output: "
        )
    return sys_prompt, user_prompt



def _api_call_wrapper(
    api_name: str,
    pred: Optional[str],
    gt: Optional[str],
    question: Optional[str],
    dataset_name: str,
    client=None,
    api_kwargs: Optional[dict] = None,
    repetition_penalty: bool = False
) -> Optional[bool]:
    """
    Execute API-based judging with up to 3 attempts.
    - The retry loop is outside; each attempt is wrapped in try/except.
    - On exception: continue to the next attempt.
    - If a valid textual response is obtained:
        * Return True if it clearly says "yes" (and not "no").
        * Return False if it clearly says "no" (and not "yes") or is ambiguous/empty.
    - If all attempts fail to produce a valid response, return None to let upstream keep prior result.
    """
    # Fast-fail for missing required fields to avoid wasting API quota
    if pred is None or gt is None or str(pred).strip() == "":
        return False

    attempts = 5
    for atpt in range(attempts):
        try:
            # Build prompts per attempt (cheap and safe if something failed previously)
            sys_prompt, user_prompt = judge_wrap_fn(pred, gt, question, repetition_penalty)

            # Call external API
            #pdb.set_trace()
            #print("################ Before")
            responses = get_api_response(
                api_name, sys_prompt, [user_prompt],
                client=client, **(api_kwargs or {})
            )
            #print("Responses", responses)
            # Validate response
            if responses and isinstance(responses[0], str) and responses[0].strip():
                t = responses[0].strip().lower()
                if not repetition_penalty:
                    if "yes" in t and "no" not in t:
                        return 1.0
                    if "no" in t and "yes" not in t:
                        return 0.0
                    # Ambiguous or mixed content -> treat as incorrect
                    print(f"Neither 'yes' nor 'no' in the API response. Will set the judgment to be incorrect. The API response is: {responses[0]}")
                    return 0.0
                else:
                    if "1" in t and "0" not in t and "-1" not in t:
                        return 1.0
                    elif "0" in t and "1" not in t and "-1" not in t:
                        return 0.0
                    elif "-1" in t and "0" not in t:
                        pred_partial=pred[:1500]
                        print(f"[Repetitive pred]={pred_partial}...")
                        return -1.0
                    print(f"Invalid API response. Will set the judgment to be incorrect. The API response is: {responses[0]}")
                    return 0.0
            # If response is empty/invalid, just try next attempt
            print(f"Failed to obtain valid API judgement. Will retry for the {atpt+1} time...")
            if isinstance(responses[0], str):
                print(f"The API response is: {responses[0]}")

            continue

        except Exception as e:
            # Print stack trace for debugging and continue retrying
            traceback.print_exc()
            #print("######################################################")
            print(f"{e}")
            continue

    # All attempts failed to yield a usable response -> let caller decide
    return None

def _strip_boxed_instruction(q: str) -> str:
    if not isinstance(q, str):
        return q
    return (
        q.replace("Put the letter of your choice within \\boxed{}.", "")
        .replace("Put your final answer within \\boxed{}.", "")
        .replace("Given the answer in a single word and put it within \\boxed{}.", "")
        .strip()
    )

def api_batch_judge(
    questions: List[Optional[str]],
    preds: List[Optional[str]],
    gts: List[Optional[str]],
    *,
    api_name: Optional[str] = 'gemini-2.5-pro',
    api_max_workers: int = 32,
    api_kwargs: Optional[Dict] = None,
    client = None,
    dataset_name: str = "",
    repetition_penalty: bool = False
) -> List[int]:
    """
    API-only judging:
    - For each (question, pred, gt), call an external API (e.g., gemini-2.5-pro / deepseek-chat)
      to decide whether pred matches gt. The API is prompted to answer strictly 'yes' or 'no'.
    - Parallelized with ProcessPoolExecutor; falls back to 0 (incorrect) on API failure or ambiguity.
    - This function depends on `_api_call_wrapper` defined earlier in your module.

    Args:
        questions: List of questions; items can be None.
        preds: List of model predictions; items can be None.
        gts: List of ground-truth answers; items can be None.
        api_name: Name of API; if None, will read from env `API_JUDGE_NAME`.
        api_max_workers: Max parallel workers; can be overridden by env `API_JUDGE_WORKERS`.
        api_kwargs: Extra kwargs passed to the API caller (e.g., temperature, model_name).
        dataset_name: Optional dataset name (passed through to the wrapper).

    Returns:
        List[int]: 0/1 flags per sample; 1 = judged correct, 0 = incorrect or undecidable.
    """
    import os
    import concurrent.futures as cf
    import traceback
    start_time = time.time()

    if not (len(questions) == len(preds) == len(gts)):
        raise ValueError("Length mismatch: `questions`, `preds`, and `gts` must have the same length.")

    n = len(preds)
    results: List[int] = [0] * n  # default to incorrect

    # Strip boxed instruction if helper exists; otherwise use original questions.
    try:
        questions_wo_inst = [_strip_boxed_instruction(q) for q in questions]
    except NameError:
        questions_wo_inst = questions

    try:
        max_workers = int(os.environ.get("API_JUDGE_WORKERS", api_max_workers))
    except Exception:
        max_workers = api_max_workers
    
    # Prepare and launch parallel API judging
    #pdb.set_trace()
    with cf.ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = []
        for i in range(n):
            fut = ex.submit(
                _api_call_wrapper,  # must be available in the module scope
                api_name,
                preds[i],
                gts[i],
                questions_wo_inst[i],
                dataset_name,
                client=client,
                api_kwargs=api_kwargs,
                repetition_penalty=repetition_penalty
            )
            futs.append((i, fut))

        for i, fut in futs:
            try:
                r = fut.result()  # Optional[bool]: True/False/None
                # Treat None (all retries failed or exception) as incorrect (0)
                results[i] = r
            except Exception:
                traceback.print_exc()
                print("WARNING: API judge fail, set the correctness to be 0")
                results[i] = 0
    end_time = time.time()
    print(f"[api_batch_judge] Completed {n} samples in {end_time - start_time:.2f} seconds using API '{api_name}'")
    return results
