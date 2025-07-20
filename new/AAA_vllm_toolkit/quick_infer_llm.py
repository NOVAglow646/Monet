model_path = '/data0/huggingface/Qwen/Qwen2.5-14B-Instruct'
from load_and_gen_vllm import *
from load_and_gen_hf import *
llm, sampling_params = vllm_llm_init(model_path)
inst_prompt = (
    "Remove tool callings like 'Based on GROUNDING(...)', 'use LINE([64, 245, 209, 266], 1, 2)->img_1.', 'Leveraging OCR(texts in image `img_1`)', etc. while keeping the original meaning of the text.\n\n"
)

demos_com_math=(
    "Here are some examples:\n\n"
    "Input: Then, use LINE([64, 245, 209, 266], 1, 2)->img_1 to find the 4 in the image, obtaining the new image `img_1`."
    "Your output: Then, find the 4 in the image, obtaining the new image `img_1`.\n\n"
    "Input: Then, use GROUNDING(the intersection of the Line with the horizontal axis) to locate the position of the intersection points of the Line on the image with the horizontal axis, which are at `bbx_4`, `bbx_5`."
    "Your output: Then, locate the position of the intersection points of the Line on the image with the horizontal axis, which are at `bbx_4`, `bbx_5`.\n\n"
    "Input: Draw the leftmost vertical Line of the nail with LINE([69, 0, 75, 157],1,4)->img_1, to obtain a new image `img_1` after drawing the line."
    "Your output: Draw the leftmost vertical Line of the nail, to obtain a new image `img_1` after drawing the line.\n\n"
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

usr_prompts = [
    inst_prompt + demos_com + query_prompt
]
tokenizer = load_tokenizer(model_path)
inputs = vllm_llm_process_batch_data(sys_prompt=sys_prompt, usr_prompts=usr_prompts, tokenizer=tokenizer)
output = vllm_llm_generate(inputs, sampling_params, llm)