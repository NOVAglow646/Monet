import os
from load_and_gen_vllm import vllm_mllm_init
mllm, sampling_params = vllm_mllm_init(
    mllm_dir="/data1/qxwang/checkpoints/Qwen2.5-VL-7B-Instruct",
    device="cuda:7"
)

s="Identify the problem. ### Step 2: Break down the problem. ### Step 3: Break down."
s0="### Step 1:"
s1="<|im_end|>"
s2="\n### Step x"
s3=" #### Step3: Break down."
s4="#### Step 4: Break down."
s5="  ### Step 4: Break down."
s6="\n### Step 3: Break down."
s7="\n\n### Step 3: Break down."
s8="### Step2"
s9=".### Step2"
s10="\n### Step2"
s11=" ### Step2"
tokenizer = mllm.get_tokenizer()
delim = "### Step "
delim_list = [tokenizer.encode(prefix+delim, add_special_tokens=False) for prefix in [""," ","\n", "#", " #"]]
print(delim_list)
print("0", tokenizer.encode(s0))
print("1", tokenizer.encode(s1))
print("2", tokenizer.encode(s2))
print("3", tokenizer.encode(s3))
print("4", tokenizer.encode(s4))
print("5", tokenizer.encode(s5))
print("6", tokenizer.encode(s6))
print("7", tokenizer.encode(s7))
print("8", tokenizer.encode(s8))
print("9", tokenizer.encode(s9))
print("10", tokenizer.encode(s10))
print("11", tokenizer.encode(s11))
print("####")

print(tokenizer.batch_encode_plus(
                        [s,s1,s2,s3,s4,s5], # list[str]
                        add_special_tokens=False,
                        padding=False,
                        return_attention_mask=False,
                    )["input_ids"] )