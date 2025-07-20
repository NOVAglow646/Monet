# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI
from tqdm import tqdm


def build_ds_client():
    """
    Build a DeepSeek client with the specified API key and base URL.
    """
    return OpenAI(api_key="sk-a62bc7c0899a47dba605e3d3ab332e37", base_url="https://api.deepseek.com")

def get_ds_response(client, sys_prompt, user_prompts, temperature=0.3):
    model = "deepseek-chat"
    responses = []
    for user_prompt in tqdm(user_prompts, desc="Processing user prompts using deepseek api", total=len(user_prompts)):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            stream=False
        )
        responses.append(response.choices[0].message.content)
    return responses