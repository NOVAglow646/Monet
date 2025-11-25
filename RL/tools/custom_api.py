# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI
from tqdm import tqdm

import os
import json

from google import genai

gemini_generation_config = {"max_output_tokens": 9000, "temperature": 0.3, "top_p": 1.0}


def build_gemini_client():
    """Build a Gemini client using GOOGLE_API_KEY from environment.

    Expected env:
        GOOGLE_API_KEY: your Google AI Studio / Gemini API key.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set in environment.")
    client = genai.Client(api_key=api_key)
    return client


def get_gemini_response(client, sys_prompt, user_prompts, temperature=0.3, model_name="gemini-2.5-pro-exp-02"):
    responses = []
    for user_prompt in user_prompts:
        contents = [
            {"role": "user", "parts": [{"text": sys_prompt + "\n" + user_prompt}]}
        ]

        gen_cfg = dict(gemini_generation_config)
        gen_cfg["temperature"] = temperature
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                generation_config={
                    "max_output_tokens": gen_cfg["max_output_tokens"],
                    "temperature": gen_cfg["temperature"],
                    "top_p": gen_cfg["top_p"],
                },
            )
            text = "".join(part.text for part in response.candidates[0].content.parts)
        except Exception as e:
            text = f"[API calling error, at tools.custom_api.py:get_gemini_response] {str(e)}"
        responses.append(text)
    return responses

def build_deepseek_client():
    """
    Build a DeepSeek client with the specified API key and base URL.
    """
    return OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY', None), base_url="https://api.deepseek.com")

def get_deepseek_response(client, sys_prompt, user_prompts, temperature=0.3, model_name="deepseek-chat"):
    """
    Get responses from DeepSeek API for a list of user prompts.
    """
    model = model_name
    responses = []
    '''for user_prompt in tqdm(
        user_prompts,
        desc="Processing user prompts using DeepSeek api",
        total=len(user_prompts),
    ):'''
    for user_prompt in user_prompts:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            stream=False,
        )
        responses.append(response.choices[0].message.content)
    return responses

def get_api_response(api_model_name, sys_prompt, user_prompts, client=None, temperature=0.3):
    if api_model_name == "gemini-2.5-pro":
        if client is None:
            client = build_gemini_client()
        return get_gemini_response(client, sys_prompt, user_prompts, temperature)
    elif api_model_name in ["deepseek-chat", "deepseek-reasoner"]:
        if client is None:
            client = build_deepseek_client()
        return get_deepseek_response(client, sys_prompt, user_prompts, temperature, model_name=api_model_name)
    else:
        raise ValueError(f"Unsupported API model name: {api_model_name}")