# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI
from tqdm import tqdm

# TAG shiyang
import os
from glob import glob
import cv2
import json
import base64
import time
import math
import traceback
import numpy as np
import sys
import tqdm

# from moviepy.editor import VideoFileClip

# os.environ['GOOGLE_APPLICATION_CREDENTIALS']='/pfs/gaohuan03/gemini_exp/mmu-gemini-2test-52d3c3234a01.json'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
    "/pfs/gaohuan03/gemini_exp/mmu-gemini-caption-1-5pro-86ec97219196.json"
)
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models

gemini_generation_config = {"max_output_tokens": 9000, "temperature": 0.3, "top_p": 1.0}
safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.OFF,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.OFF,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.OFF,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.OFF,
}
vertexai.init(project="mmu-gemini-caption-1-5pro", location="us-central1")


def build_ds_client():
    """
    Build a DeepSeek client with the specified API key and base URL.
    """
    # return OpenAI(api_key="sk-a62bc7c0899a47dba605e3d3ab332e37", base_url="https://api.deepseek.com")
    gemini_model = GenerativeModel("gemini-2.5-pro")  # NOTICE
    return gemini_model


def get_ds_response(client, sys_prompt, user_prompts, temperature=0.3):
    # model = "deepseek-chat"
    gemini_model = client
    responses = []
    for user_prompt in tqdm(
        user_prompts,
        desc="Processing user prompts using Gemini api",
        total=len(user_prompts),
    ):
        # response = client.chat.completions.create(
        #     model=model,
        #     messages=[
        #         {"role": "system", "content": sys_prompt},
        #         {"role": "user", "content": user_prompt},
        #     ],
        #     temperature=temperature,
        #     stream=False,
        # )
        contents = []
        contents.append(user_prompt)
        # responses.append(response.choices[0].message.content)
        response = gemini_model.generate_content(
            contents,
            generation_config=gemini_generation_config,
            safety_settings=safety_settings,
        )
        response = responses.text
        responses.append(response)
    return responses
