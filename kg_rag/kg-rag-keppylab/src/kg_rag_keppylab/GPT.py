"""
This file contains the functions to interact with the OpenAI API.
The fuction disease_entity_extractor_GPT is used to extract disease entities from a given text.
The function fetch_GPT_response is used to call the OpenAI API and get the response--it is decorated with the retry decorator to handle network errors.
The function get_GPT_response is a memoized version of fetch_GPT_response, which stores the response in memory to avoid repeated calls to the OpenAI API.
"""

import os
import sys

from kg_rag.config_loader import system_prompts, config_data

def disease_entity_extractor_GPT(text):
    chat_model_id, chat_deployment_id = "GPT-3.5-Turbo", "GPT-3.5-Turbo"
    prompt_updated = system_prompts["DISEASE_ENTITY_EXTRACTION"] + "\n" + "Sentence : " + text
    resp = get_GPT_response(prompt_updated, system_prompts["DISEASE_ENTITY_EXTRACTION"], chat_model_id, chat_deployment_id, temperature=0)
    try:
        entity_dict = json.loads(resp)
        return entity_dict["Diseases"]
    except:
        return None

@retry(wait=wait_random_exponential(min=10, max=30), stop=stop_after_attempt(5))
def fetch_GPT_response(instruction, system_prompt, chat_model_id, chat_deployment_id, temperature=0):
    # print('Calling OpenAI...')
    response = openai.ChatCompletion.create(
        temperature=temperature,
        deployment_id=chat_deployment_id,
        model=chat_model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction}
        ]
    )
    if 'choices' in response \
       and isinstance(response['choices'], list) \
       and len(response) >= 0 \
       and 'message' in response['choices'][0] \
       and 'content' in response['choices'][0]['message']:
        return response['choices'][0]['message']['content']
    else:
        return 'Unexpected response'

@memory.cache
def get_GPT_response(instruction, system_prompt, chat_model_id, chat_deployment_id, temperature=0):
    return fetch_GPT_response(instruction, system_prompt, chat_model_id, chat_deployment_id, temperature)
