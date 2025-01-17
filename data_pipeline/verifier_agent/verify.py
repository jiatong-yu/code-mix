import asyncio
import base64
import os
import subprocess
import traceback
from datetime import datetime, timedelta
from enum import StrEnum
from functools import partial
from pathlib import PosixPath
from typing import cast
import textwrap
import json

import httpx
# import streamlit as st
from anthropic import RateLimitError
from anthropic.types.beta import (
    BetaContentBlockParam,
    BetaTextBlockParam,
)
# from streamlit.delta_generator import DeltaGenerator

from .loop import (
    PROVIDER_TO_DEFAULT_MODEL_NAME,
    APIProvider,
    sampling_loop,
)
from .tools import ToolResult
from .utils import COLOR_BOOK, print_colored, process_text_with_json

# SYSTEM_SUFFIX = """The user will provide you a JSON format through which you can express your final answer. When you decide to stop using tools and provide your final answer to the user's question, you should provide your answer through **ONLY** a JSON object and nothing else.
# """

VERBOSE = True
class Sender(StrEnum):
    USER = "user"
    AGENT = "assistant"
    TOOL = "tool"

class VerifierAgent:
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.max_tokens = 4096
        self.results = []
        self.log = []
        self.total_cost = 0
        self.provider = APIProvider.ANTHROPIC
        self.model_name = PROVIDER_TO_DEFAULT_MODEL_NAME[self.provider]
        self.set_up_tools()
        self.verifier_prompt = None
    
    def set_up_tools(self):
        os.environ['WIDTH'] = str(1920)
        os.environ['HEIGHT'] = str(1080)
    
    def get_cost(self):
        return self.total_cost

    async def verify_code_local(self, 
                                kwargs: dict
                                ):
        try:
            content = self.verifier_prompt.format(**kwargs)
        except:
            print(f"Error in formatting the prompt with kwargs: {kwargs}")
        messages = [(
            {
                "role": Sender.USER,
                "content": [BetaTextBlockParam(type="text", text=content)],
            }
        )]
        messages, cost = await sampling_loop(
            system_prompt_suffix="",
            model=self.model_name,
            provider=self.provider,
            messages=messages,
            output_callback=_output_callback,
            tool_output_callback=_tool_output_callback,
            api_response_callback=_api_response_callback,
            api_key = self.api_key,
            max_tokens=self.max_tokens,
            temperature=0.3
        )
        self.total_cost += cost
        return
    
    async def verify_code(self, function_name: str, code:str, description:str, temperature=0.8): 
        content = self.verifier_prompt.format(function_name = function_name,
                                              code_snippet=code, description=description)
        messages = [(
            {
                "role": Sender.USER,
                "content": [BetaTextBlockParam(type="text", text=content)],
            }
        )]
        messages, cost = await sampling_loop(
            system_prompt_suffix="",
            model=self.model_name,
            provider=self.provider,
            messages=messages,
            output_callback=_output_callback,
            tool_output_callback=_tool_output_callback,
            api_response_callback=_api_response_callback,
            api_key = self.api_key,
            max_tokens=self.max_tokens,
            temperature=temperature,
        )

        # bookkeeping
        self.log.append(messages)
        print("Iteration cost: ", cost)
        self.total_cost += cost
        
        # parse result
        last_message = messages[-1]
        if not last_message['role'] == Sender.AGENT:
            raise ValueError("The last message did not come from the agent.")
        response_params = last_message['content']
        if isinstance(response_params, list):
            out = response_params[0]['text']
        else:
            out = response_params['text']
        self.results.append({
            "raw_model_output": out,
            "original_code": code,
        })
        return out
    
def _output_callback(content_block: BetaContentBlockParam | ToolResult) -> None:
    if not VERBOSE:
        return
    print("="*50)
    if content_block['type'] == "tool_use":
        print_colored(f"Tool command: {content_block['name']}", COLOR_BOOK['tool_use_input'])
        for key, value in content_block['input'].items():
            print_colored(textwrap.indent(f"{key}: {value}",'    '), COLOR_BOOK['tool_use_input'])
    elif content_block['type'] == "text":
        print_colored(content_block['text'], COLOR_BOOK['agent_response'])

def _tool_output_callback(result: ToolResult, tool_use_id: str) -> None:
    if not VERBOSE:
        return
    # print("="*50)
    # print("Tool Use Result")
    print_colored("Tool result", COLOR_BOOK['tool_use_output'])
    print_colored(f"\t- output: {result.output}", COLOR_BOOK['tool_use_output'])
    print_colored(f"\t- error: {result.error}", COLOR_BOOK['tool_use_output'])
    print_colored(f"\t- system: {result.system}", COLOR_BOOK['tool_use_output'])
    
def _api_response_callback(
    request: httpx.Request,
    response: httpx.Response | object | None,
    error: Exception | None
) -> None:
    return

# async def main():
#     # load json 
#     dataset = json.load(open("questions.json",'r'))
#     verifier = VerifierAgent()
#     for data in dataset:
#         await verifier.verify_code(data['function'], data['description'])
#     res = verifier.results
#     cost = verifier.total_cost
#     with open("cost.txt",'r') as f:
#         current_cost = float(f.read())
#     with open("cost.txt",'w') as f:
#         f.write(str(current_cost + cost))
#     with open("verification_results.json",'w') as f:
#         json.dump(res,f,indent=4)
#     with open("log.json",'w') as f:
#         json.dump(verifier.log,f,indent=4)
#     print("Total Cost: ", cost)

# if __name__ == "__main__":
#     asyncio.run(main())