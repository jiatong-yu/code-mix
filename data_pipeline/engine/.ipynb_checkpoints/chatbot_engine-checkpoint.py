import argparse
from functools import partial
import json
from datasets import Dataset
import random
import os
import torch
from fastchat.model import load_model, get_conversation_template
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig
from fastchat.conversation import get_conv_template #get_conversation_template
from concurrent.futures import ThreadPoolExecutor
import time
from datasets import load_dataset

class ChatbotEngine():
    def __init__(self, model_path) -> None:
        self.model_path = model_path

    def query(self, conv, **kwargs):
        raise NotImplementedError

    def initialize_conversation(self):
        return get_conversation_template(self.model_path)