from .chatbot_engine import *
from .openai_engine import *
from .mistral_engine import *
from .local_chatbot_engine import *
from .anthropic_engine import *

LOCAL_MODEL_PATH_DICT = {'llama-3-8B-instruct': 'meta-llama/Meta-Llama-3-8B-Instruct', \
                         'llama-3-70B-instruct': 'meta-llama/Meta-Llama-3-70B-Instruct', \
                         'llama-3-405B-instruct': 'meta-llama/Meta-Llama-3-405B-Instruct', \
                         'mistral-7b-instruct-v0.3': 'mistralai/Mistral-7B-Instruct-v0.3', \
                         'code-llama-7b-instruct': 'codellama/CodeLlama-7b-Instruct-hf', \
                         'code-llama-13b-instruct': 'codellama/CodeLlama-13b-Instruct-hf', \
                         'code-llama-34b-instruct': 'codellama/CodeLlama-34b-Instruct-hf', \
                         'code-llama-70b-instruct': 'codellama/CodeLlama-70b-Instruct-hf', \
                         # 'starcoder': 'bigcode/starcoder', \
                         # 'WizardCoder-Python-13B-V1.0': 'WizardLMTeam/WizardCoder-Python-13B-V1.0', \
                         # 'WizardCoder-Python-34B-V1.0': 'WizardLMTeam/WizardCoder-Python-34B-V1.0', \
                         # 'DeepSeek-Coder-V2-Lite-Instruct': 'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct', \
                         # 'DeepSeek-Coder-V2-Instruct': 'deepseek-ai/DeepSeek-Coder-V2-Instruct', \
                         
                  }
API_MODEL_PATH_DICT = {'gpt-4-turbo-2024-04-09': 'gpt-4-turbo-2024-04-09', \
                       'claude-3-5-sonnet-20240620': 'claude-3-5-sonnet-20240620', \
                      }

MODEL_PATH_DICT = {**LOCAL_MODEL_PATH_DICT, **API_MODEL_PATH_DICT}

def get_engine(model_path, args=None):
    model_path=MODEL_PATH_DICT[model_path]

    print(f"{args=}")
        
    if 'gpt' in model_path:
        engine = OpenAIChatbotEngine(model_path)
    elif 'mistral' in model_path.lower():
        engine = MistralChatbotEngine(model_path, args)
    elif 'claude' in model_path.lower():
        engine = AnthropicChatbotEngine(model_path)
    elif 'llama' in model_path.lower():
        engine = LocalChatbotEngine(model_path, args)
    return engine