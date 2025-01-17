from .chatbot_engine import *
from openai import AzureOpenAI


class OpenAIChatbotEngine(ChatbotEngine):
    def __init__(self, model_path) -> None:
        super().__init__(model_path)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.sleep_time = 10
        self.failed_attempts = 0
    
    def query(self, conv, temperature=0.7, repetition_penalty=1, max_new_tokens=4096, nruns=1, **kwargs):
        while True:
            try:
                prompt = conv.to_openai_api_messages()

                client = AzureOpenAI()

                response = client.chat.completions.create(
                            model=self.model_path,
                            messages = prompt,
                            n=nruns,
                            max_tokens=max_new_tokens,
                            **kwargs
                        )
                self.sleep_time = 10
                self.failed_attempts = 0
                break
            except Exception as e:
                print(e)
                # sleep for 10 seconds
                time.sleep(self.sleep_time)
                self.sleep_time *= 2
                self.failed_attempts += 1
                if self.failed_attempts > 10:
                    raise Exception("Too many failed attempts")
        outputs = [choice.message.content for choice in response.choices]
        if len(outputs) == 1:
            outputs = outputs[0]
        self.total_prompt_tokens += response.usage.prompt_tokens
        self.total_completion_tokens += response.usage.completion_tokens
        return outputs, prompt