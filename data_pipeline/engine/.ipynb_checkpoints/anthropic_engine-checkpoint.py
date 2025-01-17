from .chatbot_engine import *
import anthropic
    
class AnthropicChatbotEngine(ChatbotEngine):
    def __init__(self, model_path) -> None:
        super().__init__(model_path)
        
    def initialize_conversation(self):
        # conv = get_conv_template("claude")
        conv = get_conv_template(self.model_path)
        return conv
        
    def query(self, conv, temperature=0, repetition_penalty=1, max_new_tokens=2048, nruns=1, **kwargs):
        while True:
            try:
                client = anthropic.Anthropic()

                messages=conv.messages
                messages = [{'role': role, 'content': message} for (role, message) in conv.messages[:-1]]
                print(messages[:2])

                response = client.messages.create(
                            model=self.model_path,
                    #         system = system_prompt,
                            messages=messages,
                            max_tokens=max_new_tokens,
                            temperature=temperature,
                            **kwargs
                        )
                break
            except Exception as e:
                print(e)
                # sleep for 10 seconds
        outputs = [response]
        return outputs, prompt