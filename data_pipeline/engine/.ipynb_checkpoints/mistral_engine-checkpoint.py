from .chatbot_engine import *

class MistralChatbotEngine(ChatbotEngine):
    def __init__(self, model_path, model_args) -> None:
        super().__init__(model_path)
        # self.model = AutoModelForCausalLM.from_pretrained(model_path)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model, self.tokenizer = load_model(
            model_path,
            model_args.device,
            model_args.num_gpus,
            model_args.max_gpu_memory,
            #model_args.load_8bit,
            #model_args.cpu_offloading,
            revision=model_args.revision,
            # debug=model_args.debug,
        )
        self.device = model_args.device

    def initialize_conversation(self):
        conv = get_conversation_template('gpt-4')
        return conv

    @torch.inference_mode()
    def query(self, conv, temperature=0.7, repetition_penalty=1.0, max_new_tokens=512, **kwargs):
        messages = [{'role': role, 'content': message} for (role, message) in conv.messages[:-1]]
        # print(messages)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)

        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")

        model_inputs = encodeds.to(self.device)
        generated_ids = self.model.generate(
            model_inputs, 
            do_sample=True if temperature > 1e-5 else False,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
        )
        generated_ids = generated_ids[0][len(encodeds[0]) :]
        generated_ids = generated_ids[:-1]
        outputs = self.tokenizer.decode(generated_ids)
        return outputs, prompt