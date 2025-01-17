from .chatbot_engine import *

class LocalChatbotEngine(ChatbotEngine):
    def __init__(self, model_path, model_args) -> None:
        super().__init__(model_path)
        self.model, self.tokenizer = load_model(
            model_path,
            model_args.device,
            model_args.num_gpus,
            model_args.max_gpu_memory,
            # model_args.load_8bit,
            # model_args.cpu_offloading,
            revision=model_args.revision,
            # debug=model_args.debug,
        )
        self.device = model_args.device

    def initialize_conversation(self):
        conv = get_conversation_template(self.model_path)
        return conv

    @torch.inference_mode()
    def query(self, conv, temperature=0.7, repetition_penalty=1.0, max_new_tokens=512, **kwargs):
        prompt = conv.get_prompt()
        
        inputs = self.tokenizer([prompt])
        inputs = {k: torch.tensor(v).to(self.device) for k, v in inputs.items()}
        output_ids = self.model.generate(
            **inputs,
            do_sample=True if temperature > 1e-5 else False,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
        )

        if self.model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
        outputs = self.tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )
        return outputs, prompt