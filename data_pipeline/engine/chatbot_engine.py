import time
import os
import yaml
import logging
from typing import Any, Dict, List, Optional
from openai import AzureOpenAI
import anthropic

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Adjust as needed

def prompt_to_chatml(
    prompt: str,
    start_token: str = "<|im_start|>",
    end_token: str = "<|im_end|>"
) -> List[Dict[str, str]]:
    """
    Convert a custom ChatML-like prompt to a list of messages with roles/content.
    """
    prompt = prompt.strip()
    if not prompt.startswith(start_token) or not prompt.endswith(end_token):
        raise ValueError(
            f"Prompt must start with '{start_token}' and end with '{end_token}'."
        )

    def string_to_dict(to_convert: str) -> Dict[str, str]:
        """
        Converts a string with key=value segments into a dict.
        Example:
            " name=user university=stanford" -> {"name": "user", "university": "stanford"}
        """
        result = {}
        for s in to_convert.split(" "):
            s = s.strip()
            if s and "=" in s:
                key, val = s.split("=", 1)
                result[key] = val
        return result

    messages: List[Dict[str, str]] = []
    segments = prompt.split(start_token)
    # Skip the first empty split if the prompt starts with start_token
    for p in segments[1:]:
        parts = p.split("\n", 1)
        if len(parts) < 2:
            logger.warning("Segment does not have both role and content. Skipping.")
            continue

        role_line = parts[0].strip()
        content_block = parts[1].split(end_token, 1)[0].strip()

        if role_line.startswith("system") and role_line != "system":
            # e.g. "system name=foo"
            extra = role_line.split("system", 1)[-1]
            other_params = string_to_dict(extra)
            role = "system"
        else:
            role = role_line
            other_params = {}

        messages.append(dict(role=role, content=content_block, **other_params))

    return messages


class ChatbotEngine:
    """
    A chatbot engine that interfaces with OpenAI (Azure) and Anthropic APIs
    for text generation, question creation, and refinement tasks.
    """

    def __init__(self, config_path: str) -> None:
        """
        Args:
            config_path (str): Path to a YAML config file with client credentials, system prompts, etc.
        """
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError) as e:
            logger.exception("Failed to load config from %s.", config_path)
            raise e

        # Base cost definitions for each model: [prompt_cost, completion_cost] in USD/token
        self.cost_book: Dict[str, List[float]] = {
            "gpt-4o": [5 / 1e6, 15 / 1e6],
            "gpt-4-turbo": [10 / 1e6, 30 / 1e6],
            "gpt-4-1106-preview": [10 / 1e6, 30 / 1e6],
            "gpt-4-0613": [1 / 1e6, 2 / 1e6],
            "gpt-35-turbo": [0.5 / 1e6, 1.5 / 1e6],
            "claude-3-5-sonnet-20240620": [3 / 1e6, 15 / 1e6],
            "o1-preview": [15 / 1e6, 60 / 1e6],
            "o1-mini": [3 / 1e6, 12 / 1e6],
        }

        self.chat_history: List[Dict[str, str]] = []
        self.total_cost: float = 0.0
        self.prompt_process_func = prompt_to_chatml
        self.most_recent_model: Optional[str] = None

        # Clients (OpenAI, Anthropic) are initialized below
        self.openai_client = None
        self.anthropic_client = None

        self._init_clients()

    def _init_clients(self) -> None:
        """
        Initialize OpenAI (Azure) and Anthropic clients.
        Tries to read environment variables first, then falls back to config keys.
        """
        # System prompts
        self.openai_system_prompt = self._load_system_prompt(
            self.config["openai"].get("system_prompt_path"),
            self.config["openai"].get("system_prompt")
        )

        self.anthropic_system_prompt = self._load_system_prompt(
            self.config["anthropic"].get("system_prompt_path"),
            self.config["anthropic"].get("system_prompt")
        )

        # 1) Attempt to retrieve keys from environment variables
        openai_api_key = os.getenv("OPENAI_API_KEY", self.config["openai"].get("api_key"))
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", self.config["anthropic"].get("api_key"))

        # 2) Initialize OpenAI client
        try:
            self.openai_client = AzureOpenAI(
                api_version=self.config["openai"]["api_version"],
                azure_endpoint=self.config["openai"]["azure_endpoint"],
                api_key=openai_api_key,
            )
            logger.info("Initialized Azure OpenAI client.")
        except Exception as e:
            logger.warning("OpenAI client initialization failed: %s", e)
            self.openai_client = None

        # 3) Initialize Anthropic client
        try:
            self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
            logger.info("Initialized Anthropic client.")
        except Exception as e:
            logger.warning("Anthropic client initialization failed: %s", e)
            self.anthropic_client = None

        if not self.openai_client and not self.anthropic_client:
            raise ValueError("No client is initialized. Check your API keys or config.")

    def _load_system_prompt(self, path: Optional[str], fallback: Optional[str]) -> Optional[str]:
        """
        Helper method to load system prompt from file if it exists; otherwise use fallback.
        """
        if path and os.path.exists(path):
            with open(path, "r") as f:
                return f.read()
        return fallback

    def get_cost(self) -> float:
        """
        Returns the accumulated cost from prompts + completions (USD).
        """
        return self.total_cost

    def clear_chat_history(self) -> None:
        """
        Clears chat conversation history and resets the most recent model used.
        """
        self.chat_history.clear()
        self.most_recent_model = None

    def inference(
        self,
        prompt: str,
        kwargs: Optional[Dict[str, Any]],
        model: str,
        temperature: float = 0.5
    ) -> str:
        """
        Single-turn inference without using or saving to conversation history.

        Args:
            prompt (str): Prompt template or text to query.
            kwargs (dict): Variables to format into the prompt.
            model (str): Model name (OpenAI or Anthropic).
            temperature (float): LLM sampling temperature.

        Returns:
            str: LLM output text.
        """
        formatted_prompt = self._format_prompt(prompt, kwargs)
        messages = self.prompt_process_func(formatted_prompt)
        self._validate_messages(messages)
        if "claude" in model:
            return self._query_anthropic(messages, model, temperature)
        else:
            return self._query_openai(messages, model, temperature)

    def chat(
        self,
        prompt: str,
        kwargs: Optional[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.5
    ) -> str:
        """
        Multi-turn chat that updates (appends to) the conversation history.

        Args:
            prompt (str): Prompt template or text to query.
            kwargs (dict): Variables to format into the prompt.
            model (str): Model name (OpenAI or Anthropic). If None, reuses last used.
            temperature (float): LLM sampling temperature.

        Returns:
            str: LLM output text.
        """
        model = model or self._ensure_model()

        formatted_prompt = self._format_prompt(prompt, kwargs)
        new_messages = self.prompt_process_func(formatted_prompt)
        self._validate_messages(new_messages, consider_history=True)

        if self.most_recent_model and ("claude" in self.most_recent_model) != ("claude" in model):
            logger.warning(
                "Switching from model '%s' to '%s' mid-conversation.", 
                self.most_recent_model, model
            )

        self.chat_history.extend(new_messages)
        if "claude" in model:
            out = self._query_anthropic(self.chat_history, model, temperature)
        else:
            out = self._query_openai(self.chat_history, model, temperature)
        self.chat_history.append({"role": "assistant", "content": out})
        return out

    def memoryless_chat(
        self,
        prompt: str,
        kwargs: Optional[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.5
    ) -> str:
        """
        Multi-turn read-only chat: uses existing history but does NOT update it.

        Args:
            prompt (str): Prompt template or text to query.
            kwargs (dict): Variables to format into the prompt.
            model (str): Model name (OpenAI or Anthropic). If None, reuses last used.
            temperature (float): LLM sampling temperature.

        Returns:
            str: LLM output text.
        """
        model = model or self._ensure_model()

        formatted_prompt = self._format_prompt(prompt, kwargs)
        new_messages = self.prompt_process_func(formatted_prompt)
        self._validate_messages(new_messages, consider_history=True)

        # We do not modify self.chat_history
        inference_messages = self.chat_history + new_messages

        if self.most_recent_model and ("claude" in self.most_recent_model) != ("claude" in model):
            logger.warning(
                "Switching from model '%s' to '%s' mid-conversation (memoryless).",
                self.most_recent_model, model
            )

        if "claude" in model:
            return self._query_anthropic(inference_messages, model, temperature)
        else:
            return self._query_openai(inference_messages, model, temperature)

    def _ensure_model(self) -> str:
        """
        Helper to ensure we have a valid model for chat or memoryless_chat.
        """
        if self.most_recent_model:
            return self.most_recent_model
        raise ValueError("No model specified, and no recent model available.")

    def _format_prompt(
        self,
        prompt: str,
        kwargs: Optional[Dict[str, Any]]
    ) -> str:
        """
        Format the prompt using kwargs (if provided). Raises ValueError if missing keys.

        Args:
            prompt (str): The raw prompt template.
            kwargs (dict): Key-value pairs to fill in the prompt.

        Returns:
            str: The fully formatted prompt text.
        """
        if kwargs:
            try:
                return prompt.format(**kwargs)
            except KeyError as e:
                logger.error("Prompt formatting error: missing key '%s'.", e)
                raise ValueError(
                    f"Given kwargs {kwargs} do not match placeholders in prompt.\nPrompt: {prompt}"
                ) from e
        return prompt

    def _query_openai(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float,
        max_new_tokens: int = 4096
    ) -> str:
        """
        Query the OpenAI/Azure model with chat completions.

        Args:
            messages (list): A list of dicts with 'role' and 'content'.
            model (str): The model name for Azure OpenAI.
            temperature (float): LLM sampling temperature.
            max_new_tokens (int): Max tokens in completion.

        Returns:
            str: The text output from the model.
        """
        if not self.openai_client:
            raise ValueError("OpenAI client is not initialized or missing API key.")

        self.most_recent_model = model

        # Insert system prompt if any
        if self.openai_system_prompt:
            messages.insert(0, {"role": "system", "content": self.openai_system_prompt})

        last_msg = messages[-1]["content"] if messages else ""
        json_format = ("ONLY the json object" in last_msg) and ("o1" not in model)

        while True:
            try:
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    response_format={"type": "json_object"} if json_format else None
                )
                break
            except Exception as e:
                logger.error("OpenAI call failed: %s. Retrying in 3 seconds.", e)
                time.sleep(3)

        outputs = [choice.message.content for choice in response.choices]
        usage = response.usage
        cost_prompt = usage.prompt_tokens * self.cost_book.get(model, [0, 0])[0]
        cost_completion = usage.completion_tokens * self.cost_book.get(model, [0, 0])[1]
        self.total_cost += (cost_prompt + cost_completion)

        return outputs[0] if outputs else ""

    def _query_anthropic(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float,
        max_new_tokens: int = 4096
    ) -> str:
        """
        Query the Anthropic model with the provided conversation messages.

        Args:
            messages (list): A list of role-content dicts.
            model (str): Anthropic model name (e.g. "claude-3-5-sonnet-20240620").
            temperature (float): LLM sampling temperature.
            max_new_tokens (int): Max tokens in the model's response.

        Returns:
            str: The text output from the Anthropic model.
        """
        if not self.anthropic_client:
            raise ValueError("Anthropic client is not initialized or missing API key.")

        self.most_recent_model = model

        claude_messages = []
        for m in messages:
            claude_messages.append({
                "role": m["role"],
                "content": [{"type": "text", "text": m["content"]}],
            })

        while True:
            try:
                response = self.anthropic_client.messages.create(
                    model=model,
                    system=self.anthropic_system_prompt,
                    messages=claude_messages,
                    temperature=temperature,
                    max_tokens=max_new_tokens
                )
                break
            except Exception as e:
                logger.error("Anthropic call failed: %s. Retrying in 3 seconds.", e)
                time.sleep(3)

        usage = response.usage
        input_cost = usage.input_tokens * self.cost_book.get(model, [0, 0])[0]
        output_cost = usage.output_tokens * self.cost_book.get(model, [0, 0])[1]
        self.total_cost += (input_cost + output_cost)

        if response.content:
            return response.content[0].text
        return ""

    def _validate_messages(
        self,
        messages: List[Dict[str, Any]],
        consider_history: bool = False
    ) -> None:
        """
        Ensures conversation flow has alternating roles and ends with a user role.

        Args:
            messages (List[Dict[str, Any]]): The new messages to be added.
            consider_history (bool): If True, prepend existing self.chat_history for validation.

        Raises:
            ValueError: If roles do not alternate or the last message is not 'user'.
        """
        all_msgs = self.chat_history + messages if consider_history else messages

        prev_role = None
        for msg in all_msgs:
            current_role = msg["role"]
            if prev_role == current_role:
                raise ValueError("Consecutive messages from the same role encountered.")
            prev_role = current_role

        if prev_role != "user":
            raise ValueError("The final message must be from 'user' for a valid conversation.")