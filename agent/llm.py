import os
import openai


class OpenAIModel:
    def __init__(self, config):
        self.config = config
        self.model = config.get("model")
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url")
        self.generate_config = config.get("generate_config")
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(self, messages):
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, **self.generate_config
        )
        return response.choices[0].message.content
