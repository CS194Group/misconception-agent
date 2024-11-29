import os
from typing import Literal

import dspy
from dotenv import load_dotenv

class LanguageModel:
    def __init__(self, max_tokens: int = 100, service: Literal['lambda', 'openai'] = 'lambda'):
        load_dotenv()
        self.lm: dspy.clients.lm = None
        self._get_language_model(max_tokens, service)

    def _get_language_model(self, max_tokens: int, service: Literal['lambda', 'openai']):
        print("SERVICE: ", service)
        if service == 'lambda':
            self.lm = dspy.LM(f"openai/{os.getenv('LAMBDA_API_MODEL')}", max_tokens=max_tokens, api_key=os.getenv("LAMBDA_API_KEY"),
                    api_base=os.getenv("LAMBDA_API_BASE"))
        elif service == 'openai':
            OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
            if not OPENAI_API_KEY:
                raise EnvironmentError(
                    "OPENAI_API_KEY not found in environment variables.")
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
            self.lm = dspy.LM('openai/gpt-4o-mini', max_tokens=max_tokens)

        assert self.lm is not None, "Language Model not initialized"
        return self.lm

    def get_usage(self):

        amount_input_token = sum([x['usage']['prompt_tokens'] for x in self.lm.history])
        amount_output_token = sum([x['usage']['completion_tokens'] for x in self.lm.history])

        cost_4o_mini = amount_input_token * 0.150 / 10**6 + amount_output_token * 0.600 / 10**6
        cost_4o_mini = round(cost_4o_mini, 2)

        return amount_input_token, amount_output_token, cost_4o_mini

