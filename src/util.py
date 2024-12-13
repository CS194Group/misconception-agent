import os
from typing import Literal
#from typing import override

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
            if not os.getenv('LAMBDA_API_MODEL') or not os.getenv('LAMBDA_API_KEY') or not os.getenv('LAMBDA_API_BASE'):
                raise EnvironmentError(
                    "LAMBDA_API_MODEL, LAMBDA_API_KEY, or LAMBDA_API_BASE not found in environment variables.")
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


class Persona:
    AGENT_B = "You are Ben, a high school student with a track record of excellent grades, particularly in mathematics. Your friends admire your diligence and often seek your guidance in their studies. Your role is to scrutinize the problem at hand with your usual attention to detail, drawing from your vast knowledge of math principles. After considering your friends' approaches, carefully construct your answer, ensuring to clarify each step of your process. Your clear and logical explanations are valuable, as they will serve as a benchmark for your friends to compare and refine their own solutions."
    AGENT_A = "You are Kitty, a high school student admired for your attentiveness and detail-oriented nature. Your friends often rely on you to catch details they might have missed in their work. Your task is to carefully analyze the presented math problem, apply your attentive skills, and piece together a detailed solution. Afterward, you'll have the opportunity to review the solutions provided by your friends, offering insights and suggestions. Your careful revisions will help all of you to enhance your understanding and arrive at the most accurate solutions possible."
    AGENT_C = "You are Peter, a high school student recognized for your unique problem-solving abilities. Your peers often turn to you for assistance when they encounter challenging tasks, as they appreciate your knack for devising creative solutions. Today, your challenge is to dissect the given math problem, leveraging your unique problem-solving strategies. Once you've crafted your solution, share it with your friends, Ben and Kitty, so they can see a different perspective. Your innovative approach will not only provide an answer but also inspire Ben and Kitty to think outside the box and possibly revise their own solutions."
    AGENT_A_new = "You are a diligent, reliable, and knowledgeable."
    AGENT_B_new = "You are a attentive and detail-oriented."
    AGENT_C_new = "You have unique problem-solving abilities and can think out of the box."
    AGENT_D_new = "You are a strategic synthesizer who integrates multiple perspectives, systematically connecting complex ideas and uncovering deeper insights while maintaining intellectual flexibility and rigorous analytical precision."
    AGENT_E_new = "You are a transformative catalyst who bridges conceptual gaps, challenges conventional thinking, and generates innovative solutions by reframing problems through multiple lenses of creativity and critical analysis."


# chat_adapter_wrapper.py
from typing import Any, Dict


class PrefixedChatAdapter(dspy.adapters.chat_adapter.ChatAdapter):
    # def __init__(self, callbacks=None):
    #     super().__init__(callbacks=callbacks)
    #     self.prefix = prefix

    #@override
    def format(self, signature: dspy.signatures.signature.Signature, demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[dict[str, Any]]:
        # Get the prepared instructions from the original ChatAdapter.
        # print("Custom PrefixedChatAdapter format method called")
        prepared_instructions = dspy.adapters.chat_adapter.prepare_instructions(signature)

        # Add the desired prefix.
        prefix = inputs.pop("prefix", None)

        if prefix is not None:
            instructions = f"{prefix}\n{prepared_instructions}"
        else:
            instructions = prepared_instructions

        # Add the prefixed instructions to the messages.
        # noinspection PyListCreation
        messages: list[dict[str, Any]] = []
        messages.append({"role": "system", "content": instructions})

        # Proceed with the rest of the logic as per the original `format` method.
        incomplete_demos = [
            demo for demo in demos if not all(k in demo and demo[k] is not None for k in signature.fields)
        ]
        complete_demos = [demo for demo in demos if demo not in incomplete_demos]

        demos = incomplete_demos + complete_demos

        for demo in demos:
            messages.append(self.format_turn(signature, demo, role="user", incomplete=demo in incomplete_demos))
            messages.append(self.format_turn(signature, demo, role="assistant", incomplete=demo in incomplete_demos))

        messages.append(self.format_turn(signature, inputs, role="user"))
        return messages

# Example usage:
# Assuming you have an instance of signature, demos, and inputs
# adapter = PrefixedChatAdapter(prefix="Custom Prefix: ")
# messages = adapter.format(signature, demos, inputs)
