from dataclasses import dataclass
from typing import Literal
from dacite import from_dict

@dataclass
class ExchangeOfThoughtConfig:
    mode: Literal["Report", "bigram", "multi_4", "single", "Debate"]  # Replace 'OtherMode' with actual allowed literals
    rounds: int
    UsePersonaPromts: bool

@dataclass
class TelepropmterConfig:
    type: str #Literal['BootstrapFewShot', 'MIPROv2']

@dataclass
class DspyConfig:
    telepropmter: TelepropmterConfig # Replace 'str' with actual type

@dataclass
class Config:
    ExchangeOfThought: ExchangeOfThoughtConfig
    Dspy: DspyConfig

def load_config(args: dict) -> Config:
    return from_dict(data_class=Config, data=args)

# Example
# args = {
#     "ExchangeOfThought": {
#         "mode": "Debate",  # Options: "Debate", "Report", "Memory", "Relay"
#         "rounds": 5        # Example integer value for rounds
#     },
#     "Dspy": {
#         "max_tokens": 1500,           # Example integer value for max_tokens
#         "service": "openai",          # Options: "lambda", "openai"
#         "max_labeled_demos": 20,      # Example integer value for max_labeled_demos
#         "telepropmter": {             # Nested TelepropmterConfig
#             "max_labeled_demos": 10   # Example integer value for TelepropmterConfig.max_labeled_demos
#         }
#     }
# }

# --- OLD Config
# from dataclasses import dataclass
# from typing import Literal
# from dacite import from_dict
#
# @dataclass
# class ExchangeOfThoughtConfig:
#     mode: Literal["Debate", "Report", "Memory", "Relay"]  # Replace 'OtherMode' with actual allowed literals
#     rounds: int
#
# @dataclass
# class TelepropmterConfig:
#     type: str
#
# @dataclass
# class DspyConfig:
#     service: Literal['lambda', 'openai']
#     telepropmter: TelepropmterConfig # Replace 'str' with actual type
#
# @dataclass
# class Config:
#     ExchangeOfThought: ExchangeOfThoughtConfig
#     Dspy: DspyConfig
#
# def load_config(args: dict) -> Config:
#     return from_dict(data_class=Config, data=args)
#
# # Example
# # args = {
# #     "ExchangeOfThought": {
# #         "mode": "Debate",  # Options: "Debate", "Report", "Memory", "Relay"
# #         "rounds": 5        # Example integer value for rounds
# #     },
# #     "Dspy": {
# #         "max_tokens": 1500,           # Example integer value for max_tokens
# #         "service": "openai",          # Options: "lambda", "openai"
# #         "max_labeled_demos": 20,      # Example integer value for max_labeled_demos
# #         "telepropmter": {             # Nested TelepropmterConfig
# #             "max_labeled_demos": 10   # Example integer value for TelepropmterConfig.max_labeled_demos
# #         }
# #     }
# # }
