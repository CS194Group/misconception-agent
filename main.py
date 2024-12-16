import os
import certifi

os.environ['SSL_CERT_FILE'] = certifi.where()

import asyncio
import wandb
import weave

import pathlib
import time
from typing import Literal, Callable, List

import dspy
from colorama import init
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot
from sklearn.model_selection import train_test_split
from dataclasses import asdict

import uuid


# TODO: check if /models exists and if not create
pathlib.Path("models").mkdir(parents=True, exist_ok=True)
pathlib.Path("data").mkdir(parents=True, exist_ok=True)
pathlib.Path("logs").mkdir(parents=True, exist_ok=True)

from src.config import Config, load_config
from src.util import Persona
from src.agents import Agent, AdvancedAgent
from src.dataloader import DataManager
from src.evaluation import EvaluationManager
from src.eot import ExchangeOfThought
from src.util import LanguageModel, PrefixedChatAdapter


# Initialize colorama
init(autoreset=True)

# CONSTANTS
DEBUG: bool = True
SEED: int = 77
API: Literal['lambda', 'openai'] = 'lambda'
MAX_TOKEN: int = 100
ID = uuid.uuid4().hex[:8]

lm_wrapper = LanguageModel(max_tokens=MAX_TOKEN, service=API)
custom_adapter = PrefixedChatAdapter()
dspy.configure(lm=lm_wrapper.lm, adapter=custom_adapter)


def evaluate_with_weave(evaluation_dataset: List[dspy.Example], model: dspy.Module, scorer: Callable) -> None:
    weave_eval = weave.Evaluation(
        dataset=[dspy.Example(**row).toDict() for row in evaluation_dataset],
        scorers=[scorer],
        evaluation_name=f"run-{ID}",
    )

    @weave.op()
    def my_model(QuestionText, AnswerText, ConstructName, SubjectName, CorrectAnswer):
        print("Evaluating ...")
        model_output = model(QuestionText, AnswerText, ConstructName, SubjectName, CorrectAnswer)
        return model_output

    result = asyncio.run(weave_eval.evaluate(my_model))
    wandb.run.config.update({'weave.run.name': ID})
    wandb.log(result, step=1)
    print("Weave Result: " + str(result))



def main(args: Config):
    weave.init(project_name="llma-agents" if not DEBUG else "llma-agents-debug")
    start = time.time()

    # wandb.init(project="llma-agents" if not DEBUG else "llma-agents-debug")
    examples = DataManager.get_examples(pathlib.Path("data"), debug=DEBUG)

    # Split in 80% validation as this is what is suggested here https://dspy.ai/learn/optimization/overview/
    train_data, val_data = train_test_split(examples, test_size=0.8, random_state=SEED)

    # Collect persona prompts


    # Set up Agents
    agent_a = Agent(name="Agent A" , persona_promt=None)
    agent_b = Agent(name="Agent B" , persona_promt=None)
    agent_c = Agent(name="Agent C" , persona_promt=None)
    agent_d = Agent(name="Agent D" , persona_promt=None)
    agent_e = Agent(name="Agent E" , persona_promt=None)

    # agent_a = AdvancedAgent(name="Agent A" , persona_promt=Persona.AGENT_A_new)
    # agent_b = AdvancedAgent(name="Agent B" , persona_promt=Persona.AGENT_B_new)
    # agent_c = AdvancedAgent(name="Agent C" , persona_promt=Persona.AGENT_C_new)
    # agent_d = AdvancedAgent(name="Agent D" , persona_promt=Persona.AGENT_D_new)
    # agent_e = AdvancedAgent(name="Agent E" , persona_promt=Persona.AGENT_E_new)

    if args.ExchangeOfThought.mode != "single":
        predict = ExchangeOfThought(
            agent_a, agent_b, agent_c, agent_d, agent_e, rounds=args.ExchangeOfThought.rounds, mode=args.ExchangeOfThought.mode)
        persona_prompts = {
            "Agent A Persona": agent_a.prefix_promt,
            "Agent B Persona": agent_b.prefix_promt,
            "Agent C Persona": agent_c.prefix_promt,
            "Agent D Persona": agent_d.prefix_promt,
            "Agent E Persona": agent_e.prefix_promt,
            "debug": DEBUG
        }
    else:
        predict = Agent(name="Agent A" , persona_promt=None)
        persona_prompts = {
            "Agent A Persona": agent_a.prefix_promt,
            "debug": DEBUG
        }

    wandb.config.update(persona_prompts)

    eval_manager = EvaluationManager(retrive_method=args.Dspy.evaluation.type)

    # compile
    print("Start training ...")
    if args.Dspy.telepropmter.type == "BootstrapFewShot":
        teleprompter = BootstrapFewShot(metric=eval_manager.metric_vector_search_weave, max_labeled_demos=3)
        compiled_predictor = teleprompter.compile(predict, trainset=train_data)
        compiled_predictor.save("models" / pathlib.Path(f'compiled_model-{ID}.dspy'))
        print("Finished training BootstrapFewShot...")
    elif args.Dspy.telepropmter.type == "MIPROv2":
        teleprompter = dspy.MIPROv2(metric=eval_manager.metric_vector_search_weave, auto='medium', num_threads=6)
        compiled_predictor = teleprompter.compile(predict, trainset=train_data, requires_permission_to_run=False)
        compiled_predictor.save("models" / pathlib.Path(f'compiled_model-{ID}.dspy'))
        print("Finished training MIPROv2...")
    else:
        predict.load("models" / pathlib.Path(f"compiled_model-{args.Dspy.telepropmter.type}.dspy"))
        print("Finished loading")

    # --- DO NOT CHANGE anything below this line ---
    evaluate_with_weave(val_data, compiled_predictor, eval_manager.metric_vector_search_weave)

    end = time.time()
    usage = lm_wrapper.get_usage()

    wandb.log({
        "usage_cost_cents": usage[2],
        "input_tokens": usage[0],
        "output_tokens": usage[1],
        "time_taken_seconds": end - start,
        "dataset" : {
            "train.size": len(train_data),
            "val.size": len(val_data)
        }
    })
    #wandb.save("models" / pathlib.Path(f'compiled_model-{ID}.dspy'))

    print(f"Usage cost (in cents) about {usage[2]}, Input Tokens: {usage[0]}, Output Tokens {usage[1]}")
    print("Time taken (in seconds)", end - start)
    print("Run ID: ", ID)

    wandb.finish()
    weave.finish()

if __name__ == "__main__":
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(project="llma-agents" if not DEBUG else "llma-agents-debug", name=f"run-{ID}")
    USE_WANDB_CONFIG = True
    if USE_WANDB_CONFIG:
        args = load_config(dict(wandb.config))
    else:
        args_dict = {
            "ExchangeOfThought": {
                "mode": "multi_4",
                "rounds": 1
            },
            "Dspy": {
                "telepropmter": {             # Nested TelepropmterConfig
                    "type": "BootstrapFewShot"  #Literal['BootstrapFewShot', 'MIPROv2'] # Example integer value for TelepropmterConfig.max_labeled_demos
                },
                "evaluation": {
                    "type": "multi"
                }
            }
        }
        args = load_config(args_dict)

    main(args)