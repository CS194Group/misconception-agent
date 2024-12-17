import asyncio
import wandb
import weave
import os
import certifi

os.environ['SSL_CERT_FILE'] = certifi.where()

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
from src.agents import Agent
from src.dataloader import DataManager
from src.evaluation import EvaluationManager
from src.eot import ExchangeOfThought
from src.util import LanguageModel, PrefixedChatAdapter


# Initialize colorama
init(autoreset=True)

# CONSTANTS
DEBUG: bool = True
SEED: int = 39
API: Literal['lambda', 'openai'] = 'openai'
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
    if args.ExchangeOfThought.UsePersonaPromts:
        persona_prompts = {
            "A": Persona.AGENT_A_original,
            "B": Persona.AGENT_B_original,
            "C": Persona.AGENT_C_original,
            "D": Persona.AGENT_D_new,
            "E": Persona.AGENT_E_new
        }
    else:
        persona_prompts = {
            "A": None,
            "B": None,
            "C": None,
            "D": None,
            "E": None
        }

    # agent_a = AdvancedAgent(name="Agent A" , persona_promt=Persona.AGENT_A_new)
    # agent_b = AdvancedAgent(name="Agent B" , persona_promt=Persona.AGENT_B_new)
    # agent_c = AdvancedAgent(name="Agent C" , persona_promt=Persona.AGENT_C_new)
    # agent_d = AdvancedAgent(name="Agent D" , persona_promt=Persona.AGENT_D_new)
    # agent_e = AdvancedAgent(name="Agent E" , persona_promt=Persona.AGENT_E_new)

    if args.ExchangeOfThought.mode != "single":
        agent_a = Agent(name="Agent A" , persona_promt=persona_prompts["A"])
        agent_b = Agent(name="Agent B" , persona_promt=persona_prompts["B"])
        agent_c = Agent(name="Agent C" , persona_promt=persona_prompts["C"])
        agent_d = Agent(name="Agent D" , persona_promt=persona_prompts["D"])
        agent_e = Agent(name="Agent E" , persona_promt=persona_prompts["E"])
        predict = ExchangeOfThought(
            agent_a, agent_b, agent_c, agent_d, agent_e, rounds=args.ExchangeOfThought.rounds, mode=args.ExchangeOfThought.mode)
    else:
        predict = Agent(name="Agent A" , persona_promt=persona_prompts["A"])
        if args.ExchangeOfThought.rounds != 1:
            wandb.finish()
            weave.finish()
            exit("Invalid rounds for single agent")

    merged_config = {
        **persona_prompts,
        "debug": DEBUG,
        "seed": SEED,
        "api": API,
        "dataset": {"train.size": len(train_data), "val.size": len(val_data)},
        "version": "v2",
    }
    wandb.config.update(merged_config)

    eval_manager = EvaluationManager()

    # compile
    print("Start training ...")
    if args.Dspy.telepropmter.type == "BootstrapFewShot":
        teleprompter = BootstrapFewShot(metric=eval_manager.metric_vector_search, max_labeled_demos=3)
        compiled_predictor = teleprompter.compile(predict, trainset=train_data)
        compiled_predictor.save("models" / pathlib.Path(f'compiled_model-{ID}.dspy'))
        print("Finished training BootstrapFewShot...")
    elif args.Dspy.telepropmter.type == "MIPROv2":
        teleprompter = dspy.MIPROv2(metric=eval_manager.metric_vector_search, auto='medium', num_threads=6)
        compiled_predictor = teleprompter.compile(predict, trainset=train_data, requires_permission_to_run=False)
        compiled_predictor.save("models" / pathlib.Path(f'compiled_model-{ID}.dspy'))
        print("Finished training MIPROv2...")
    elif args.Dspy.telepropmter.type == "untrained":
        compiled_predictor = predict
        pass
    else:
        predict.load("models" / pathlib.Path(f"compiled_model-{args.Dspy.telepropmter.type}.dspy"))
        compiled_predictor = predict
        print("Finished loading")

    # --- DO NOT CHANGE anything below this line ---
    evaluate_with_weave(val_data, compiled_predictor, eval_manager.metric_vector_search_weave)

    end = time.time()
    usage = lm_wrapper.get_usage()

    wandb.log({
        "usage_cost_cents": usage[2],
        "input_tokens": usage[0],
        "output_tokens": usage[1],
        "time_taken_seconds": end - start
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
        print("WARNING: WanDB config not used!")
        args_dict = {
            "ExchangeOfThought": {
                "mode": "Debate",
                "rounds": 1,
                "UsePersonaPromts": True
            },
            "Dspy": {
                "telepropmter": {             # Nested TelepropmterConfig
                    "type": "BootstrapFewShot"  #Literal['BootstrapFewShot', 'MIPROv2'] # Example integer value for TelepropmterConfig.max_labeled_demos
                }
            }
        }
        args = load_config(args_dict)

    main(args)