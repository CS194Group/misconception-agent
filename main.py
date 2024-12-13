import asyncio
import os

from weave.trace.context.call_context import get_current_call

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

from src.agents import Agent, AdvancedAgent
from src.dataloader import DataManager
from src.evaluation import EvaluationManager
from src.predict_model import ExchangeOfThought
from src.util import LanguageModel, PrefixedChatAdapter
from src.util import Persona
import uuid

# Initialize colorama
init(autoreset=True)

# CONSTANTS
DEBUG: bool = True
SEED: int = 39
API: Literal['lambda', 'openai'] = 'lambda'
MAX_TOKEN: int = 100
ID = uuid.uuid4().hex[:8]

lm_wrapper = LanguageModel(max_tokens=MAX_TOKEN, service=API)
custom_adapter = PrefixedChatAdapter()
dspy.configure(lm=lm_wrapper.lm, adapter=custom_adapter)

# TODO: check if /models exists and if not create
pathlib.Path("models").mkdir(parents=True, exist_ok=True)
pathlib.Path("data").mkdir(parents=True, exist_ok=True)

def evaluate_with_weave(evaluation_dataset: List[dspy.Example], model: dspy.Module, scorer: Callable) -> None:


    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(project="llma-agents" if not DEBUG else "llma-agents-debug", name=f"run-{ID}")
    weave.init(project_name="llma-agents" if not DEBUG else "llma-agents-debug")

    weave_eval = weave.Evaluation(
        dataset=[dspy.Example(**row).toDict() for row in evaluation_dataset],
        scorers=[scorer],
        evaluation_name=f"run-{ID}",
    )

    @weave.op()
    def my_model(QuestionText, AnswerText, ConstructName, SubjectName, CorrectAnswer):
        model_output = model(QuestionText, AnswerText, ConstructName, SubjectName, CorrectAnswer)
        # current_call = get_current_call()
        # if current_call:
        #     current_call.attributes.update({
        #         "wandb.run.id": wandb.run.id,
        #         "wandb.run.name": wandb.run.name,
        #         "wandb.run.url": wandb.run.url,
        #         "wandb.run.project": wandb.run.project,
        #         "wandb.run.entity": wandb.run.entity,
        #         "wandb.run.config": wandb.run.config
        #     })
        return model_output

    result = asyncio.run(weave_eval.evaluate(my_model))
    wandb.run.config.update({'weave.run.name': ID})
    wandb.log(result, step=1)
    wandb.finish()
    weave.finish()
    print("Weave Result: " + str(result))


if __name__ == "__main__":
    # Load training and test sets
    start = time.time()

    # wandb.init(project="llma-agents" if not DEBUG else "llma-agents-debug")
    examples = DataManager.get_examples(pathlib.Path("data"), debug=DEBUG)

    # Split in 80% validation as this is what is suggested here https://dspy.ai/learn/optimization/overview/
    train_data, val_data = train_test_split(examples, test_size=0.8, random_state=SEED)

    # Set up Agents
    # agent_a = Agent(name="Agent A" , persona_promt=Persona.AGENT_A_new)
    # agent_b = Agent(name="Agent B" , persona_promt=Persona.AGENT_B_new)
    # agent_c = Agent(name="Agent C" , persona_promt=Persona.AGENT_C_new)
    # agent_d = Agent(name="Agent d" , persona_promt=Persona.AGENT_D_new)
    # agent_e = Agent(name="Agent e" , persona_promt=Persona.AGENT_E_new)

    agent_a = AdvancedAgent(name="Agent A" , persona_promt=Persona.AGENT_A_new)
    agent_b = AdvancedAgent(name="Agent B" , persona_promt=Persona.AGENT_B_new)
    agent_c = AdvancedAgent(name="Agent C" , persona_promt=Persona.AGENT_C_new)
    agent_d = AdvancedAgent(name="Agent D" , persona_promt=Persona.AGENT_D_new)
    agent_e = AdvancedAgent(name="Agent E" , persona_promt=Persona.AGENT_E_new)

    # print(agent_a(train_data[3]['QuestionText'], train_data[3]['AnswerText'], train_data[3]['ConstructName'], train_data[3]['SubjectName'], train_data[3]['CorrectAnswer']))

    predict = ExchangeOfThought(
        agent_a, agent_b, agent_c, agent_d, agent_e, rounds=1, mode="Report")
    # predict = Agent(name="Single Agent" )
    eval_manager = EvaluationManager()

    # compile
    # teleprompter = BootstrapFewShot(metric=eval_manager.metric_vector_search, max_labeled_demos=3)
    # teleprompter = dspy.MIPROv2(metric=evaluation_metric, auto='medium', num_threads=6)
    # compiled_predictor = teleprompter.compile(predict, trainset=train_data, requires_permission_to_run=False)
    # compiled_predictor = teleprompter.compile(predict, trainset=train_data)
    # compiled_predictor.save("models" / pathlib.Path('compiled_model.dspy'))

    # train_model()
    predict.load("models" / pathlib.Path('compiled_model.dspy'))

    # --- DO NOT CHANGE anything below this line ---
    evaluate_with_weave(val_data, predict, eval_manager.metric_vector_search_weave)

    end = time.time()
    usage = lm_wrapper.get_usage()

    print(f"Usage cost (in cents) about {usage[2]}, Input Tokens: {usage[0]}, Output Tokens {usage[1]}")
    print("Time taken (in seconds)", end - start)