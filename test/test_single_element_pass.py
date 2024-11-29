
import pathlib
from typing import Literal

import dspy
from sklearn.model_selection import train_test_split

from src import util
from src.agents import Agent
from src.dataloader import DataManager
from src.predict_model import ExchangeOfThought
from src.util import LanguageModel, Persona

element_id = 1889
DEBUG = True
SEED=42
API: Literal['lambda', 'openai'] = 'lambda'
MAX_TOKEN: int = 100

lm_wrapper = LanguageModel(max_tokens=MAX_TOKEN, service=API)
dspy.configure(lm=lm_wrapper.lm)

examples = DataManager.get_examples(pathlib.Path("../data"), debug=False)
single_example = None
for example in examples:
    if example.MisconceptionId == element_id:
        single_example = example
        break

# Set up Agents
# Set up Agents
agent_a = Agent(name="Agent A", persona_promt=Persona.AGENT_A)
agent_b = Agent(name="Agent B", persona_promt=Persona.AGENT_B)
agent_c = Agent(name="Agent C", persona_promt=Persona.AGENT_C)

predict = ExchangeOfThought(
    agent_a, agent_b, agent_c, rounds=1, mode="Report")


# import time
# start = time.time()
# forward(self, QuestionText, AnswerText, ConstructName, SubjectName, CorrectAnswer)
# def example_func():
def test_simple_agent_predict():
    predict(single_example.QuestionText, single_example.AnswerText, single_example.ConstructName, single_example.SubjectName, single_example.CorrectAnswer)
    assert True

# Measure time of predict function
# import timeit
# output_time = timeit.timeit(example_func, number=50)
# print("output_time:", output_time)
# end = time.time()
# print(out)
# print("Time to run single element (seconds):", end-start)

# from timeit import timeit
# output_time = timeit(lambda: , number=1)
# print("output_time:", output_time)