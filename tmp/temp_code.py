


exit(1)
### TEST Agents
import pathlib

import dspy
from dspy import Evaluate
from sklearn.model_selection import train_test_split

from src import util
from src.agents import Agent
from src.dataloader import DataManager
from src.evaluation import EvaluationManager
from src.predict_model import ExchangeOfThought

DEBUG = True
SEED=42
lm = util.get_language_model()
dspy.configure(lm=lm)

# print(EvaluationManager.metric("Does not think 'other' is an acceptable category for data", "Believes that 'other' is not a suitable category for data."))

examples = DataManager.get_examples(pathlib.Path("../data"), debug=DEBUG)

# Split in 80% validation as this is what is suggested here https://dspy.ai/learn/optimization/overview/
train_data, val_data = train_test_split(examples, test_size=0.8, random_state=SEED)

# Set up Agents
agent_a = Agent(name="Agent A")
agent_b = Agent(name="Agent B")
agent_c = Agent(name="Agent C")

predict = ExchangeOfThought(
    agent_a, agent_b, agent_c, rounds=1, mode="Report")

# forward(self, QuestionText, AnswerText, ConstructName, SubjectName, CorrectAnswer)
# from timeit import timeit
# output_time = timeit(lambda: predict("What is the capital of France?", "London", "Geography", "World", "Paris"), number=1)
# print("output_time:", output_time)
# compile
teleprompter = dspy.BootstrapFewShot(metric=EvaluationManager.metric, max_labeled_demos=2)
compiled_predictor = teleprompter.compile(predict, trainset=train_data)
compiled_predictor.save(pathlib.Path('./compiled_model.dspy'))

evaluate_program = Evaluate(devset=val_data, metric=EvaluationManager.metric,
                                num_threads=1, display_progress=True, display_table=10)

output = evaluate_program(compiled_predictor)
print("End")

### END Test Agents

# import pathlib
#
# from src.dataloader import DataManager
#
# df = DataManager.get_examples(pathlib.Path("data/"), debug=True)

print("End")
# import os
#
# import dspy
# from dotenv import load_dotenv
# from litellm import api_base
#
# # Load environment variables
# load_dotenv()
#
# lm = dspy.LM(f"openai/{os.getenv('API_MODEL')}", max_tokens=250, api_key=os.getenv("OPENAI_API_KEY"), api_base=os.getenv("OPENAI_API_BASE"))
# print("lm:", lm("This is a test."))