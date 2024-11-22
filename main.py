'''
TODO:
1. Making the code run properly(debuging the FAISS code??)
2. Figuring out how to handle the inconsistency between training data and prediction results.
3. Trying more advanced structure.
'''

import os
import dspy

from dotenv import load_dotenv
from dspy.evaluate import Evaluate
from colorama import Fore, Style, init
from dspy.teleprompt import BootstrapFewShot

from Agents import Agent
from Evaluation import evaluate_answers
from src.predict_model import ExchangeOfThought
from Dataloader import load_data, load_misconceptions

# Initialize colorama
init(autoreset=True)

# Load environment variables
load_dotenv()

# Configure OpenAI API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY not found in environment variables.")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
lm = dspy.LM('openai/gpt-3.5-turbo')
dspy.configure(lm=lm)


if __name__ == "__main__":
    # Misconception mapping
    misconceptions = load_misconceptions('data/misconception_mapping.csv')

    # Load training and test sets
    train_data = load_data('data/train.csv')
    test_data = load_data('data/train.csv', is_test=True)

    # Set up Agents
    agent_a = Agent(name="Agent A")
    agent_b = Agent(name="Agent B")
    agent_c = Agent(name="Agent C")

    # evaluate
    evaluate_program = Evaluate(devset=test_data, metric=evaluate_answers, num_threads=8, display_progress=True, display_table=10)
    predict = ExchangeOfThought(agent_a, agent_b, agent_c, rounds=3, mode="Report")
    eval1 = evaluate_program(predict)

    # compile
    teleprompter = BootstrapFewShot(metric=evaluate_answers, max_labeled_demos=1)
    compiled_predictor = teleprompter.compile(predict, trainset=train_data)

    # evaluate again
    eval_compiled = evaluate_program(compiled_predictor)
    compiled_predictor.save('./compiled_model.pkl')
    print(eval_compiled)
