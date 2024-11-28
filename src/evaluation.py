from src.agents import SummeryAgent
from src.db import MisconceptionDB

retrieve_model = MisconceptionDB("./data/misconception_mapping.csv")
summery_agent = SummeryAgent(name='Summery Agent')

# Evaluating the answer


def evaluate_answers(gold, pred, trace=None):

    try:
        gold_answer_id = gold.answer
        mis_out = summery_agent(pred.question, pred.answer)

        final_index = []
        for mis in [mis_out.misconceptionA, mis_out.misconceptionB, mis_out.misconceptionC, mis_out.misconceptionD]:
            final_index.append(retrieve_model.hybrid_search(mis)[
                               0].misconception_id)

        for i, mis in enumerate(final_index):
            if mis == gold_answer_id[i]:
                return True
        return False

    except:
        return False
