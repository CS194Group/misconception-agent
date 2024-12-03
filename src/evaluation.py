import pathlib

import dspy

from src.agents import SummeryAgent
from src.dataloader import DataManager
from src.db import MisconceptionDB
from src.util import LanguageModel
from typing import Literal
import os

class SimpleScorerAgentSignature(dspy.Signature):
    """Predict the similarity score between the groundtruth and prediction. Focus on how similar the content is only. Give a high score if the content is the same. Give a low score if the content is different. Provide a score between 0 and 1.
    """
    groundtruth: str = dspy.InputField()
    prediction: str = dspy.InputField()
    score = dspy.OutputField()

# Evaluating the answer
class EvaluationManager:
    def __init__(self):
        self.misconception_db = MisconceptionDB(pathlib.Path("data/misconception_mapping.csv"))
        self.misconceptions_df = DataManager.get_misconceptions(pathlib.Path("data/misconception_mapping.csv"))
        self.summery_agent = SummeryAgent(name='Summery Agent')

    @staticmethod
    def metric(gold, pred, trace=None):
        # Ensure prediction_score is a float
        try:
            print(f"pred: {pred}")
            prediction = dspy.ChainOfThought(SimpleScorerAgentSignature)(groundtruth=gold.MisconceptionText,
                                                                         prediction=pred)
            score = float(prediction.score.strip())
        except:
            score = 0.01
        return max(0.0, min(1.0, score))

    # def l2_distance(self, gold, pred, trace=None):
    #     return evaluate_answers(gold, pred, trace)

    # This function is probably not working anymore because it's
    def evaluate_answers(self, gold, pred, trace=None):

        try:
            gold_answer_id = gold.answer
            mis_out = self.summery_agent(pred.question, pred.answer)
            print(f"mis_out: {mis_out}")

            final_index = []
            for mis in [mis_out.misconceptionA, mis_out.misconceptionB, mis_out.misconceptionC, mis_out.misconceptionD]:
                final_index.append(self.misconception_db.hybrid_search(mis)[
                                   0].misconception_id)

            for i, mis in enumerate(final_index):
                if mis == gold_answer_id[i]:
                    return True
            return False

        except:
            return False
    
    def map_at_25(self, misconception: str, ground_truth: int):
        """
        misconception: str - the misconception text returned by th EoT model
        ground_truth: int - the ground truth misconception id

        returns: float - the MAP@25 score
        """
        searched_misconceptions = self.misconception_db.hybrid_search(misconception, top_k=25, pre_filter_k=50)
        misconception_ids = [misconception.misconception_id for misconception in searched_misconceptions]
        cutoff = min(len(misconception_ids), 25)
        for k, pred_id in enumerate(misconception_ids[:cutoff]):
            if int(pred_id)==ground_truth:
                return 1.0/(k+1) #we only consider one unique label as relavent for each observation
        return 0.0

if __name__ == "__main__":
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise EnvironmentError("OPENAI_API_KEY not found in environment variables.")
    
    llm = dspy.LM('openai/gpt-4o-mini', max_tokens=130, api_key=OPENAI_API_KEY)
    dspy.configure(lm=llm)
    evaluation_manager = EvaluationManager()
    print(evaluation_manager.misconception_db.hybrid_search("doesn't know triangle's shape", top_k=5, pre_filter_k=5))
    print(evaluation_manager.map_at_25("doesn't know triangle's shape", 1279))