import pathlib

import dspy

from src.agents import SummeryAgent
from src.dataloader import DataManager
from src.db import MisconceptionDB

class SimpleScorerAgentSignature(dspy.Signature):
    """Predict the similarity score between the groundtruth and prediction. Focus on how similar the content is only. Give a high score if the content is the same. Give a low score if the content is different. Provide a score between 0 and 1.
    """
    groundtruth: str = dspy.InputField()
    prediction: str = dspy.InputField()
    score = dspy.OutputField()

# Evaluating the answer
class EvaluationManager:
    def __init__(self):
        self.misconception_db = MisconceptionDB(pathlib.Path("/data/misconception_mapping.csv"))
        self.misconceptions_df = DataManager.get_misconceptions(pathlib.Path("/data/misconception_mapping.csv"))
        self.summery_agent = SummeryAgent(name='Summery Agent')

    @staticmethod
    def metric(gold, pred, trace=None):
        # Ensure prediction_score is a float
        try:
            prediction = dspy.ChainOfThought(SimpleScorerAgentSignature)(groundtruth=gold.MisconceptionText,
                                                                         prediction=pred)
            score = float(prediction.score.strip())
        except:
            score = 0.01
        return max(0.0, min(1.0, score))

    #def metric(gold, pred, trace=None):
    # db.vector_search(pred, k=1) --> List [Tuple[int class_id, float distance]]
    # sort by distance, lowestance first
    # MAP_25 gold_class_id and np.array[25 prediction class_id] -- score: float
    # MAP_25(List [Tuple[int class_id, float distance]], gold) --> gold.class_id


    # def l2_distance(self, gold, pred, trace=None):
    #     return evaluate_answers(gold, pred, trace)

    # This function is probably not working anymore because it's
    def evaluate_answers(self, gold, pred, trace=None):

        try:
            gold_answer_id = gold.answer
            mis_out = self.summery_agent(pred.question, pred.answer)

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
