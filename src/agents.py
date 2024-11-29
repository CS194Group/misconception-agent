import dspy
from dataclasses import dataclass
from typing import Tuple


# Agents' data return format


@dataclass
class Misconception:
    misconception_id: float
    misconception: str
    similarity: float = 0.0

###########################################################################################################
# The basic agent


BaseAnswerPrompt = '''
Output each option's misconceptions in the following format:
(Assume that B is the correct answer)
'A': 'MisconceptionA',
'B': NaN,
'C': 'MisconceptionC',
'D': 'MisconceptionD'
If the option is correct, it's misconception should be NaN.
'''


class BaseAgentSignature(dspy.Signature):
    """Explain the misconception the student has based on his answer."""
    context = dspy.InputField(
        desc='Debate history of other agents for reference. Empty if no history is available.')

    QuestionText = dspy.InputField(desc='The question text.')
    AnswerText = dspy.InputField(desc='The student wrong answer text.')
    ConstructName = dspy.InputField()
    SubjectName = dspy.InputField(desc="The subject of the question.")
    CorrectAnswer = dspy.InputField(desc="The correct answer.")
    MisconceptionText = dspy.OutputField(
        desc='Explaination the misconception the student has based on his answer in maximum two sentences.')

class Agent(dspy.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.process = dspy.Predict(BaseAgentSignature)

    def forward(self, QuestionText, AnswerText, ConstructName, SubjectName, CorrectAnswer, context=None, text_only=True) -> dspy.Prediction:
        # Directly pass the inputs to the process method
        outputs = self.process(
            context=context,
            QuestionText=QuestionText,
            AnswerText=AnswerText,
            ConstructName=ConstructName,
            SubjectName=SubjectName,
            CorrectAnswer=CorrectAnswer
        )
        if text_only:
            return outputs.completions[0].MisconceptionText
        else:
            return outputs


#########################################################################################################################

#########################################################################################################################
# The agent to evaluate the similarity between two sentences


class SemanticSearchModuleRefereeAnswer(dspy.Signature):
    """Generates similarity score and explanation based on input two misconceptions"""
    query = dspy.InputField(desc='Query misconception generate by agent')
    candidate = dspy.InputField(
        desc='Possible misconception that is similar to the query')
    score = dspy.OutputField(
        desc='Predict the similarity score between the query and candidate. Just output a score between 0 and 1.')
    explanation = dspy.OutputField(
        desc="Gives explanation why you think query and candidate has the score")


class SemanticSearchModule(dspy.Module):

    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(SemanticSearchModuleRefereeAnswer)

    def forward(self, query: str, candidate: str) -> Tuple[float, str]:
        result = self.predictor(
            instruction="Evaluate the semantic similarity between the following two texts, providing a score between 0 and 1 along with an explanation. Consider factors such as conceptual relevance and thematic overlap.",
            query=query,
            candidate=candidate
        )
        try:
            score = float(result.score)
            explanation = result.explanation
        except:
            score = 0.0
            explanation = "Failed to calculate similarity score."
        return max(0.0, min(1.0, score)), explanation

#########################################################################################################################

#########################################################################################################################
# Pick out misconception of each option, based on the basic agents' output


class SummeryRefereeAnswer(dspy.Signature):
    """Generates misconception based on input question and correct answer"""
    question = dspy.InputField(
        desc='Question and answer of the question that you need to find the misconception from.')
    context = dspy.InputField(
        desc='Debate history of other agents for reference.')
    # answer = dspy.OutputField(desc='Index of top 5 most relevant misconceptions in the misconception_mapping.csv')
    misconceptionA = dspy.OutputField(desc='Misconception of option A.')
    misconceptionB = dspy.OutputField(desc='Misconception of option B.')
    misconceptionC = dspy.OutputField(desc='Misconception of option C.')
    misconceptionD = dspy.OutputField(desc='Misconception of option D.')


class SummeryAgent(dspy.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.process = dspy.ChainOfThought(SummeryRefereeAnswer)

    def forward(self, question, context=None) -> dspy.Prediction:
        """Generates the agent's response based on question and optional context."""
        output = self.process(question=question, context=context)
        return output

#########################################################################################################################
