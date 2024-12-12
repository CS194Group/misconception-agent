from dataclasses import dataclass
from typing import Tuple
import dspy
import logging
import pdb

from src.agents_component import MisAgent, FinAgent, SolveAgent_api

logging.basicConfig(
    level=logging.WARNING, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    handlers=[
        logging.FileHandler("agents.log"), 
        logging.StreamHandler()
    ]
)

# Agents' data return format
@dataclass
class Misconception:
    misconception_id: float
    misconception: str
    similarity: float = 0.0

###########################################################################################################
# The basic agent

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
        desc='Explaination of the misconception the student has based on his answer in one percise sentences.')

class Agent(dspy.Module):
    def __init__(self, name, persona_promt=None):
        super().__init__()
        self.name = name
        self.prefix_promt = persona_promt
        self.process = dspy.Predict(BaseAgentSignature)

    def forward(self, QuestionText, AnswerText, ConstructName, SubjectName, CorrectAnswer, context=None) -> str:
        # Directly pass the inputs to the process method
        try:
            outputs = self.process(
                context=context,
                QuestionText=QuestionText,
                AnswerText=AnswerText,
                ConstructName=ConstructName,
                SubjectName=SubjectName,
                CorrectAnswer=CorrectAnswer,
                prefix = self.prefix_promt
            )

            return outputs.completions[0].MisconceptionText
        except Exception as e:
            print(e)
            return "Failed to generate misconception explanation."
        
# other architecture of agents (not in use)

class AdvancedAgent(dspy.Module):
    def __init__(self, name, tools=None, persona_promt=None):
        super().__init__()
        self.name = name
        self.prefix_promt = persona_promt

        # TODO Write the prompt
        # self.solve_agent = SolveAgent("solve_agent", tools)
        self.solve_agent = SolveAgent_api("solve_agent")
        self.mis_agent = MisAgent("mis_agent")
        self.fin_agent = FinAgent("fin_agent")

    def forward(self, QuestionText, AnswerText, ConstructName, SubjectName, CorrectAnswer, context=None) -> str:
        # Directly pass the inputs to the process method
        # try:

        answer_reasoning = self.solve_agent(
            context=context,
            QuestionText=QuestionText,
            ConstructName=ConstructName,
            SubjectName=SubjectName,
            CorrectAnswer=CorrectAnswer,
        )

        logging.warning(f"answer_reasoning: {answer_reasoning}")

        misconception_choice = self.mis_agent(
            context=context,
            QuestionText=QuestionText,
            AnswerText=AnswerText,
            ConstructName=ConstructName,
            SubjectName=SubjectName,
            CorrectAnswer=CorrectAnswer,
            CorrectReasoning=answer_reasoning,
        )

        logging.warning(f"misconception_choice: {misconception_choice}")

        misconception = self.fin_agent(
            context=context,
            QuestionText=QuestionText,
            AnswerText=AnswerText,
            ConstructName=ConstructName,
            SubjectName=SubjectName,
            CorrectAnswer=CorrectAnswer,
            CorrectReasoning=answer_reasoning,
            MisconceptionReasoning=misconception_choice,
        )

        return misconception
    
        # except Exception as e:
        #     print(e)
        #     return "Failed to generate misconception explanation."
        

# All code down below not used any more at the moment at least (it will be modified in the future)
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
