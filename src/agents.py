import dspy
import faiss

import pandas as pd
import numpy as np

from typing import List, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

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

class BaseRefereeAnswer(dspy.Signature):
    """Generates misconception based on input question and correct answer"""
    question = dspy.InputField(desc='Question and answer of the question that you need to find the misconception from.')
    context = dspy.InputField(desc='Debate history of other agents for reference.')
    qustion = dspy.OutputField(desc='Repeat the qestion.')
    answer = dspy.OutputField(desc=BaseAnswerPrompt)

class Agent(dspy.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.process = dspy.ChainOfThought(BaseRefereeAnswer)

    def forward(self, question, context=None) -> dspy.Prediction:
        """Generates the agent's response based on question and optional context."""
        response = self.process(question=question, context=context)
        return response
    
#########################################################################################################################

#########################################################################################################################
# The agent to evaluate the similarity between two sentences

class SemanticSearchModuleRefereeAnswer(dspy.Signature):
    """Generates similarity score and explanation based on input two misconceptions"""
    query = dspy.InputField(desc='Query misconception generate by agent')
    candidate = dspy.InputField(desc='Possible misconception that is similar to the query')
    score = dspy.OutputField(desc='Predict the similarity score between the query and candidate. Just output a score between 0 and 1.')
    explanation = dspy.OutputField(desc="Gives explanation why you think query and candidate has the score")

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
    question = dspy.InputField(desc='Question and answer of the question that you need to find the misconception from.')
    context = dspy.InputField(desc='Debate history of other agents for reference.')
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

#########################################################################################################################
# Retrive model, using SentenceTransformer and SemanticSearchModule

class MisconceptionDB:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        required_columns = ['MisconceptionId', 'MisconceptionName']
        assert all(col in self.df.columns for col in required_columns)

        self.semantic_search = SemanticSearchModule()
        self.encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.init_faiss_index()

    def init_faiss_index(self):
        try:
            texts = self.df['MisconceptionName'].tolist()
            embeddings = self.encoder.encode(texts)

            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)

            self.index.add(np.array(embeddings).astype('float32'))

            self.embeddings = embeddings
            
        except Exception as e:
            raise Exception(f"Error initializing FAISS index: {e}")
        
    def vector_search(self, query: str, k: int = 15) -> List[Tuple[int, float]]:
        query_vector = self.encoder.encode(query)
        
        distances, indices = self.index.search(
            np.array([query_vector]).astype('float32'), 
            k
        )
        
        return list(zip(indices[0], distances[0]))
        
    def hybrid_search(self, query: str, top_k: int = 1, pre_filter_k: int = 5) -> List[Misconception]:
        # First use FAISS to pick pre_filter_k misconception, and then use agent to rerank
        
        vector_results = self.vector_search(query, pre_filter_k)
        
        results = []
        for idx, vector_distance in vector_results:
            row = self.df.iloc[idx]
            
            semantic_score, explanation = self.semantic_search(
                query, 
                row['MisconceptionName']
            )
            
            vector_score = 1 / (1 + vector_distance)
            
            misconception = Misconception(
                misconception_id=float(row['MisconceptionId']),
                misconception=row['MisconceptionName'],
                similarity=0.7 * semantic_score + 0.3 * vector_score
            )
            
            results.append(misconception)

        return sorted(
            results,
            key=lambda x: x.similarity,
            reverse=True
        )[:top_k]
    
#########################################################################################################################
