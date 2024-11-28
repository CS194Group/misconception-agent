import pathlib
import pickle
from typing import Tuple, List

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.agents import SemanticSearchModule, Misconception

#########################################################################################################################
# Retrive model, using SentenceTransformer and SemanticSearchModule

#########################################################################################################################
class MisconceptionDB:
    def __init__(self, csv_path: pathlib.Path):
        """
        Initializes the MisconceptionDB with data from a CSV file.

        Args:
            csv_path (pathlib.Path): Path to the CSV file containing misconception data.
                                       The CSV must include 'MisconceptionId' and 'MisconceptionName' columns.

        Raises:
            AssertionError: If the required columns are not present in the CSV.
            Exception: If there is an error initializing the FAISS index.
        """
        self.embeddings = None
        self.index = None
        self.df = pd.read_csv(csv_path)
        required_columns = ['MisconceptionId', 'MisconceptionName']
        assert all(col in self.df.columns for col in required_columns)

        self.semantic_search = SemanticSearchModule()
        self.encoder = SentenceTransformer(
            'paraphrase-multilingual-MiniLM-L12-v2')

        self.index, self.embeddings = self.init_faiss_index()

    def init_faiss_index(self) -> Tuple[faiss.Index, np.ndarray]:
        """
        Initializes the FAISS index using the misconception names.

        This method encodes the misconception names into embeddings, creates a FAISS index
        with L2 distance metric, and adds the embeddings to the index.

        Raises:
           Exception: If there is an error during the initialization of the FAISS index.
        """
        try:
            misconception_texts = self.df['MisconceptionName'].tolist()
            misconception_texts.sort()

            # caching embeddings to disk
            pickle_file = pathlib.Path.cwd() / pathlib.Path('data/db.pkl')
            if pickle_file.exists():
                with pickle_file.open('rb') as f:
                    embeddings = pickle.load(f)
            else:
                embeddings = self.encoder.encode(misconception_texts)  # Replace with your function
                with pickle_file.open('wb') as f:
                    pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(np.array(embeddings).astype('float32'))

            return index, embeddings

        except Exception as e:
            raise Exception(f"Error initializing FAISS index: {e}")

    def vector_search(self, query: str, k: int = 15) -> List[Tuple[int, float]]:
        """
        Performs a vector-based search to find the top k closest misconceptions to the query.

        Args:
           query (str): The search query string.
           k (int, optional): The number of top results to return. Defaults to 15.

        Returns:
           List[Tuple[int, float]]: A list of tuples where each tuple contains the index
                                    of the misconception in the DataFrame and its corresponding
                                    L2 distance from the query.
                                    Lower distances indicate higher similarity.
        """
        query_vector = self.encoder.encode(query)

        distances, indices = self.index.search(
            np.array([query_vector]).astype('float32'),
            k
        )

        return list(zip(indices[0], distances[0]))

    def hybrid_search(self, query: str, top_k: int = 1, pre_filter_k: int = 5) -> List[Misconception]:
        """
        Performs a hybrid search combining vector-based and semantic-based ranking.

        Steps:
           1. Uses FAISS to retrieve the top `pre_filter_k` closest misconceptions based on vector similarity.
           2. Reranks these misconceptions using a semantic scoring mechanism.
           3. Combines both scores to compute a final similarity score.
           4. Returns the top `top_k` misconceptions based on the combined similarity.

        Args:
           query (str): The search query string.
           top_k (int, optional): The number of top results to return after reranking. Defaults to 1.
           pre_filter_k (int, optional): The number of initial results to retrieve using vector search.
                                          Defaults to 5.

        Returns:
           List[Misconception]: A list of Misconception objects sorted by their combined similarity score
                                in descending order.
                                Higher similarity scores indicate more relevant misconceptions.
        """
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
