import pathlib
import pickle
from typing import Tuple, List

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.agents import Misconception

#########################################################################################################################
# Retrive model, using SentenceTransformer and SemanticSearchModule

#########################################################################################################################
class MisconceptionDB:
    def __init__(self, csv_path: pathlib.Path, encoder_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
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

        # self.semantic_search = SemanticSearchModule()
        self.encoder = SentenceTransformer(encoder_name)
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

            # caching embeddings to disk
            pickle_file = pathlib.Path.cwd() / pathlib.Path('data/db.pkl')
            if pickle_file.exists():
                with pickle_file.open('rb') as f:
                    embeddings = pickle.load(f)
            else:
                embeddings = self.encoder.encode(misconception_texts)  # Replace with your function
                with pickle_file.open('wb') as f:
                    pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

            embeddings = np.array(embeddings, dtype='float32')
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            faiss.normalize_L2(embeddings)
            index.add(embeddings)

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
        query_vector = np.array([query_vector], dtype='float32')
        faiss.normalize_L2(query_vector)

        distances, indices = self.index.search(
            query_vector,
            k
        )

        return sorted(
            list(zip(indices[0], distances[0])),
            key=lambda x: x[1]
        )


    def replace_ids_with_texts(self, search_results: List[Tuple[int, float]]) -> List[
        Tuple[str, float]]:
        """
        Replaces the indices in the search results with the corresponding misconception texts.

        Args:
            search_results (List[Tuple[int, float]]): A list of tuples where each tuple contains:
                                                      - An index of the misconception in the dataframe.
                                                      - Its corresponding similarity score or distance.
            misconception_db (MisconceptionDB): An instance of the MisconceptionDB containing the data.

        Returns:
            List[Tuple[str, float]]: A list of tuples where each tuple contains:
                                     - The misconception text.
                                     - Its corresponding similarity score or distance.
        """

        # Map the indices to texts
        replaced_results = [
            (self.df.iloc[idx]['MisconceptionName'], distance)
            for idx, distance in search_results
        ]

        return replaced_results

    # def hybrid_search(self, query: str, top_k: int = 3, pre_filter_k: int = 25) -> List[Misconception]:
    #     """
    #     Performs a hybrid search combining vector-based and semantic-based ranking.
    #
    #     Steps:
    #        1. Uses FAISS to retrieve the top `pre_filter_k` closest misconceptions based on vector similarity.
    #        2. Reranks these misconceptions using a semantic scoring mechanism.
    #        3. Combines both scores to compute a final similarity score.
    #        4. Returns the top `top_k` misconceptions based on the combined similarity.
    #
    #     Args:
    #        query (str): The search query string.
    #        top_k (int, optional): The number of top results to return after reranking. Defaults to 1.
    #        pre_filter_k (int, optional): The number of initial results to retrieve using vector search.
    #                                       Defaults to 5.
    #
    #     Returns:
    #        List[Misconception]: A list of Misconception objects sorted by their combined similarity score
    #                             in descending order.
    #                             Higher similarity scores indicate more relevant misconceptions.
    #     """
    #     # First use FAISS to pick pre_filter_k misconception, and then use agent to rerank
    #
    #     vector_results = self.vector_search(query, pre_filter_k)
    #
    #     results = []
    #     for idx, vector_distance in vector_results:
    #         row = self.df.iloc[idx]
    #
    #         semantic_score, explanation = self.semantic_search(
    #             query,
    #             row['MisconceptionName']
    #         )
    #
    #         vector_score = 1 / (1 + vector_distance)
    #
    #         misconception = Misconception(
    #             misconception_id=float(row['MisconceptionId']),
    #             misconception=row['MisconceptionName'],
    #             similarity=0.7 * semantic_score + 0.3 * vector_score
    #         )
    #
    #         results.append(misconception)
    #
    #     return sorted(
    #         results,
    #         key=lambda x: x.similarity,
    #         reverse=True
    #     )[:top_k]

    def calculate_l2_distance(self, query: str, true_class_id: int) -> float:
        """
        Calculates the L2 distance between a query embedding and the true misconception class embedding.

        Args:
            query (str): The input query string.
            true_class_id (int): The MisconceptionId of the true misconception class.

        Returns:
            float: The L2 distance between the query embedding and the true misconception class embedding.

        Raises:
            ValueError: If the true_class_id is not found in the database.
        """
        if true_class_id not in self.df['MisconceptionId'].values:
            raise ValueError(f"True class ID {true_class_id} not found in the database.")

        # Get the embedding for the query
        query_vector = self.encoder.encode(query)
        query_vector = np.array([query_vector], dtype='float32')
        faiss.normalize_L2(query_vector)

        # Retrieve the embedding for the true misconception class
        true_class_index = self.df.index[self.df['MisconceptionId'] == true_class_id][0]
        true_class_embedding = self.embeddings[true_class_index].astype('float32')

        # Reshape vectors to 2D arrays as required by Faiss
        query_vector = query_vector.reshape(1, -1)
        true_class_embedding = true_class_embedding.reshape(1, -1)

        # Compute the squared L2 distance using Faiss
        distances = faiss.pairwise_distances(query_vector, true_class_embedding, metric=faiss.METRIC_L2)
        #l2_distance = np.sqrt(distances[0][0])  # Take the square root to get the L2 distance
        return float(distances[0][0])