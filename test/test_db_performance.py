import pathlib
from pathlib import Path

from src.db import MisconceptionDB

import numpy as np
np.random.seed(42)
import random
random.seed(42)

db = MisconceptionDB(pathlib.Path(__file__).parent.parent / Path("data/misconception_mapping.csv"), "paraphrase-mpnet-base-v2")

class_id = 20
simulated_model_query = "Thinks the whole number portion of a mixed number is multiplied by its fractional part."
result0 = db.calculate_l2_distance("Believes the number of wholes in a mixed number multiplies by the fraction part", class_id)
result1 = db.calculate_l2_distance(simulated_model_query, class_id)
result2 = db.calculate_l2_distance("Student thinks that any two angles along a straight line are equal", class_id) # other misconception
result3 = db.vector_search(simulated_model_query, 25)
result3a = db.replace_ids_with_texts(result3)
# result5 = db.hybrid_search(simulated_model_query, 3)
# result5a = db.replace_ids_with_texts(result5)
print("END")