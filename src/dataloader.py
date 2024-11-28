import pathlib

import dspy
import pandas as pd
from sklearn.model_selection import train_test_split


# Data loading part

def load_misconceptions(filepath):
    misconceptions_df = pd.read_csv(filepath)
    return {row['MisconceptionId']: row['MisconceptionName'] for _, row in misconceptions_df.iterrows()}

def load_data(filepath, is_test=False):
    df = pd.read_csv(filepath)
    examples = []
    total_rows = len(df)
    split_index = int(total_rows * 0.8)

    if not is_test:
        df_subset = df.iloc[:split_index]
    else:
        df_subset = df.iloc[split_index:]

    for _, row in df_subset.iterrows():
        misconceptions = [
            row['MisconceptionAId'],
            row['MisconceptionBId'],
            row['MisconceptionCId'],
            row['MisconceptionDId']
        ]
        example = dspy.Example(
            question="The question is: " + row['QuestionText'] + "\nAnd here is the possible answers." + "\nA: " + row['AnswerAText'] + "\nB: " + row['AnswerBText'] +
            "\nC: " + row['AnswerCText'] + "\nD: " + row['AnswerDText'] + "\nThe correct answer is: " + row['CorrectAnswer'],
            answer=misconceptions
        ).with_inputs("question")
        examples.append(example)

    return examples


import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split


class DataManager:

    # Retrieves misconceptions from disk and returns a DataFrame with columns misconception_id, misconception
    @staticmethod
    def _get_misconceptions(file_path: pathlib.Path) -> pd.DataFrame:
        misconceptions = pd.read_csv(file_path)
        # Ensuring columns match the actual file structure
        misconceptions.columns = ['misconception_id', 'misconception']
        return misconceptions

    @staticmethod
    def get_data(data_folder: pathlib.Path, debug=False) -> pd.DataFrame:
        train_file = data_folder / 'train.csv'
        misconception_mapping_file = data_folder / 'misconception_mapping.csv'
        misconception_mapping = DataManager._get_misconceptions(misconception_mapping_file)
        result = DataManager._join_data(train_file, misconception_mapping)
        result = result[result['MisconceptionText'].notna()]
        return  result if not debug else result[:5]

    # Reads data from disk and separates question, answers, and misconceptions
    # Output DataFrame columns are ID, question, answers, and misconceptions (text)
    @staticmethod
    def _join_data(file_path: pathlib.Path, misconception_mapping: pd.DataFrame) -> pd.DataFrame:
        data = pd.read_csv(file_path)

        # Define columns that will be reshaped
        answer_columns = ['AnswerAText', 'AnswerBText', 'AnswerCText', 'AnswerDText']
        misconception_columns = ['MisconceptionAId', 'MisconceptionBId', 'MisconceptionCId', 'MisconceptionDId']
        answer_labels = ['A', 'B', 'C', 'D']

        # Reshape data to have each answer option as a separate row
        rows = []
        for i, row in data.iterrows():
            # Map the CorrectAnswer letter to the corresponding answer text
            correct_answer_letter = row['CorrectAnswer']
            try:
                correct_answer_text = row[f'Answer{correct_answer_letter}Text']
            except KeyError:
                raise ValueError(f"Correct answer letter '{correct_answer_letter}' not found in answer columns")
                #correct_answer_text = None  # Handle unexpected answer letters

            for answer, answer_col, miscon_col in zip(answer_labels, answer_columns, misconception_columns):
                # Get answer text and misconception ID
                answer_text = row[answer_col]
                misconception_id = row[miscon_col]

                # Find misconception text if a misconception ID is present
                misconception_text = misconception_mapping[
                    misconception_mapping['misconception_id'] == misconception_id
                ]['misconception'].values
                misconception_text = misconception_text[0] if len(misconception_text) > 0 else None

                # Append the reshaped row to the list with CorrectAnswer as the actual text
                rows.append({
                    'QuestionId': row['QuestionId'],
                    'ConstructName': row['ConstructName'],
                    'SubjectName': row['SubjectName'],
                    'CorrectAnswer': correct_answer_text,  # Updated to actual answer text
                    'QuestionText': row['QuestionText'],
                    'Answer': answer,
                    'AnswerText': answer_text,
                    'MisconceptionText': misconception_text
                })

        # Convert list of rows into DataFrame
        reshaped_data = pd.DataFrame(rows)
        return reshaped_data

    # Returns test and train data
    @staticmethod
    def test_train_split(data: pd.DataFrame, test_size: float = 0.2) -> (pd.DataFrame, pd.DataFrame):
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
        return train_data, test_data
