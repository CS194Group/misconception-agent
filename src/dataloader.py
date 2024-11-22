import dspy
import pandas as pd

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
