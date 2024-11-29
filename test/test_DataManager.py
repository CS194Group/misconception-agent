import pathlib
import pytest
from src.dataloader import DataManager


def test_answer_text_and_correct_answer_difference():
    # Load the DataFrame
    data_path = pathlib.Path(__file__).parent.parent / "data"
    df = DataManager.get_data(data_path, debug=True)

    # Ensure the DataFrame contains the required columns
    assert 'AnswerText' in df.columns, "Column 'AnswerText' is missing."
    assert 'CorrectAnswer' in df.columns, "Column 'CorrectAnswer' is missing."

    # Check that the columns are always different
    differing_rows = df[df['AnswerText'] == df['CorrectAnswer']]

    assert differing_rows.empty, f"Found rows where 'AnswerText' and 'CorrectAnswer' are the same:\n{differing_rows}"


if __name__ == "__main__":
    pytest.main()
