# TODO: Add custom questions
# TODO: Connect with the backend
# TODO: Fix the layout

import streamlit as st
import pandas as pd

# Load the CSV data
data = pd.read_csv('./data/test.csv')

# Set up page configuration and styles
st.set_page_config(
    page_title="Multi-Agent Misconception Quiz",
    page_icon="❓",
)

st.markdown("""
<style>
div.stButton > button {
    width: 100%; /* Make buttons take the full width of the parent container */
    margin-top: 5px; /* Add some space between buttons */
}
div.stButton {
    display: flex;
    justify-content: center;
}
</style>
""", unsafe_allow_html=True)

# Session state defaults
default_values = {'current_index': 0,
                  'selected_option': None, 'answer_submitted': False}
for key, value in default_values.items():
    st.session_state.setdefault(key, value)

# Reset the quiz


def restart_quiz():
    st.session_state.current_index = 0
    st.session_state.selected_option = None
    st.session_state.answer_submitted = False

# Submit the selected answer


def submit_answer(correct_answer):
    if st.session_state.selected_option is not None:
        st.session_state.answer_submitted = True
    else:
        st.warning("Please select an option before submitting.")

# Proceed to the next question


def next_question():
    st.session_state.current_index += 1
    st.session_state.selected_option = None
    st.session_state.answer_submitted = False

# Wrap LaTeX expressions


def wrap_latex(text: str) -> str:
    return text.replace("\\[", "$").replace("\\]", "$").replace("\\(", "$").replace("\\)", "$")


# Title
st.title("Multi-Agent Misconception Quiz")

# Load current question
current_row = data.iloc[st.session_state.current_index]
construct_name = current_row['ConstructName']
question_text = wrap_latex(current_row['QuestionText'])
options = {
    "A": wrap_latex(current_row['AnswerAText']),
    "B": wrap_latex(current_row['AnswerBText']),
    "C": wrap_latex(current_row['AnswerCText']),
    "D": wrap_latex(current_row['AnswerDText'])
}
# Assuming 'CorrectAnswer' column specifies the correct answer
correct_answer = current_row.get('CorrectAnswer', 'A')

# Display the question
st.subheader(f"Question {st.session_state.current_index + 1}")
st.markdown(f"**Construct Name:** {construct_name}")
st.markdown(f"**Question:** {question_text}")

# Display answer options as wide buttons
if st.session_state.answer_submitted:
    for key, value in options.items():
        if key == correct_answer:
            st.success(f"{value} (Correct answer)")
        elif key == st.session_state.selected_option:
            st.error(f"{value} (Your choice)")
        else:
            st.write(f"{value}")
else:
    for key, value in options.items():
        if st.button(value, key=key):
            st.session_state.selected_option = key

st.write("---")

# Navigation
if st.session_state.answer_submitted:
    if st.session_state.current_index < len(data) - 1:
        st.button("Next", on_click=next_question)
    else:
        st.write("Quiz completed! Thank you for participating.")
        if st.button("Restart", on_click=restart_quiz):
            pass
else:
    st.button("Submit", on_click=lambda: submit_answer(correct_answer))
