import streamlit as st
import pandas as pd


class QuizApp:
    def __init__(self, data_path='./data/test.csv'):
        """Initialize the quiz application with modern styling."""
        # Configure page with wider layout
        st.set_page_config(
            page_title="Multi-Agent Misconception Quiz",
            page_icon="❓",
        )

        # Load custom CSS
        self._load_custom_css()

        # Load data
        self.data = pd.read_csv(data_path)

        # Initialize session state
        self._initialize_session_state()

    def _load_custom_css(self):
        """Load custom CSS for a modern look."""
        with open('./app/styles.css') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    def _initialize_session_state(self):
        """Initialize or reset session state variables."""
        default_values = {
            'current_index': 0,
            'selected_option': None,
            'answer_submitted': False
        }
        for key, value in default_values.items():
            st.session_state.setdefault(key, value)

    @staticmethod
    def _wrap_latex(text: str) -> str:
        """Convert LaTeX delimiters to Markdown-friendly format."""
        return text.replace("\\[", "$").replace("\\]", "$") \
                   .replace("\\(", "$").replace("\\)", "$")

    def _get_current_question(self):
        """Retrieve the current question from the dataset."""
        current_row = self.data.iloc[st.session_state.current_index]
        return {
            'question_text': self._wrap_latex(current_row['QuestionText']),
            'options': {
                "A": self._wrap_latex(current_row['AnswerAText']),
                "B": self._wrap_latex(current_row['AnswerBText']),
                "C": self._wrap_latex(current_row['AnswerCText']),
                "D": self._wrap_latex(current_row['AnswerDText'])
            },
            'correct_answer': current_row.get('CorrectAnswer', 'A')
        }

    def _select_answer(self, selected_key):
        """Handle answer selection and submission."""
        st.session_state.selected_option = selected_key
        st.session_state.answer_submitted = True

    def _next_question(self):
        """Move to the next question."""
        st.session_state.current_index += 1
        st.session_state.selected_option = None
        st.session_state.answer_submitted = False

    def _restart_quiz(self):
        """Reset the quiz to its initial state."""
        st.session_state.current_index = 0
        st.session_state.selected_option = None
        st.session_state.answer_submitted = False

    def run(self):
        """Main application runner."""
        st.title("Multi-Agent Misconception")
        st.markdown(
            "Test your knowledge of multi-agent systems with this quiz!")

        # Get current question details
        question = self._get_current_question()

        # Display question text
        st.subheader(f"Question")
        container = st.container(border=True)
        container.write(f"{question['question_text']}")

        # Handle answer display and selection
        if st.session_state.answer_submitted:
            for key, value in question['options'].items():
                if key == question['correct_answer']:
                    st.markdown(f"""
                    <div style="background-color: #e6f2e6; color: black; 
                                padding: 10px; margin-bottom: 10px; 
                                border-radius: 10px; border: 2px solid green;">
                    {value} (Correct answer)
                    </div>
                    """, unsafe_allow_html=True)
                elif key == st.session_state.selected_option:
                    st.markdown(f"""
                    <div style="background-color: #f2e6e6; color: black; 
                                padding: 10px; margin-bottom: 10px; 
                                border-radius: 10px; border: 2px solid red;">
                    {value} (Your choice)
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color: #f0f0f0; color: black; 
                                padding: 10px; margin-bottom: 10px; 
                                border-radius: 10px;">
                    {value}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            for key, value in question['options'].items():
                if st.button(value, key=f"option_{key}",
                             on_click=self._select_answer,
                             args=(key,)):
                    pass

        # Misconception container
        st.subheader("Misconception")
        misconception_container = st.container()
        misconception_container.markdown("""
            <div style="background-color: white; 
                        padding: 20px; 
                        border-radius: 10px; 
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                This is a placeholder for the misconception.
            </div>
        """, unsafe_allow_html=True)

        # Navigation buttons
        if st.session_state.answer_submitted:
            col1, col2 = st.columns(2)
            with col1:
                if st.session_state.current_index < len(self.data) - 1:
                    st.button("Next", on_click=self._next_question)
            with col2:
                if st.session_state.current_index == len(self.data) - 1:
                    st.button("Restart Quiz", on_click=self._restart_quiz)


# Run the application
if __name__ == "__main__":
    quiz_app = QuizApp()
    quiz_app.run()
