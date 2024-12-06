import dspy
import sys
import os
import re
import pdb

import streamlit as st
import pandas as pd

from openai import OpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from predict_model import ExchangeOfThought
from agents import Agent

# initialize
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(
    api_key = OPENAI_API_KEY,
)
if not OPENAI_API_KEY:
    pdb.set_trace()
    raise EnvironmentError(
        "OPENAI_API_KEY not found in environment variables.")
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
lm = dspy.LM('openai/gpt-3.5-turbo')
dspy.configure(lm=lm)

class QuizApp:
    def __init__(self, q_data_path='./data/train.csv', mis_data_path='./data/misconception_mapping.csv'):
        """Initialize the quiz application with modern styling."""
        # Configure page with wider layout
        st.set_page_config(
            page_title="Multi-Agent Misconception Quiz",
            page_icon="❓",
            layout="centered"
        )

        # Custom CSS for a more modern and clean look
        st.markdown("""
        <style>
        .stApp {
            background-color: #f0f4f8;
            font-family: 'Roboto', sans-serif;
        }
        .stContainer {
            background-color: white;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 20px;
            margin-bottom: 20px;
        }
        .stButton>button {
            background-color: #4A90E2;
            color: white;
            border-radius: 10px;
            border: none;
            padding: 10px 20px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #357ABD;
            transform: scale(1.05);
        }
        .stRadio>div {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        h1, h2, h3 {
            color: #2C3E50;
            font-weight: 600;
        }
        .misconception-option {
            transition: all 0.3s ease;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
        }
        .misconception-option:hover {
            transform: translateX(10px);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        </style>
        """, unsafe_allow_html=True)

        # Title with gradient effect
        st.markdown("""
        <h1 style="
            background: linear-gradient(to right, #4A90E2, #50C878);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            padding: 20px;
        ">Multi-Agent Misconception Quiz</h1>
        """, unsafe_allow_html=True)

        st.markdown("## Test your knowledge of multi-agent systems with this interactive quiz! 🧠", unsafe_allow_html=True)

        # Load custom CSS
        # self._load_custom_css()

        # Set up Agents
        agent_a = Agent(name="Agent A")
        agent_b = Agent(name="Agent B")
        agent_c = Agent(name="Agent C")

        # evaluate
        self.model = ExchangeOfThought(agent_a, agent_b, agent_c, rounds=3, mode="Report")
        self.model.load('./compiled_model.pkl')

        # Load data
        self.data = pd.read_csv(q_data_path)

        self.mis_data = pd.read_csv(mis_data_path)

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
            'answer_submitted': False,
            'balloon_shown': False
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
        current_row = self.data.iloc[st.session_state.current_index + 1]
        return {
            'question_text': self._wrap_latex(current_row['QuestionText']),
            'options': {
                "A": self._wrap_latex(current_row['AnswerAText']),
                "B": self._wrap_latex(current_row['AnswerBText']),
                "C": self._wrap_latex(current_row['AnswerCText']),
                "D": self._wrap_latex(current_row['AnswerDText'])
            },
            'correct_answer': current_row.get('CorrectAnswer', 'A'),
            'misconceptions': {
                "A": current_row.get('MisconceptionAId'),
                "B": current_row.get('MisconceptionBId'),
                "C": current_row.get('MisconceptionCId'),
                "D": current_row.get('MisconceptionDId'),
            }
        }
    
    def _get_current_misconception(self, index):
        misconceptions = {}
        for keys, value in index.items():
            if pd.isna(value):
                misconceptions[keys] = "There's no apparent misconception."
            
            try:
                int_index = int(value)
                misconceptions[keys] = self.mis_data.iloc[int_index]['MisconceptionName']
            except (ValueError, TypeError):
                misconceptions[keys] = "There's no apparent misconception."

        return misconceptions

    def _select_answer(self, selected_key):
        """Handle answer selection and submission."""
        st.session_state.selected_option = selected_key
        st.session_state.answer_submitted = True

    def _next_question(self):
        """Move to the next question."""
        st.session_state.current_index += 1
        st.session_state.selected_option = None
        st.session_state.answer_submitted = False
        st.session_state.balloon_shown = False

    def _restart_quiz(self):
        """Reset the quiz to its initial state."""
        st.session_state.current_index = 0
        st.session_state.selected_option = None
        st.session_state.answer_submitted = False
        st.session_state.balloon_shown = False

    def _ini_miscon(self, misconception_container, option):
        misconception_container.markdown(f"""
            <div style="background-color: white; 
                        padding: 20px; 
                        border-radius: 10px; 
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        margin-bottom: 15px;">
                See the {option}'s misconception after you make your choice!.
            </div>
        """, unsafe_allow_html=True)

    def _update_miscon(self, misconception_container, misconception, option):
        misconception_container.markdown(f"""
            <div style="background-color: white; 
                        padding: 20px; 
                        border-radius: 10px; 
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        margin-bottom: 15px;">
                <span style="font-size: 20px; font-weight: bold;">
                    {option}'s misconception:
                </span> 
                {misconception}.
            </div>
        """, unsafe_allow_html=True)

    def create_misconception_display(self, misconception_dict):
        # Custom CSS for detailed explanation
        st.markdown("""
        <style>
        .explanation-card {
            background: linear-gradient(135deg, #f6f8f9 0%, #e5ebee 100%);
            border-radius: 15px;
            padding: 25px;
            margin-top: 20px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
            transition: all 0.3s ease;
            border-left: 5px solid #1a73e8;
        }
        .explanation-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(0, 0, 0, 0.1);
        }
        .explanation-title {
            color: #1a73e8;
            font-weight: 700;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
        }
        .explanation-title svg {
            margin-right: 10px;
        }
        .explanation-content {
            color: #2c3e50;
            line-height: 1.6;
        }
        .stRadio > div {
            background-color: #f1f3f5;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
        }
        .stRadio div[role="radiogroup"] > div {
            background-color: white;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            margin-bottom: 10px;
            padding: 10px;
            transition: all 0.3s ease;
        }
        .stRadio div[role="radiogroup"] > div:hover {
            background-color: #f8f9fa;
            border-color: #1a73e8;
        }
        </style>
        """, unsafe_allow_html=True)

        st.subheader("🔍 Select a Misconception:")
        selected_misconception = st.radio(
            "",
            list(misconception_dict.keys()),
            format_func=lambda x: misconception_dict[x]['text'],
            key="misconception_selector"
        )

        # Detailed explanation with enhanced styling
        st.markdown(f"""
        <div class="explanation-card">
            <div class="explanation-title">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 20h9"></path>
                    <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"></path>
                </svg>
                <h3>Detailed Explanation</h3>
            </div>
            <div class="explanation-content">
                <p>{misconception_dict[selected_misconception]['explanation']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Optional: Add an expand/collapse section for more details
        with st.expander("🌟 Understand the Reasoning"):
            st.markdown("""
            ### Key Insights
            - Carefully analyze the mathematical reasoning
            - Identify the specific error in logical thinking
            - Learn how to avoid similar misconceptions
            """)

        return selected_misconception

    def run(self):
        """Main application runner."""

        if "openai_model" not in st.session_state:
            st.session_state["openai_model"] = "gpt-3.5-turbo"

        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "system", "content": "Find the misconception of the question."}]

        question = self._get_current_question()

        misconception = self._get_current_misconception(question['misconceptions'])

        # Display question text
        # st.subheader(f"Question")
        # container = st.container(border=True)
        # container.write(f"{question['question_text']}")
        st.markdown("""
        <div class="stContainer" style="
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-left: 5px solid #4A90E2;
        ">
        """, unsafe_allow_html=True)
        st.subheader("📝 Current Question")
        st.write(f"{question['question_text']}")
        st.markdown("</div>", unsafe_allow_html=True)

        # Handle answer display and selection
        if st.session_state.answer_submitted:

            if st.session_state.selected_option == question['correct_answer']:
                if not st.session_state.balloon_shown:
                    st.balloons()
                    st.session_state.balloon_shown = True

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

        st.markdown("""
        <div class="stContainer" style="
            background: linear-gradient(135deg, #ffffff 0%, #f1f2f6 100%);
            border-left: 5px solid #8e44ad;
        ">
        """, unsafe_allow_html=True)
        st.subheader("🕵️ Golden Misconception Analysis")

        misconception_container = {}
        for option in ['A', 'B', 'C', 'D']:
            misconception_container[option] = st.empty()
            self._ini_miscon(misconception_container[option], option)

        # Get answer from gpt and our model
        pred = self.model(question['question_text'])

        if st.session_state.answer_submitted:
            for option in ['A', 'B', 'C', 'D']:
                self._update_miscon(misconception_container[option], misconception[option], option)

            misconception_pattern = r"(?:'A':|'B':|'C':|'D':)\s*(?:(?:'Misconception:\s*(.*?)')|NaN)"
            matches = re.findall(misconception_pattern, pred.answer)

            result = []
            for i, match in enumerate(matches):
                option = chr(65 + i)
                if match:
                    result.append(f"{option}: {match}")
                else:
                    result.append(f"{option}: This options has no apparent misconception.")

            # Misconception dictionary with more detailed descriptions
            misconception_dict = {
                "A": {
                    "text": question['options']['A'],
                    "explanation": result[0]
                },
                "B": {
                    "text": question['options']['B'],
                    "explanation": result[1]
                },
                "C": {
                    "text": question['options']['C'],
                    "explanation": result[2]
                },
                "D": {
                    "text": question['options']['D'],
                    "explanation": result[3]
                }
            }

            selected_misconception = self.create_misconception_display(misconception_dict)

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
