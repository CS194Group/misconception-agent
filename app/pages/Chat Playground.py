import dspy
import sys
import os
import pdb

import streamlit as st

from openai import OpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from predict_model import ExchangeOfThought
from agents import Agent

class QuizApp:
    def __init__(self):
        """Initialize the quiz application with modern styling."""
        # Configure page with wider layout
        self._setup_page_config()

        # st.set_page_config(page_title="Chat Playground", page_icon="🤖")
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
        self.client = OpenAI(
            api_key = OPENAI_API_KEY,
        )
        if not OPENAI_API_KEY:
            pdb.set_trace()
            raise EnvironmentError(
                "OPENAI_API_KEY not found in environment variables.")
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

        # Load custom CSS
        self._load_custom_css()

        # Set up Agents
        lm = dspy.LM('openai/gpt-3.5-turbo')
        dspy.configure(lm=lm)
        agent_a = Agent(name="Agent A")
        agent_b = Agent(name="Agent B")
        agent_c = Agent(name="Agent C")
        self.model = ExchangeOfThought(agent_a, agent_b, agent_c, rounds=3, mode="Report")

        self.option_a = ''
        self.option_b = ''
        self.option_c = ''
        self.option_d = ''

        # evaluate
        self.model.load('./compiled_model.pkl')

    def _setup_page_config(self):
        """Configure page settings and style."""
        st.set_page_config(
            page_title="Chat Playground", 
            page_icon="🧩", 
            layout="wide"
        )
        # Custom CSS for enhanced styling
        st.markdown("""
        <style>
        .stApp {
            background-color: #f0f2f6;
        }
        .stTextArea, .stTextInput {
            background-color: white;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 10px 20px;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .highlight-box {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        </style>
        """, unsafe_allow_html=True)

    # def _load_custom_css(self):
    #     """Load custom CSS for a modern look."""
    #     with open('./app/styles.css') as f:
    #         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    def _load_custom_css(self):
        """Load custom CSS for styling the Streamlit app with enhanced font styling."""
        st.markdown("""
        <style>
        /* Background and overall app styling */
        .stApp {
            background-color: #f0f2f6;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        /* Input areas styling */
        .stTextArea textarea, .stTextInput input {
            background-color: white;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
            font-family: 'Comic Sans MS', cursive, sans-serif;
            font-size: 16px;
            color: #333;
            padding: 10px;
            transition: all 0.3s ease;
        }

        /* Input areas focus state */
        .stTextArea textarea:focus, .stTextInput input:focus {
            border-color: #4CAF50;
            box-shadow: 0 0 5px rgba(76, 175, 80, 0.5);
            outline: none;
        }

        /* Placeholder text styling */
        .stTextArea textarea::placeholder, .stTextInput input::placeholder {
            color: #888;
            font-style: italic;
        }

        /* Button styling */
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 10px 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-weight: bold;
            transition: background-color 0.3s ease;
        }

        .stButton>button:hover {
            background-color: #45a049;
        }

        /* Additional styling for labels and headers */
        .stMarkdown h4 {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #2c3e50;
            font-weight: 600;
        }
        </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def _wrap_latex(text: str) -> str:
        """Convert LaTeX delimiters to Markdown-friendly format."""
        return text.replace("\\[", "$").replace("\\]", "$") \
                   .replace("\\(", "$").replace("\\)", "$")
    
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

    def run(self):
        """Main application runner."""
        # Title and description
        st.title("🧩 Chat Playground")
        st.markdown("**Explore and uncover hidden misconceptions in your questions!**")

        # Initialize session state
        if "openai_model" not in st.session_state:
            st.session_state["openai_model"] = "gpt-3.5-turbo"
        
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "system", "content": "Find the misconception of the question."}
            ]

        # Create columns with custom layout
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("#### 📝 Enter Your Question")
            user_question = st.text_area("", height=150, key="question_input")
            # st.info(r"Example input: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \)")
            st.markdown("Example input:\n Simplify the following, if possible:")
            st.latex(r"\frac{m^{2}+2m-3}{m-3}")

        with col2:
            st.markdown("#### 🔍 Multiple Choice Options")
            with st.container(border=True):
                self.option_a = st.text_input("Option A", key="option_a")
                self.option_b = st.text_input("Option B", key="option_b")
                self.option_c = st.text_input("Option C", key="option_c")
                self.option_d = st.text_input("Option D", key="option_d")

        # Misconception section
        st.markdown("## 💡 Misconception Analysis")
        misconception_container = st.empty()
        self._ini_miscon(misconception_container, 'Misconception Insights')

        # Generate Response Button
        if st.button('🚀 Generate Insights', use_container_width=True):
            if user_question:
                # Prepare full question with options
                full_question = f"""
                Question: {user_question}
                Options:
                A: {self.option_a}
                B: {self.option_b}
                C: {self.option_c}
                D: {self.option_d}

                Please analyze and provide misconceptions of all the options, skip if there's no option provided.
                """

                try:
                    # Analyze misconceptions
                    pred_p = self.model(full_question)
                    self._update_miscon(misconception_container, pred_p.answer, 'Misconception Insights')

                    # Chat functionality
                    if full_question:
                        st.session_state.messages.append({"role": "user", "content": full_question})
                        
                        with st.chat_message("user"):
                            st.markdown(full_question)
                        
                        with st.chat_message("assistant"):
                            stream = self.client.chat.completions.create(
                                model=st.session_state["openai_model"],
                                messages=[
                                    {"role": m["role"], "content": m["content"]}
                                    for m in st.session_state.messages
                                ],
                                stream=True,
                            )
                            response = st.write_stream(stream)
                        
                        st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.warning("🚨 Please input a question!")

        # Add some visual flair
        st.markdown("---")
        st.markdown("*Powered by AI Misconception Analysis* 🤖✨")


if __name__ == "__main__":
    quiz_app = QuizApp()
    quiz_app.run()
