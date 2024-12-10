import streamlit as st

def main():
    st.set_page_config(
        page_title="Misconception Detect Agent",
        page_icon="🔍",
        layout="wide"
    )

    st.markdown("""
    <style>
    /* Global Styling */
    body {
        background-color: #f4f6f9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    }

    /* Main Container Styling */
    .main {
        max-width: 900px;
        margin: 0 auto;
        padding: 2rem;
        background-color: white;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
        border-radius: 15px;
    }

    /* Title Styling */
    h1 {
        color: #1a73e8;
        font-weight: 700;
        text-align: center;
        margin-bottom: 30px;
        background: linear-gradient(45deg, #1a73e8, #6a11cb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Radio Button Styling */
    .stRadio > div {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e9ecef;
        transition: all 0.3s ease;
    }

    .stRadio > div:hover {
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        transform: translateY(-5px);
    }

    /* Custom Checkbox Styling */
    .stRadio div[role="radiogroup"] > div {
        background-color: white;
        border-radius: 8px;
        margin-bottom: 10px;
        padding: 10px;
        transition: background-color 0.3s ease;
    }

    .stRadio div[role="radiogroup"] > div:hover {
        background-color: #f1f3f5;
    }

    /* Information Box Styling */
    .stAlert {
        border-radius: 10px;
        background-color: #e6f2ff;
        border-left: 5px solid #1a73e8;
    }
    </style>
    """, unsafe_allow_html=True)

    # Animated Title
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="font-size: 2.5em; animation: pulse 2s infinite;">
            🧠 Misconception Detect Agent
        </h1>
    </div>
    <style>
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    </style>
    """, unsafe_allow_html=True)

    st.info("🕵️ Explore and Identify Mathematical Misconceptions! Select a scenario to uncover hidden reasoning errors.")

    st.success("Navigate through different misconceptions using the radio buttons below.")

    if st.checkbox("Show Detailed Instructions", value=True):
        st.markdown("""
        ### 🎯 How to Use the Misconception Detector
        1. Navigate to "Data Analysis" to see our experiments results and "Multi-Agent Misconception Quiz" to see examples!
        2. Navigate to "Chat playground" to test your own math problems!
        3. Let our agent to analyze the reasoning behind the mathematical error!
        """)

    col1, col2 = st.columns([1, 1])

    with col1:
        # Image selection for first column
        st.markdown("<div class='image-container'>", unsafe_allow_html=True)
        robo_selected = st.image("./app/images/robo.webp", width=1000, output_format="auto")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        # Image selection for second column
        st.markdown("<div class='image-container'>", unsafe_allow_html=True)
        chat_selected = st.image("./app/images/chat.png", width=1000, output_format="auto")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # st.write(f"Current Analysis: {analysis_type} | Difficulty: {difficulty}")


if __name__ == "__main__":
    main()