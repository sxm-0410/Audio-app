import streamlit as st
from pathlib import Path

st.set_page_config(page_title="AuraLift", layout="centered")

# 💅 Custom CSS for layout and buttons
st.markdown(
    """
    <style>
    .feature-button-container {
        display: flex;
        justify-content: space-between;
        margin-top: 2rem;
    }
    .feature-button {
        flex: 1;
        margin: 0 10px;
    }
    .feature-button button {
        width: 100%;
        height: 60px;
        font-size: 16px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎧 AuraLift")
st.subheader("AI-powered music manipulation and generation")
st.write("AuraLift is an all-in-one AI toolkit empowering anyone to easily create, enhance, and transform audio.")

st.markdown("### Choose a Feature:")

# 👇 Use correct lowercase path for Streamlit's multi-page routing
with st.container():
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("🎤 Vocal & Non-Vocal Separation", use_container_width=True):
            st.switch_page(str(Path("pages") / "1_Vocal_and_NonVocal_Separation.py"))

    with col2:
        if st.button("🎛️ Vocal processing", use_container_width=True):
            st.switch_page(str(Path("pages") / "2_Vocal_Processing.py"))

    with col3:
        if st.button("🎶 Generate AI Music", use_container_width=True):
            st.switch_page(str(Path("pages") / "3_Generate_AI_Music.py"))
