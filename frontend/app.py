# frontend/app.py
import sys
import os

# Add the project root directory to sys.path (so `crew.*` imports work when running Streamlit)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import streamlit as st
from crew.main import kickoff_query  # expects signature: kickoff_query(query: str)

# ----------------- Page config -----------------
st.set_page_config(
    page_title="Canada AI Strategy Chatbot",
    page_icon="🇨🇦",
    layout="wide",
)
st.title("🇨🇦 Canada AI Strategy & Readiness Chatbot")

# ----------------- Sidebar -----------------
with st.sidebar:
    st.subheader("How to use")
    st.markdown(
        "- Ask questions about Canada's AI strategy, readiness, capacity, and competitiveness.\n"
        "- Answers are grounded in the embedded Global AI methodology PDF and the AI-score CSV "
        "retrieved through a vector database.\n"
        "- If you update the PDF or CSV, re-run your ingest script (e.g., `python ingest.py`)."
    )

    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.history = []

# ----------------- Chat state -----------------
if "history" not in st.session_state:
    # list[dict]: {"role": "user"|"assistant", "content": str}
    st.session_state.history = []

# ----------------- Chat input -----------------
prompt = st.chat_input("Ask about Canada's AI strategy, readiness, and competitiveness...")
if prompt:
    # show user message immediately
    st.session_state.history.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        try:
            # Calls the CrewAI pipeline (Canada AI Strategy Analyst using RAG)
            answer = kickoff_query(prompt)
        except Exception as e:
            answer = f"Sorry, something went wrong: `{e}`"

    st.session_state.history.append({"role": "assistant", "content": str(answer)})

# ----------------- Render conversation -----------------
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------- Quick starters -----------------
st.divider()
st.caption("Quick example questions about Canada’s AI strategy:")
cols = st.columns(3)
examples = [
    "How does Canada compare to other leading countries in overall AI readiness?",
    "What does the Global AI methodology say about how Canada's AI capacity is measured?",
    "In which pillars is Canada strongest or weakest according to the AI scores?",
]

for i, ex in enumerate(examples):
    if cols[i % 3].button(ex):
        st.session_state.history.append({"role": "user", "content": ex})

        with st.spinner("Thinking..."):
            try:
                answer = kickoff_query(ex)
            except Exception as e:
                answer = f"Sorry, something went wrong: `{e}`"

        st.session_state.history.append({"role": "assistant", "content": str(answer)})
        st.rerun()
