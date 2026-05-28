from crewai import Agent
from crew.llm import chatgpt_llm
from crew.tools import (
    retrieve_context
)

# ----------------------------------------
# Agent 1: Canada AI Strategy Researcher
# ----------------------------------------
canada_ai_researcher = Agent(
    role="Canada AI Strategy Researcher",
    goal=(
        "Retrieve and curate the most relevant, high-signal snippets from the "
        "Canada AI vector store, which contains the Global AI methodology PDF "
        "and the AI score CSV dataset."
    ),
    backstory=(
        "You specialize in analyzing the Global AI Index methodology and Canada's "
        "AI competitiveness indicators. You locate, extract, and package grounded "
        "evidence that helps explain Canada’s AI readiness, strengths, weaknesses, "
        "and overall posture in the AI landscape."
    ),
    llm=chatgpt_llm,
    tools=[retrieve_context],
    verbose=True,
)

# ----------------------------------------
# Agent 2: AI Strategy & Competitiveness Expert
# ----------------------------------------
domain_expert = Agent(
    role="AI Strategy & Competitiveness Expert",
    goal=(
        "Write concise, evidence-grounded explanations about Canada's AI strategy, "
        "AI readiness, national capabilities, and comparative strengths — strictly "
        "using the retrieved context and citations."
    ),
    backstory=(
        "You are an expert in AI policy, AI readiness evaluation, national strategy, "
        "and competitiveness. You interpret retrieved evidence from the Global AI "
        "methodology and the AI score dataset to produce clear, meaningful insights "
        "without hallucination."
    ),
    llm=chatgpt_llm,
    verbose=True,
)
