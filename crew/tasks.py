# crew/tasks.py
from __future__ import annotations

from crewai import Task

# ✅ Use the updated agent names
from crew.agents import canada_ai_researcher, domain_expert


SYSTEM_RULES = """
You are part of a multi-agent system analyzing Canada's AI strategy and AI competitiveness.
You must:
- Use only information retrieved from the Canada AI vector store (PDF methodology + CSV AI scores).
- Avoid hallucinating facts that are not present in the retrieved context.
- Clearly distinguish between evidence (what the documents say) and interpretation.
- Be concise, factual, and policy-aware.
"""


# ---------- Task 1: Retrieval / context pack ----------
task_gather = Task(
    description=(
        "Given the user's query (in {query}):\n"
        "1) Use the tool `retrieve_context` to gather the most relevant text chunks from the "
        "Canada AI vector store (PDF methodology + CSV AI score data).\n"
        "2) Optionally, lightly organize or trim the retrieved text so it is easier for the "
        "next agent to read (but do not invent new information).\n\n"
        "Return a JSON-like block with at least one key:\n"
        "- If the user mentions two or more countries, explicitly compare them using a short bullet list "
        "of their key scores/ranks before giving your narrative explanation."
        "- 'context': a single string containing the key retrieved passages that best address the query."
    ),
    expected_output="A compact context pack with a 'context' field containing the most relevant retrieved passages.",
    agent=canada_ai_researcher,
)


task_answer = Task(
    description=(
        f"{SYSTEM_RULES}\n\n"
        "Using the retrieved context from the researcher, write a clear, grounded answer "
        "to the user’s question about Canada’s AI strategy, AI readiness, methodology, or scores.\n\n"
        "Guidelines:\n"
        "- Use ONLY information from the retrieved context (methodology PDF + score CSV).\n"
        "- Refer to methodology sections or numeric scores when relevant.\n"
        "- Explain what the retrieved evidence means for Canada in plain English.\n"
        "- If the context does not contain enough information, acknowledge the gap.\n\n"
        "- If important numerical or structured facts appear in the context, present them as dot points.\n"
        "- If the user mentions two or more countries, explicitly compare them using a short bullet list "
        "of their key scores/ranks before giving your narrative explanation."
        "Produce a natural, well-written paragraph-style answer.\n"
        "End with a single bold sentence starting with: 'Summary:' summarizing the overall takeaway."
    ),
    expected_output=(
        "A natural-language, grounded answer followed by a final bold 'Summary:' sentence."
    ),
    agent=domain_expert,
    context=[task_gather],
)

