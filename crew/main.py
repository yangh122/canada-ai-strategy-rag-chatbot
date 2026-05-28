# crew/main.py
from crewai import Crew
from crew.tasks import task_gather, task_answer


def kickoff_query(query: str):
    """
    Run the two-agent Canada AI Strategy crew on a single user query.

    - task_gather: uses tools to retrieve context, citations, summary, keywords
    - task_answer: uses that context pack to produce a grounded final answer
    """
    crew = Crew(
        agents=[task_gather.agent, task_answer.agent],
        tasks=[task_gather, task_answer],
        verbose=True,
    )

    # Our tasks only expect "query" now
    return crew.kickoff(inputs={"query": query})


if __name__ == "__main__":
    # Example query focused on Canada's AI strategy / capacity
    q = "How does Canada compare to other leading countries in AI readiness and investment?"
    ans = kickoff_query(q)

    print("\n=== FINAL ANSWER ===\n")
    print(ans)
