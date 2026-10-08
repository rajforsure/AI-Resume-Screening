"""
Candidate scoring and ranking engine.
Implements the 100-point baseline evaluation rubric:
- AI / Agentic / RAG Project Depth (40 pts) [with shallow wrapper penalty of 5-15 pts]
- Python & Backend Engineering (30 pts)
- Cloud / Deployment / Full Stack (15 pts)
- GitHub Activity (10 pts)
- Engineering Depth Signals (5 pts)

Includes deterministic multi-level tie-breaking for fair, explainable ranking.
"""

from typing import Any, Dict, List, Optional, Tuple
from src.config import config
from src.models import (
    EligibleCandidate,
    ExtractedResumeData,
    GitHubEnrichment,
    ScoreBreakdown,
)
from src.utils import setup_logger

logger = setup_logger("scorer")


def score_ai_project_depth(
    data: ExtractedResumeData,
    raw_text: str,
    llm_assessment: Optional[Dict[str, Any]] = None,
) -> Tuple[float, List[str], List[str]]:
    """
    Evaluates AI / Agentic / RAG Project Depth (0 to 40 points).
    Applies strict penalties (5 to 15 points) for shallow wrappers around LLM APIs.
    Returns (score, strengths, concerns).
    """
    strengths: List[str] = []
    concerns: List[str] = []

    # If an LLM assessment exists and is valid, incorporate it
    if llm_assessment and "ai_depth_score" in llm_assessment:
        score = float(llm_assessment.get("ai_depth_score", 0.0))
        penalty = float(llm_assessment.get("penalty_applied", 0.0))
        if penalty > 0:
            concerns.append(f"LLM quality audit flagged shallow wrapper (-{penalty:.1f} pts penalty applied)")
        for st in llm_assessment.get("strengths", []):
            strengths.append(st)
        for cn in llm_assessment.get("concerns", []):
            concerns.append(cn)
        return min(40.0, max(0.0, score)), strengths, concerns

    # Rule-based evaluation
    raw_lower = raw_text.lower()
    score = 0.0

    # 1. Frameworks & Architecture Depth
    has_stateful_agent = any(
        kw in raw_lower for kw in ["langgraph", "crewai", "autogen", "google adk", "multi-agent", "multi agent"]
    )
    if has_stateful_agent:
        score += 15.0
        strengths.append("Stateful multi-agent architecture or orchestration framework identified")

    has_rag_vectors = any(
        kw in raw_lower
        for kw in ["rag", "retrieval augmented", "vector search", "pinecone", "chroma", "faiss", "weaviate", "qdrant"]
    )
    if has_rag_vectors:
        score += 14.0
        strengths.append("Retrieval-Augmented Generation (RAG) and vector database integration demonstrated")

    has_tools_evaluation = any(
        kw in raw_lower for kw in ["tool calling", "function calling", "ragas", "trulens", "eval pipeline", "evaluation"]
    )
    if has_tools_evaluation:
        score += 8.0
        strengths.append("Tool-calling agents or evaluation pipelines (Ragas/TruLens) utilized")

    has_embeddings_custom = any(
        kw in raw_lower for kw in ["embeddings", "embedding model", "fine-tuning", "peft", "lora", "ollama", "vllm"]
    )
    if has_embeddings_custom:
        score += 7.0
        strengths.append("Hands-on custom embeddings, local LLM deployment, or fine-tuning")

    # If general LLM mentions but none of the deep signals above
    if score == 0.0:
        if any(kw in raw_lower for kw in ["langchain", "llamaindex", "llm", "large language model", "gpt"]):
            score += 12.0
            strengths.append("Foundational LLM integration and prompting evidence")

    # 2. Check for Shallow Wrappers or Tutorial Indicators
    shallow_signals = [
        "simple chatbot", "basic chatbot", "chatgpt clone", "wrapper around openai",
        "streamlit chatbot", "simple prompt", "api call to chatgpt"
    ]
    is_shallow = any(sig in raw_lower for sig in shallow_signals)

    # Check if AI was merely listed in skills with zero implementation in projects
    ai_in_projects = any(
        p.has_ai_elements or any(
            kw in (p.title + " " + p.description).lower()
            for kw in ["ai", "llm", "rag", "langchain", "agent", "vector", "gpt", "model"]
        )
        for p in data.projects
    )

    if not ai_in_projects and len(data.projects) > 0:
        # Candidate only listed keywords without actual project demonstration
        score = min(score, 12.0)
        penalty = 12.0
        score = max(0.0, score - penalty)
        concerns.append(
            f"AI keywords present in skills but missing project implementation (-{penalty:.0f} pts penalty)"
        )
    elif is_shallow:
        # Thin wrapper around OpenAI API without workflow, retrieval, or tool use
        penalty = 10.0
        score = max(0.0, score - penalty)
        concerns.append(
            f"Project appears to be a thin LLM wrapper lacking state or workflow (-{penalty:.0f} pts penalty)"
        )

    # Tutorial-style project indicators (e.g. Titanic, Iris, or generic toy repos)
    if any(tut in raw_lower for tut in ["titanic dataset", "iris dataset", "mnist tutorial"]):
        score = max(0.0, score - 5.0)
        concerns.append("Toy or tutorial-style dataset referenced (-5 pts penalty)")

    final_score = min(40.0, max(0.0, round(score, 1)))
    return final_score, strengths, concerns


def score_python_backend(
    data: ExtractedResumeData,
    raw_text: str,
) -> Tuple[float, List[str], List[str]]:
    """
    Evaluates Python & Backend Engineering (0 to 30 points).
    Covers Python proficiency, web frameworks (FastAPI, Flask, Django),
    database architectures, and async/backend patterns.
    """
    strengths: List[str] = []
    concerns: List[str] = []
    score = 0.0
    raw_lower = raw_text.lower()

    # Core Python & Frameworks (up to 14 pts)
    if "fastapi" in raw_lower:
        score += 10.0
        strengths.append("Modern asynchronous API design with FastAPI")
    elif "django" in raw_lower:
        score += 9.0
        strengths.append("Enterprise backend engineering with Django")
    elif "flask" in raw_lower:
        score += 8.0
        strengths.append("Lightweight REST service development with Flask")
    elif "python" in raw_lower:
        score += 5.0
        strengths.append("Demonstrated Python scripting and foundational programming")

    # Database & Storage systems (up to 8 pts)
    db_matches = []
    if any(db in raw_lower for db in ["postgresql", "postgres", "mysql", "sqlite", "sql"]):
        db_matches.append("Relational SQL")
        score += 4.0
    if any(db in raw_lower for db in ["mongodb", "redis", "dynamodb"]):
        db_matches.append("NoSQL / Key-Value")
        score += 4.0

    if db_matches:
        strengths.append(f"Database engineering ({', '.join(db_matches)})")
    else:
        concerns.append("Limited database or persistent storage engineering evidence")

    # Architecture & Backend engineering patterns (up to 8 pts)
    if any(pat in raw_lower for pat in ["rest api", "restful", "rest apis", "api development", "endpoints"]):
        score += 3.5
    if any(pat in raw_lower for pat in ["celery", "redis", "rabbitmq", "kafka", "asyncio", "background tasks"]):
        score += 4.5
        strengths.append("Distributed queue or asynchronous concurrency handling (Redis/Celery/AsyncIO)")

    final_score = min(30.0, max(0.0, round(score, 1)))
    return final_score, strengths, concerns


def score_cloud_fullstack(
    data: ExtractedResumeData,
    raw_text: str,
) -> Tuple[float, List[str], List[str]]:
    """
    Evaluates Cloud / Deployment / Full Stack (0 to 15 points).
    Covers containerization (Docker, K8s), cloud providers (AWS, GCP, Azure),
    and modern frontend integration (React, Next.js).
    """
    strengths: List[str] = []
    concerns: List[str] = []
    score = 0.0
    raw_lower = raw_text.lower()

    # Containers & Orchestration (up to 6 pts)
    if "docker" in raw_lower:
        score += 4.0
        strengths.append("Containerization experience with Docker")
    if "kubernetes" in raw_lower or "k8s" in raw_lower:
        score += 2.0
        strengths.append("Container orchestration with Kubernetes")

    # Cloud Providers & CI/CD (up to 5 pts)
    cloud_found = []
    for c in ["aws", "gcp", "google cloud", "azure"]:
        if c in raw_lower:
            cloud_found.append(c.upper())
    if cloud_found:
        score += 3.5
        strengths.append(f"Cloud infrastructure deployment ({', '.join(cloud_found)})")

    if any(ci in raw_lower for ci in ["ci/cd", "github actions", "gitlab ci"]):
        score += 2.0
        strengths.append("Automated CI/CD pipelines")

    # Fullstack / Frontend integration (up to 4 pts)
    if any(fe in raw_lower for fe in ["react", "next.js", "nextjs", "vue", "typescript"]):
        score += 3.5
        strengths.append("Full-stack capability with modern frontend frameworks")

    final_score = min(15.0, max(0.0, round(score, 1)))
    return final_score, strengths, concerns


def score_engineering_depth(
    data: ExtractedResumeData,
    raw_text: str,
) -> Tuple[float, List[str], List[str]]:
    """
    Evaluates Engineering Depth Signals (0 to 5 points).
    Rewards automated testing, system design, monitoring, and code quality practices.
    """
    strengths: List[str] = []
    concerns: List[str] = []
    score = 0.0
    raw_lower = raw_text.lower()

    if any(t in raw_lower for t in ["pytest", "unittest", "unit testing", "test-driven", "tdd"]):
        score += 2.5
        strengths.append("Demonstrated automated testing practices (Pytest/Unittest)")
    else:
        concerns.append("No explicit automated testing or Pytest evidence mentioned")

    if any(sd in raw_lower for sd in ["system design", "microservices", "clean architecture", "design patterns", "oop"]):
        score += 1.5
        strengths.append("Solid grasp of system design and object-oriented principles")

    if any(m in raw_lower for m in ["logging", "monitoring", "grafana", "prometheus", "sentry"]):
        score += 1.0
        strengths.append("Observability and production logging awareness")

    final_score = min(5.0, max(0.0, round(score, 1)))
    return final_score, strengths, concerns


def evaluate_eligible_candidate(
    data: ExtractedResumeData,
    raw_text: str,
    github_enrichment: GitHubEnrichment,
    llm_assessment: Optional[Dict[str, Any]] = None,
) -> Tuple[float, ScoreBreakdown, str, str, List[str], List[str]]:
    """
    Scores an eligible candidate across all five rubric dimensions.
    Returns:
    (total_score, breakdown, project_summary, github_summary, strengths, concerns)
    """
    ai_score, ai_str, ai_con = score_ai_project_depth(data, raw_text, llm_assessment)
    py_score, py_str, py_con = score_python_backend(data, raw_text)
    cloud_score, cloud_str, cloud_con = score_cloud_fullstack(data, raw_text)
    eng_score, eng_str, eng_con = score_engineering_depth(data, raw_text)

    # GitHub score capped at 10.0
    gh_score = min(10.0, max(0.0, round(github_enrichment.total_score, 1)))

    total_score = min(100.0, round(ai_score + py_score + cloud_score + gh_score + eng_score, 1))

    breakdown = ScoreBreakdown(
        ai_project_depth=ai_score,
        python_backend=py_score,
        cloud_fullstack=cloud_score,
        github=gh_score,
        engineering_depth=eng_score,
    )

    all_strengths = ai_str + py_str + cloud_str + eng_str
    if github_enrichment.profile_found and gh_score > 0:
        all_strengths.append(f"Public GitHub engineering record ({gh_score}/10 pts)")

    all_concerns = ai_con + py_con + cloud_con + eng_con
    if not github_enrichment.profile_found and github_enrichment.username:
        all_concerns.append(f"GitHub profile not resolved or rate-limited: {github_enrichment.summary}")

    # Build project summary
    if llm_assessment and llm_assessment.get("project_summary"):
        proj_summary = llm_assessment["project_summary"]
    elif data.projects:
        titles = [p.title for p in data.projects[:3] if p.title]
        proj_summary = f"Key projects: {', '.join(titles)}. " + (
            "Includes production-ready AI/backend workflows." if ai_score >= 25 else "Demonstrates functional software development."
        )
    else:
        proj_summary = "Hands-on projects in Python engineering and AI applications."

    gh_summary = github_enrichment.summary

    return total_score, breakdown, proj_summary, gh_summary, all_strengths, all_concerns


def rank_eligible_candidates(candidates: List[EligibleCandidate]) -> List[EligibleCandidate]:
    """
    Sorts eligible candidates deterministically using the mandated multi-level tie breaker:
    1. total_score descending
    2. ai_project_depth descending
    3. python_backend descending
    4. candidate_name alphabetically (case-insensitive)

    Assigns 1-indexed rank to sorted candidates.
    """
    sorted_candidates = sorted(
        candidates,
        key=lambda c: (
            -c.total_score,
            -c.score_breakdown.ai_project_depth,
            -c.score_breakdown.python_backend,
            c.candidate_name.lower(),
        ),
    )

    ranked_list: List[EligibleCandidate] = []
    for rank_idx, cand in enumerate(sorted_candidates, start=1):
        # Update rank
        cand_dict = cand.model_dump()
        cand_dict["rank"] = rank_idx
        ranked_list.append(EligibleCandidate(**cand_dict))

    return ranked_list
