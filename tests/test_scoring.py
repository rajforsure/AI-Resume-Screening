"""
Unit tests for candidate scoring, rubric evaluation, and deterministic tie-breaking.
Covers:
- Shallow LLM wrapper receives reduced AI score (5-15 point penalty)
- Missing GitHub profile leaves candidate eligible and scorable
- Deterministic tie-breaking behavior
- Maximum rubric caps
"""

import pytest
from src.models import (
    EligibleCandidate,
    ExtractedProject,
    ExtractedResumeData,
    GitHubEnrichment,
    ScoreBreakdown,
)
from src.scorer import (
    evaluate_eligible_candidate,
    rank_eligible_candidates,
    score_ai_project_depth,
)


def test_shallow_llm_wrapper_penalty():
    """A thin wrapper around OpenAI API must receive a reduced AI score compared to deep agentic work."""
    # Deep Candidate
    deep_data = ExtractedResumeData(
        candidate_name="Deep Candidate",
        skills=["Python", "LangGraph", "ChromaDB", "Ragas"],
        projects=[
            ExtractedProject(
                title="Stateful Multi-Agent System",
                description="LangGraph multi-agent orchestration with ChromaDB RAG, tool calling, and Ragas eval.",
                technologies=["Python", "LangGraph", "ChromaDB"],
                has_ai_elements=True,
            )
        ],
    )
    deep_raw = (
        "Built a stateful multi-agent system with LangGraph and ChromaDB vector search. "
        "Implemented tool-calling agents and evaluated with Ragas."
    )
    deep_score, _, deep_concerns = score_ai_project_depth(deep_data, deep_raw)

    # Shallow Wrapper Candidate
    shallow_data = ExtractedResumeData(
        candidate_name="Shallow Candidate",
        skills=["Python", "OpenAI"],
        projects=[
            ExtractedProject(
                title="Simple Chatbot",
                description="Built a simple chatbot using a wrapper around OpenAI API and basic prompt in Streamlit.",
                technologies=["Python", "OpenAI API"],
                has_ai_elements=True,
                is_shallow_wrapper=True,
            )
        ],
    )
    shallow_raw = "Built a simple chatbot wrapper around OpenAI with streamlit chatbot."
    shallow_score, _, shallow_concerns = score_ai_project_depth(shallow_data, shallow_raw)

    # Verify shallow receives significantly lower score (penalty applied)
    assert shallow_score < deep_score
    assert (deep_score - shallow_score) >= 10.0
    assert any("shallow" in c.lower() or "penalty" in c.lower() for c in shallow_concerns)


def test_missing_github_candidate_remains_scorable():
    """Missing or private GitHub must NOT make candidate ineligible and candidate can still achieve high score."""
    data = ExtractedResumeData(
        candidate_name="Grace No-GitHub",
        skills=["Python", "FastAPI", "PostgreSQL", "Redis", "LangChain", "LangGraph", "Docker", "AWS", "Pytest"],
        projects=[
            ExtractedProject(
                title="AI Knowledge Assistant",
                description="Stateful multi-agent RAG pipeline built with LangGraph, FAISS vector search, and FastAPI.",
                technologies=["Python", "LangGraph", "FAISS", "FastAPI"],
                has_ai_elements=True,
            )
        ],
    )
    raw = (
        "Grace No-GitHub\nPython engineer building stateful multi-agent RAG systems with LangGraph, FAISS, and FastAPI. "
        "PostgreSQL and Redis database integration, Docker, AWS cloud deployment, Pytest unit testing."
    )
    no_gh_enrichment = GitHubEnrichment(
        username=None,
        profile_found=False,
        public_repos=0,
        recent_activity_score=0.0,
        maintained_repos_score=0.0,
        total_score=0.0,
        summary="No GitHub profile provided",
    )

    total_score, breakdown, proj_sum, gh_sum, strengths, concerns = evaluate_eligible_candidate(
        data=data,
        raw_text=raw,
        github_enrichment=no_gh_enrichment,
    )

    # Candidate should still receive high score (> 50.0) across other dimensions
    assert total_score >= 50.0
    assert breakdown.github == 0.0
    assert breakdown.ai_project_depth >= 25.0
    assert breakdown.python_backend >= 15.0
    assert "No GitHub profile provided" in gh_sum


def test_score_caps_respected():
    """Scores must be strictly clamped to their rubric maxima."""
    data = ExtractedResumeData(
        candidate_name="Super Candidate",
        skills=["Python", "FastAPI", "Django", "Flask", "LangChain", "LangGraph", "Docker", "AWS", "GCP", "Pytest"],
    )
    super_raw = (
        "Super candidate with LangGraph, LangChain, RAG, vector search, FAISS, tool calling, "
        "FastAPI, Django, Flask, PostgreSQL, MongoDB, Redis, Docker, Kubernetes, AWS, GCP, Pytest."
    )
    gh_max = GitHubEnrichment(
        username="superdev",
        profile_found=True,
        public_repos=50,
        recent_activity_score=5.0,
        maintained_repos_score=5.0,
        total_score=10.0,
        summary="Very active",
    )

    total_score, breakdown, _, _, _, _ = evaluate_eligible_candidate(data, super_raw, gh_max)

    assert breakdown.ai_project_depth <= 40.0
    assert breakdown.python_backend <= 30.0
    assert breakdown.cloud_fullstack <= 15.0
    assert breakdown.github <= 10.0
    assert breakdown.engineering_depth <= 5.0
    assert total_score <= 100.0


def test_deterministic_tie_breaker():
    """
    Tie breaker order:
    1. total_score descending
    2. ai_project_depth descending
    3. python_backend descending
    4. candidate_name alphabetically
    """
    c1 = EligibleCandidate(
        rank=0,
        candidate_name="Zack",
        eligible=True,
        total_score=80.0,
        score_breakdown=ScoreBreakdown(
            ai_project_depth=35.0,
            python_backend=25.0,
            cloud_fullstack=10.0,
            github=5.0,
            engineering_depth=5.0,
        ),
        matched_skills=[],
        project_summary="",
        github_summary="",
        strengths=[],
        concerns=[],
    )

    c2 = EligibleCandidate(
        rank=0,
        candidate_name="Adam",
        eligible=True,
        total_score=80.0,
        score_breakdown=ScoreBreakdown(
            ai_project_depth=35.0,
            python_backend=25.0,
            cloud_fullstack=10.0,
            github=5.0,
            engineering_depth=5.0,
        ),
        matched_skills=[],
        project_summary="",
        github_summary="",
        strengths=[],
        concerns=[],
    )

    c3 = EligibleCandidate(
        rank=0,
        candidate_name="Brian",
        eligible=True,
        total_score=80.0,
        score_breakdown=ScoreBreakdown(
            ai_project_depth=38.0,  # Higher AI score
            python_backend=22.0,
            cloud_fullstack=10.0,
            github=5.0,
            engineering_depth=5.0,
        ),
        matched_skills=[],
        project_summary="",
        github_summary="",
        strengths=[],
        concerns=[],
    )

    c4 = EligibleCandidate(
        rank=0,
        candidate_name="Higher Total",
        eligible=True,
        total_score=85.0,  # Higher total score
        score_breakdown=ScoreBreakdown(
            ai_project_depth=30.0,
            python_backend=25.0,
            cloud_fullstack=15.0,
            github=10.0,
            engineering_depth=5.0,
        ),
        matched_skills=[],
        project_summary="",
        github_summary="",
        strengths=[],
        concerns=[],
    )

    ranked = rank_eligible_candidates([c1, c2, c3, c4])

    assert [c.candidate_name for c in ranked] == ["Higher Total", "Brian", "Adam", "Zack"]
    assert ranked[0].rank == 1
    assert ranked[1].rank == 2
    assert ranked[2].rank == 3
    assert ranked[3].rank == 4
