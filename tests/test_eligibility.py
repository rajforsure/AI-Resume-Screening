"""
Unit tests for hard eligibility evaluation.
Covers:
- Python + AI candidate -> eligible
- Python-only candidate -> rejected
- AI-only candidate -> rejected
- Java/React-only candidate -> rejected
- Python + RAG candidate -> eligible
"""

import pytest
from src.eligibility import evaluate_hard_eligibility
from src.models import ExtractedProject, ExtractedResumeData


def test_python_and_ai_candidate_eligible():
    """Python + AI candidate must be eligible."""
    data = ExtractedResumeData(
        candidate_name="Alice AI",
        skills=["Python", "LangChain", "FastAPI", "PostgreSQL"],
        projects=[
            ExtractedProject(
                title="Agentic Workflow",
                description="Built an agent with LangChain and tool-calling.",
                technologies=["Python", "LangChain"],
                has_ai_elements=True,
            )
        ],
    )
    raw_text = (
        "Alice AI\nPython Developer with LangChain and multi-agent systems experience.\n"
        "Projects: Built agentic workflow with LangChain and tool-calling."
    )
    res = evaluate_hard_eligibility(data, raw_text)
    assert res.eligible is True
    assert res.has_python_evidence is True
    assert res.has_ai_evidence is True
    assert len(res.rejection_reasons) == 0


def test_python_only_candidate_rejected():
    """Python-only candidate without meaningful AI must be rejected."""
    data = ExtractedResumeData(
        candidate_name="Bob Backend",
        skills=["Python", "Django", "PostgreSQL", "Redis"],
        projects=[
            ExtractedProject(
                title="E-commerce Backend",
                description="Built REST APIs using Django and PostgreSQL.",
                technologies=["Python", "Django"],
                has_ai_elements=False,
            )
        ],
    )
    raw_text = (
        "Bob Backend\nPython Backend Engineer specializing in Django, PostgreSQL, and Redis.\n"
        "Projects: High-throughput order management REST APIs."
    )
    res = evaluate_hard_eligibility(data, raw_text)
    assert res.eligible is False
    assert res.has_python_evidence is True
    assert res.has_ai_evidence is False
    assert any("AI/LLM/RAG/agentic" in reason for reason in res.rejection_reasons)


def test_ai_only_candidate_rejected():
    """AI-only candidate without Python evidence must be rejected."""
    data = ExtractedResumeData(
        candidate_name="Charlie Javascript-AI",
        skills=["JavaScript", "TypeScript", "LangChain", "React"],
        projects=[
            ExtractedProject(
                title="AI Frontend Portal",
                description="Integrated LangChainJS with Next.js and React.",
                technologies=["JavaScript", "LangChainJS"],
                has_ai_elements=True,
            )
        ],
    )
    raw_text = (
        "Charlie Javascript-AI\nFrontend developer integrating LangChain and AI assistants in React.\n"
        "No Python experience, strictly Node.js and TypeScript."
    )
    res = evaluate_hard_eligibility(data, raw_text)
    assert res.eligible is False
    assert res.has_python_evidence is False
    assert res.has_ai_evidence is True
    assert any("Python evidence" in reason for reason in res.rejection_reasons)


def test_java_react_only_candidate_rejected():
    """Java / React only candidate must be rejected for both missing Python and missing AI."""
    data = ExtractedResumeData(
        candidate_name="David Fullstack",
        skills=["Java", "Spring Boot", "React", "MySQL"],
        projects=[
            ExtractedProject(
                title="Banking Portal",
                description="Enterprise banking web portal built with Spring Boot and React.",
                technologies=["Java", "React"],
                has_ai_elements=False,
            )
        ],
    )
    raw_text = (
        "David Fullstack\nJava Enterprise Engineer with Spring Boot and React.\n"
        "No Python or AI knowledge."
    )
    res = evaluate_hard_eligibility(data, raw_text)
    assert res.eligible is False
    assert res.has_python_evidence is False
    assert res.has_ai_evidence is False
    assert len(res.rejection_reasons) == 2


def test_python_and_rag_candidate_eligible():
    """Python + RAG candidate must be eligible."""
    data = ExtractedResumeData(
        candidate_name="Elena RAG",
        skills=["Python", "FastAPI", "Vector Search"],
        projects=[
            ExtractedProject(
                title="Document Q&A RAG System",
                description="Built a RAG pipeline utilizing FAISS vector search and embeddings in Python.",
                technologies=["Python", "FAISS", "RAG"],
                has_ai_elements=True,
            )
        ],
    )
    raw_text = (
        "Elena RAG\nSoftware Engineer with experience in Python and RAG systems.\n"
        "Developed retrieval augmented generation pipeline with FAISS vector database."
    )
    res = evaluate_hard_eligibility(data, raw_text)
    assert res.eligible is True
    assert res.has_python_evidence is True
    assert res.has_ai_evidence is True
    assert len(res.rejection_reasons) == 0


def test_polyglot_developer_with_java_react_and_python_ai_eligible():
    """JavaScript/Java/React presence must NOT cause rejection when Python + AI is present."""
    data = ExtractedResumeData(
        candidate_name="Frank Polyglot",
        skills=["Java", "React", "TypeScript", "Python", "LangGraph", "Docker"],
        projects=[
            ExtractedProject(
                title="Multi-Agent Pipeline",
                description="Implemented multi-agent research assistant using LangGraph in Python.",
                technologies=["Python", "LangGraph"],
                has_ai_elements=True,
            )
        ],
    )
    raw_text = (
        "Frank Polyglot\nFull stack developer proficient in React, Java, and Python.\n"
        "Engineered autonomous AI agents using LangGraph and Python."
    )
    res = evaluate_hard_eligibility(data, raw_text)
    assert res.eligible is True
    assert res.has_python_evidence is True
    assert res.has_ai_evidence is True
