"""
Deterministic hard eligibility filter module.
Evaluates candidate eligibility strictly outside the LLM.
Candidate is eligible only if BOTH Python evidence and meaningful AI/agentic evidence are present.
"""

import re
from typing import List, Tuple
from src.models import EligibilityResult, ExtractedResumeData
from src.utils import setup_logger

logger = setup_logger("eligibility")

# Regex for genuine Python presence
PYTHON_REGEX = re.compile(r"\b(python|python3|py)\b", re.IGNORECASE)

# Patterns representing valid AI, RAG, and agentic engineering evidence
AI_EVIDENCE_PATTERNS = [
    # Frameworks
    (re.compile(r"\blangchain\b", re.IGNORECASE), "LangChain framework"),
    (re.compile(r"\blanggraph\b", re.IGNORECASE), "LangGraph stateful workflows"),
    (re.compile(r"\bllama[-_ ]?index\b", re.IGNORECASE), "LlamaIndex framework"),
    (re.compile(r"\b(google[-_ ]?adk|agent[-_ ]?development[-_ ]?kit)\b", re.IGNORECASE), "Google ADK"),
    (re.compile(r"\bcrew[-_ ]?ai\b", re.IGNORECASE), "CrewAI multi-agent framework"),
    (re.compile(r"\bautogen\b", re.IGNORECASE), "AutoGen agentic framework"),
    # RAG & Retrieval
    (re.compile(r"\b(rag|retrieval[-_ ]augmented[-_ ]generation)\b", re.IGNORECASE), "RAG architecture"),
    (re.compile(r"\b(vector[-_ ]search|semantic[-_ ]search)\b", re.IGNORECASE), "Vector / Semantic Search"),
    (re.compile(r"\b(vector[-_ ]database|vector[-_ ]db|vector[-_ ]store)\b", re.IGNORECASE), "Vector Database"),
    (re.compile(r"\b(pinecone|chroma|chromadb|faiss|weaviate|qdrant|milvus)\b", re.IGNORECASE), "Vector Store (Chroma/FAISS/Pinecone/Qdrant)"),
    (re.compile(r"\b(embeddings?|vector[-_ ]embeddings?)\b", re.IGNORECASE), "Vector Embeddings"),
    # Agentic & Tool Use
    (re.compile(r"\b(agentic|ai[-_ ]agents?|llm[-_ ]agents?)\b", re.IGNORECASE), "Agentic AI / LLM Agents"),
    (re.compile(r"\b(multi[-_ ]agent|agent[-_ ]workflows?)\b", re.IGNORECASE), "Multi-Agent Workflows"),
    (re.compile(r"\b(tool[-_ ]calling|function[-_ ]calling)\b", re.IGNORECASE), "Tool / Function Calling"),
    # Evaluation & Custom LLMs
    (re.compile(r"\b(ragas|trulens|eval(?:uation)?[-_ ]pipelines?)\b", re.IGNORECASE), "RAG / LLM Evaluation Pipeline"),
    (re.compile(r"\b(prompt[-_ ]engineering|prompt[-_ ]chaining|few[-_ ]shot)\b", re.IGNORECASE), "Prompt Engineering / Chaining"),
    (re.compile(r"\b(fine[-_ ]tuning|peft|lora|rlhf)\b", re.IGNORECASE), "Model Fine-Tuning / Adaptation"),
    (re.compile(r"\b(llm|large[-_ ]language[-_ ]models?|gpt-4|gpt-3\.5|gemini|claude|ollama|vllm|hugging[-_ ]?face)\b", re.IGNORECASE), "LLM / Generative AI System"),
]


def check_python_evidence(data: ExtractedResumeData, raw_text: str) -> Tuple[bool, List[str]]:
    """
    Verifies that Python appears as a genuine skill, project technology,
    internship/work technology, or implementation language.
    Filters out explicit negations (e.g. 'No Python experience').
    """
    evidence: List[str] = []

    # Check for explicit negation of Python (e.g. "No Python", "without Python")
    negative_python = re.search(
        r"\b(no|without|zero|not|non[- ])\s*(?:prior|hands-on|experience in)?\s*python\b",
        raw_text,
        re.IGNORECASE,
    )

    # 1. Check extracted skills list
    for skill in data.skills:
        if skill.lower() in ("python", "python3"):
            if not negative_python:
                evidence.append("Python listed in candidate skills")
                break

    # 2. Check extracted projects
    for proj in data.projects:
        proj_text = f"{proj.title} {proj.description} {' '.join(proj.technologies)}"
        if PYTHON_REGEX.search(proj_text):
            if not re.search(r"\b(no|without|zero|not)\s+python\b", proj_text, re.IGNORECASE):
                evidence.append(f"Python utilized in project: '{proj.title}'")
                break

    # 3. Check work experience
    for exp in data.experience:
        if PYTHON_REGEX.search(exp):
            if not re.search(r"\b(no|without|zero|not)\s+python\b", exp, re.IGNORECASE):
                evidence.append("Python utilized in work experience / internship")
                break

    # 4. Check raw text boundary if no negation was found
    if not evidence and not negative_python and PYTHON_REGEX.search(raw_text):
        evidence.append("Python identified in resume implementation details")

    has_python = len(evidence) > 0
    return has_python, evidence


def check_ai_evidence(data: ExtractedResumeData, raw_text: str) -> Tuple[bool, List[str]]:
    """
    Verifies that the candidate possesses at least one meaningful AI/LLM/RAG/agentic
    project, framework, or implementation.
    """
    evidence: List[str] = []

    # Combine searchable texts
    combined_texts = [raw_text]
    for proj in data.projects:
        combined_texts.append(f"{proj.title} {proj.description} {' '.join(proj.technologies)}")
    combined_blob = " ".join(combined_texts)

    for pattern, description in AI_EVIDENCE_PATTERNS:
        if pattern.search(combined_blob):
            evidence.append(description)

    has_ai = len(evidence) > 0
    return has_ai, evidence


def evaluate_hard_eligibility(
    data: ExtractedResumeData,
    raw_text: str,
) -> EligibilityResult:
    """
    Applies deterministic hard filtering before ranking.
    Candidate is eligible only if BOTH Python evidence and meaningful AI evidence exist.
    Other technologies (Java, React, JS) do NOT disqualify the candidate.
    """
    has_python, python_evidence = check_python_evidence(data, raw_text)
    has_ai, ai_evidence = check_ai_evidence(data, raw_text)

    rejection_reasons: List[str] = []
    evidence_notes: List[str] = []

    if not has_python:
        rejection_reasons.append(
            "Missing genuine Python evidence across skills, projects, and work experience"
        )
    else:
        evidence_notes.extend(python_evidence)

    if not has_ai:
        rejection_reasons.append(
            "Missing meaningful AI/LLM/RAG/agentic project, framework, or implementation evidence"
        )
    else:
        evidence_notes.extend(ai_evidence)

    is_eligible = has_python and has_ai

    logger.debug(
        f"Candidate '{data.candidate_name}' eligibility: {is_eligible} "
        f"(Python: {has_python}, AI: {has_ai})"
    )

    return EligibilityResult(
        eligible=is_eligible,
        has_python_evidence=has_python,
        has_ai_evidence=has_ai,
        rejection_reasons=rejection_reasons,
        matched_skills=data.skills,
        evidence_notes=evidence_notes,
    )
