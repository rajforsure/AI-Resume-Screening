"""
Data models and schemas for resume screening and ranking.
Uses Pydantic for validation, type-safety, and serialization.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class ScoreBreakdown(BaseModel):
    """Detailed score breakdown according to the 100-point rubric."""
    ai_project_depth: float = Field(
        ...,
        description="AI / Agentic / RAG Project Depth (0 to 40 pts). Penalized if shallow.",
    )
    python_backend: float = Field(
        ...,
        description="Python & Backend Engineering (0 to 30 pts).",
    )
    cloud_fullstack: float = Field(
        ...,
        description="Cloud / Deployment / Full Stack (0 to 15 pts).",
    )
    github: float = Field(
        ...,
        description="GitHub Activity & relevant repositories (0 to 10 pts).",
    )
    engineering_depth: float = Field(
        ...,
        description="Engineering Depth Signals: testing, CI/CD, architecture (0 to 5 pts).",
    )


class GitHubEnrichment(BaseModel):
    """GitHub profile and repository enrichment data."""
    username: Optional[str] = None
    profile_found: bool = False
    public_repos: int = 0
    recent_activity_score: float = 0.0  # 0 to 5
    maintained_repos_score: float = 0.0  # 0 to 5
    total_score: float = 0.0  # 0 to 10
    summary: str = "No GitHub profile enriched"
    error: Optional[str] = None


class ExtractedProject(BaseModel):
    """Structured project extracted from a resume."""
    title: str = "Project"
    description: str = ""
    technologies: List[str] = Field(default_factory=list)
    has_ai_elements: bool = False
    is_shallow_wrapper: bool = False


class ExtractedResumeData(BaseModel):
    """Semantic data extracted from resume text."""
    candidate_name: str = "Unknown Candidate"
    email: Optional[str] = None
    phone: Optional[str] = None
    github_url: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    projects: List[ExtractedProject] = Field(default_factory=list)
    experience: List[str] = Field(default_factory=list)
    education: List[str] = Field(default_factory=list)
    raw_text_length: int = 0


class EligibilityResult(BaseModel):
    """Outcome of hard eligibility evaluation."""
    eligible: bool
    has_python_evidence: bool
    has_ai_evidence: bool
    rejection_reasons: List[str] = Field(default_factory=list)
    matched_skills: List[str] = Field(default_factory=list)
    evidence_notes: List[str] = Field(default_factory=list)


class EligibleCandidate(BaseModel):
    """Ranked eligible candidate output schema."""
    rank: int = Field(..., description="Deterministic rank among eligible candidates (1-indexed)")
    candidate_name: str
    eligible: bool = True
    total_score: float = Field(..., description="Overall score out of 100")
    score_breakdown: ScoreBreakdown
    matched_skills: List[str]
    project_summary: str
    github_summary: str
    strengths: List[str]
    concerns: List[str]


class RejectedCandidate(BaseModel):
    """Rejected candidate output schema."""
    candidate_name: str
    eligible: bool = False
    rejection_reasons: List[str]
    matched_skills: List[str]


class BatchSummary(BaseModel):
    """Aggregated statistics for a batch screening run."""
    total_resumes: int
    successfully_parsed: int
    eligible: int
    rejected: int
    failed_unreadable: int


class RunOutput(BaseModel):
    """Top-level machine-readable output structure."""
    batch_summary: BatchSummary
    candidates: List[Union[EligibleCandidate, RejectedCandidate]]
