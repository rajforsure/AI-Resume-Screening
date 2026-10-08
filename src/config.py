"""
Configuration management for the AI Resume Screening & Ranking System.
Loads environment variables from .env if present and sets runtime defaults.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional
from dotenv import load_dotenv

# Load .env file automatically if present
load_dotenv()


@dataclass(frozen=True)
class ScoringWeights:
    """Baseline weights matching the 100-point rubric."""
    ai_project_depth: float = 40.0
    python_backend: float = 30.0
    cloud_fullstack: float = 15.0
    github: float = 10.0
    engineering_depth: float = 5.0

    @property
    def total(self) -> float:
        return (
            self.ai_project_depth
            + self.python_backend
            + self.cloud_fullstack
            + self.github
            + self.engineering_depth
        )


@dataclass
class AppConfig:
    """Application-wide configuration."""
    # LLM Settings
    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "auto").lower()
    )
    groq_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY")
    )
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    )
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    openai_base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )
    gemini_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY")
    )
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    )

    # GitHub Settings
    github_token: Optional[str] = field(
        default_factory=lambda: os.getenv("GITHUB_TOKEN")
    )
    github_timeout: int = 10
    github_max_repos: int = 10

    # Scoring Weights
    weights: ScoringWeights = field(default_factory=ScoringWeights)

    # General
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper()
    )


# Singleton instance
config = AppConfig()
