"""
LLM provider adapter module.
Provides an extensible adapter interface for semantic extraction and project-quality assessment.
Supports Gemini (REST), OpenAI / OpenAI-compatible endpoints, and a deterministic Heuristic fallback.
Ensures failed LLM calls never crash the batch.
"""

from abc import ABC, abstractmethod
import json
import re
from typing import Any, Dict, Optional, Tuple
import requests

from src.config import config
from src.utils import setup_logger

logger = setup_logger("llm")


def clean_json_markdown(raw_response: str) -> str:
    """Removes markdown code fences like ```json ... ``` from LLM outputs."""
    raw_response = raw_response.strip()
    if raw_response.startswith("```"):
        # Match ```json or ``` and strip
        lines = raw_response.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw_response = "\n".join(lines).strip()
    return raw_response


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM provider adapters."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if the provider has necessary API credentials configured."""
        pass

    @abstractmethod
    def extract_semantic_data(self, resume_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Extracts structured resume data (candidate_name, email, phone, github_url, skills, projects, etc.).
        Returns (result_dict, error_message).
        """
        pass

    @abstractmethod
    def assess_project_quality(self, extracted_data: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Evaluates project depth, identifying agentic/RAG workflows and penalizing shallow wrappers.
        Returns (assessment_dict, error_message).
        """
        pass


class GeminiProvider(BaseLLMProvider):
    """Adapter for Google Gemini REST API."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or config.gemini_api_key
        self.model = model or config.gemini_model or "gemini-1.5-flash"

    @property
    def provider_name(self) -> str:
        return f"Gemini ({self.model})"

    def is_available(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def _call_gemini(self, system_prompt: str, user_prompt: str) -> Tuple[Optional[str], Optional[str]]:
        """Executes a REST call to the Gemini generateContent API."""
        if not self.is_available():
            return None, "Gemini API key not configured"

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"{system_prompt}\n\nTask:\n{user_prompt}"}],
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=20)
            if resp.status_code != 200:
                err = f"Gemini HTTP {resp.status_code}: {resp.text[:200]}"
                logger.warning(err)
                return None, err

            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return None, "Gemini returned no response candidates"

            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                return None, "Gemini returned empty parts"

            return parts[0].get("text", ""), None
        except Exception as e:
            err = f"Gemini connection error: {str(e)}"
            logger.warning(err)
            return None, err

    def extract_semantic_data(self, resume_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        system_prompt = (
            "You are an expert ATS parser. Extract structured information from the resume text.\n"
            "Respond ONLY with a valid JSON object matching this schema:\n"
            "{\n"
            '  "candidate_name": "string",\n'
            '  "email": "string or null",\n'
            '  "phone": "string or null",\n'
            '  "github_url": "string or null",\n'
            '  "skills": ["string"],\n'
            '  "projects": [{"title": "string", "description": "string", "technologies": ["string"], "has_ai_elements": bool}],\n'
            '  "experience": ["string"],\n'
            '  "education": ["string"]\n'
            "}"
        )
        raw, err = self._call_gemini(system_prompt, resume_text[:6000])
        if err or not raw:
            return None, err

        try:
            parsed = json.loads(clean_json_markdown(raw))
            return parsed, None
        except json.JSONDecodeError as e:
            return None, f"JSON parse error from Gemini: {str(e)}"

    def assess_project_quality(self, extracted_data: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        system_prompt = (
            "You are a Senior AI Engineer evaluating candidate projects for an SDE Intern role.\n"
            "Evaluate the candidate's AI/agentic project depth on a 40-point scale.\n"
            "Penalize shallow projects (thin wrappers around LLM API without workflow/RAG/state/tool use) by 5 to 15 points.\n"
            "Respond ONLY with a valid JSON object matching:\n"
            "{\n"
            '  "ai_depth_score": number (0 to 40),\n'
            '  "is_shallow_wrapper": bool,\n'
            '  "penalty_applied": number (0 to 15),\n'
            '  "project_summary": "concise 2-line summary of projects",\n'
            '  "strengths": ["bullet 1", "bullet 2"],\n'
            '  "concerns": ["bullet 1"]\n'
            "}"
        )
        content = json.dumps(extracted_data, indent=2)
        raw, err = self._call_gemini(system_prompt, content[:4000])
        if err or not raw:
            return None, err

        try:
            parsed = json.loads(clean_json_markdown(raw))
            return parsed, None
        except json.JSONDecodeError as e:
            return None, f"JSON parse error from Gemini project assessment: {str(e)}"


class OpenAIProvider(BaseLLMProvider):
    """Adapter for OpenAI and OpenAI-compatible REST endpoints (Groq, Together, Ollama, etc.)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or config.openai_api_key
        self.base_url = (base_url or config.openai_base_url or "https://api.openai.com/v1").rstrip("/")
        self.model = model or config.openai_model or "gpt-4o-mini"

    @property
    def provider_name(self) -> str:
        return f"OpenAI ({self.model})"

    def is_available(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def _call_openai(self, system_prompt: str, user_prompt: str) -> Tuple[Optional[str], Optional[str]]:
        if not self.is_available():
            return None, "OpenAI API key not configured"

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=20)
            if resp.status_code == 429:
                if not getattr(self, "_rate_limit_warned", False):
                    logger.warning(
                        f"\n  [Notice] {self.provider_name} rate limit reached. "
                        "Seamlessly switching to deterministic heuristic fallback for remaining batch."
                    )
                    self._rate_limit_warned = True
                return None, f"{self.provider_name} HTTP 429: rate limited"

            if resp.status_code != 200:
                err = f"{self.provider_name} HTTP {resp.status_code}"
                logger.warning(f"\n  [Warning] {err}")
                return None, err

            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                return None, f"{self.provider_name} returned no choices"

            content = choices[0].get("message", {}).get("content", "")
            return content, None
        except Exception as e:
            err = f"{self.provider_name} connection error: {str(e)}"
            logger.warning(f"\n  [Warning] {err}")
            return None, err

    def extract_semantic_data(self, resume_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        system_prompt = (
            "You are an ATS parsing assistant. Extract candidate profile into JSON with keys:\n"
            "candidate_name (string), email (string/null), phone (string/null), github_url (string/null), "
            "skills (list of strings), projects (list of objects with title, description, technologies, has_ai_elements), "
            "experience (list of strings), education (list of strings)."
        )
        raw, err = self._call_openai(system_prompt, resume_text[:6000])
        if err or not raw:
            return None, err

        try:
            parsed = json.loads(clean_json_markdown(raw))
            return parsed, None
        except json.JSONDecodeError as e:
            return None, f"JSON parse error from OpenAI: {str(e)}"

    def assess_project_quality(self, extracted_data: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        system_prompt = (
            "You are a Senior AI Engineer evaluating candidate projects for an SDE Intern role.\n"
            "Assess AI/agentic project depth on a 40-point scale. Return JSON with keys:\n"
            "ai_depth_score (float 0-40), is_shallow_wrapper (bool), penalty_applied (float 0-15), "
            "project_summary (string), strengths (list of strings), concerns (list of strings)."
        )
        content = json.dumps(extracted_data, indent=2)
        raw, err = self._call_openai(system_prompt, content[:4000])
        if err or not raw:
            return None, err

        try:
            parsed = json.loads(clean_json_markdown(raw))
            return parsed, None
        except json.JSONDecodeError as e:
            return None, f"JSON parse error from OpenAI project assessment: {str(e)}"


class GroqProvider(OpenAIProvider):
    """Adapter for Groq ultra-fast LLM inference API (OpenAI compatible)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        super().__init__(
            api_key=api_key or config.groq_api_key,
            base_url=base_url or config.groq_base_url,
            model=model or config.groq_model or "qwen/qwen3.8-27b",
        )

    @property
    def provider_name(self) -> str:
        return f"Groq ({self.model})"


class HeuristicFallbackProvider(BaseLLMProvider):
    """
    Deterministic rule-based extractor and project quality evaluator.
    Used when no external LLM API key is configured or when network calls fail.
    Guarantees 100% resilient operation without requiring external cloud credentials.
    """

    @property
    def provider_name(self) -> str:
        return "Deterministic Heuristic Provider"

    def is_available(self) -> bool:
        return True

    def extract_semantic_data(self, resume_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        # The extraction is delegated to src/extractor.py heuristics
        return None, "Heuristic fallback active (delegated to rule-based extractor)"

    def assess_project_quality(self, extracted_data: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        # Handled deterministically in src/scorer.py
        return None, "Heuristic fallback active (delegated to rule-based scorer)"


def get_llm_adapter(preferred_provider: Optional[str] = None) -> BaseLLMProvider:
    """
    Factory function to obtain the appropriate LLM provider adapter.
    Selects Groq, Gemini, or OpenAI if configured; otherwise gracefully falls back to Heuristic provider.
    """
    choice = (preferred_provider or config.llm_provider or "auto").lower()

    if choice in ("groq", "llama"):
        provider = GroqProvider()
        if provider.is_available():
            logger.info(f"Using LLM provider: {provider.provider_name}")
            return provider
        logger.warning("Groq requested but GROQ_API_KEY is not configured.")

    elif choice in ("gemini", "google"):
        provider = GeminiProvider()
        if provider.is_available():
            logger.info(f"Using LLM provider: {provider.provider_name}")
            return provider
        logger.warning("Gemini requested but GEMINI_API_KEY is not configured.")

    elif choice in ("openai", "gpt"):
        provider = OpenAIProvider()
        if provider.is_available():
            logger.info(f"Using LLM provider: {provider.provider_name}")
            return provider
        logger.warning("OpenAI requested but OPENAI_API_KEY is not configured.")

    elif choice == "auto":
        # Check Groq first, then Gemini, then OpenAI
        if config.groq_api_key:
            provider = GroqProvider()
            logger.info(f"Auto-selected LLM provider: {provider.provider_name}")
            return provider
        if config.gemini_api_key:
            provider = GeminiProvider()
            logger.info(f"Auto-selected LLM provider: {provider.provider_name}")
            return provider
        if config.openai_api_key:
            provider = OpenAIProvider()
            logger.info(f"Auto-selected LLM provider: {provider.provider_name}")
            return provider

    logger.info("Using Deterministic Heuristic Provider (no external LLM credentials required).")
    return HeuristicFallbackProvider()
