"""
GitHub enrichment module.
Fetches public engineering signals from the GitHub REST API.
Includes caching, rate-limit awareness, and resilient fallbacks so API errors never crash the batch.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional
import requests

from src.config import config
from src.models import GitHubEnrichment
from src.utils import extract_github_username, setup_logger

logger = setup_logger("github")

# In-memory cache to prevent duplicate network calls during batch processing
_GITHUB_CACHE: Dict[str, GitHubEnrichment] = {}
_GITHUB_RATE_LIMIT_WARNED = False


def get_headers() -> Dict[str, str]:
    """Generates standard GitHub API request headers."""
    headers = {
        "User-Agent": "AI-Resume-Screening-System/1.0",
        "Accept": "application/vnd.github.v3+json",
    }
    if config.github_token:
        headers["Authorization"] = f"Bearer {config.github_token}"
    return headers


def parse_iso_datetime(dt_str: Optional[str]) -> Optional[datetime]:
    """Safely parses ISO-8601 timestamps from GitHub API."""
    if not dt_str:
        return None
    try:
        # Replace trailing 'Z' with UTC offset
        cleaned = dt_str.replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned)
    except Exception:
        return None


def calculate_recent_activity_score(latest_date: Optional[datetime]) -> float:
    """
    Estimates recent activity score (0 to 5 points) based on recency of pushes/updates.
    - Active in last 60 days: 5.0
    - Active in last 180 days: 4.0
    - Active in last 365 days: 3.0
    - Active in last 2 years: 1.5
    - Older or inactive: 0.5
    """
    if not latest_date:
        return 0.0

    now = datetime.now(timezone.utc)
    days_ago = (now - latest_date).days

    if days_ago <= 60:
        return 5.0
    elif days_ago <= 180:
        return 4.0
    elif days_ago <= 365:
        return 3.0
    elif days_ago <= 730:
        return 1.5
    else:
        return 0.5


def calculate_repo_relevance_score(repos: List[dict]) -> tuple[float, int]:
    """
    Evaluates maintained public repositories (0 to 5 points).
    Rewards repositories featuring Python, AI/LLM, backend, or active development.
    Returns (score, relevant_repo_count).
    """
    if not repos:
        return 0.0, 0

    ai_python_keywords = {
        "python", "langchain", "langgraph", "rag", "agent", "llm",
        "fastapi", "flask", "django", "machine-learning", "deep-learning",
        "nlp", "pytorch", "tensorflow", "vector", "ai", "openai",
    }

    relevant_count = 0
    for repo in repos:
        lang = (repo.get("language") or "").lower()
        name = (repo.get("name") or "").lower()
        desc = (repo.get("description") or "").lower()
        topics = [t.lower() for t in repo.get("topics", [])]

        is_python = "python" in lang or "jupyter" in lang
        has_keywords = any(
            kw in name or kw in desc or kw in topics
            for kw in ai_python_keywords
        )

        if is_python or has_keywords:
            relevant_count += 1

    if relevant_count >= 3:
        score = 5.0
    elif relevant_count == 2:
        score = 4.0
    elif relevant_count == 1:
        score = 3.0
    elif len(repos) > 0:
        # Active repositories exist, though not strictly Python/AI focused
        score = 1.5
    else:
        score = 0.0

    return score, relevant_count


def enrich_github_profile(github_url_or_handle: Optional[str]) -> GitHubEnrichment:
    """
    Enriches a candidate's profile via GitHub public API.
    Capped strictly at 10 points (0-5 recent activity + 0-5 maintained/relevant repos).
    Gracefully handles rate limits, missing profiles, and connection issues.
    """
    username = extract_github_username(github_url_or_handle)
    if not username:
        return GitHubEnrichment(
            username=None,
            profile_found=False,
            public_repos=0,
            recent_activity_score=0.0,
            maintained_repos_score=0.0,
            total_score=0.0,
            summary="No GitHub profile provided",
            error=None,
        )

    # Check memory cache
    cache_key = username.lower()
    if cache_key in _GITHUB_CACHE:
        logger.debug(f"Retrieved cached GitHub enrichment for @{username}")
        return _GITHUB_CACHE[cache_key]

    headers = get_headers()
    timeout = config.github_timeout

    try:
        # Fetch user profile
        user_url = f"https://api.github.com/users/{username}"
        user_res = requests.get(user_url, headers=headers, timeout=timeout)

        # Check for rate limit
        if user_res.status_code in (403, 429):
            err_msg = "GitHub API rate limit exceeded (resumed without penalty)"
            global _GITHUB_RATE_LIMIT_WARNED
            if not _GITHUB_RATE_LIMIT_WARNED:
                logger.warning(
                    "\n  [Notice] GitHub API rate limit reached. "
                    "Resuming remaining batch without external GitHub profile calls."
                )
                _GITHUB_RATE_LIMIT_WARNED = True
            enrichment = GitHubEnrichment(
                username=username,
                profile_found=False,
                summary=f"GitHub enrichment skipped (@{username}): {err_msg}",
                error=err_msg,
            )
            _GITHUB_CACHE[cache_key] = enrichment
            return enrichment

        if user_res.status_code == 404:
            enrichment = GitHubEnrichment(
                username=username,
                profile_found=False,
                summary=f"GitHub profile @{username} not found (404)",
                error="User not found",
            )
            _GITHUB_CACHE[cache_key] = enrichment
            return enrichment

        if user_res.status_code != 200:
            enrichment = GitHubEnrichment(
                username=username,
                profile_found=False,
                summary=f"GitHub API returned status {user_res.status_code}",
                error=f"HTTP {user_res.status_code}",
            )
            _GITHUB_CACHE[cache_key] = enrichment
            return enrichment

        user_data = user_res.json()
        public_repos = user_data.get("public_repos", 0)

        # Fetch recent repositories
        repos_url = f"https://api.github.com/users/{username}/repos?sort=updated&per_page={config.github_max_repos}"
        repos_res = requests.get(repos_url, headers=headers, timeout=timeout)
        repos_data = repos_res.json() if repos_res.status_code == 200 and isinstance(repos_res.json(), list) else []

        # Find latest activity timestamp across user profile and repos
        latest_date = parse_iso_datetime(user_data.get("updated_at"))
        for repo in repos_data:
            pushed_at = parse_iso_datetime(repo.get("pushed_at"))
            updated_at = parse_iso_datetime(repo.get("updated_at"))
            for candidate_dt in (pushed_at, updated_at):
                if candidate_dt and (latest_date is None or candidate_dt > latest_date):
                    latest_date = candidate_dt

        # Compute scoring signals
        act_score = calculate_recent_activity_score(latest_date)
        repo_score, relevant_count = calculate_repo_relevance_score(repos_data)
        total_score = min(10.0, round(act_score + repo_score, 1))

        date_desc = latest_date.strftime("%b %Y") if latest_date else "unknown"
        summary = (
            f"Active profile @{username} with {public_repos} public repos. "
            f"Latest activity: {date_desc} ({act_score}/5 pts). "
            f"Identified {relevant_count} relevant/Python/AI repos ({repo_score}/5 pts)."
        )

        enrichment = GitHubEnrichment(
            username=username,
            profile_found=True,
            public_repos=public_repos,
            recent_activity_score=act_score,
            maintained_repos_score=repo_score,
            total_score=total_score,
            summary=summary,
            error=None,
        )
        _GITHUB_CACHE[cache_key] = enrichment
        return enrichment

    except requests.exceptions.RequestException as e:
        logger.warning(f"Network error querying GitHub for @{username}: {str(e)}")
        enrichment = GitHubEnrichment(
            username=username,
            profile_found=False,
            summary=f"GitHub connection failed for @{username}: {str(e)}",
            error=str(e),
        )
        _GITHUB_CACHE[cache_key] = enrichment
        return enrichment
    except Exception as e:
        logger.error(f"Unexpected error in GitHub enrichment for @{username}: {str(e)}")
        enrichment = GitHubEnrichment(
            username=username,
            profile_found=False,
            summary=f"Error evaluating GitHub @{username}: {str(e)}",
            error=str(e),
        )
        _GITHUB_CACHE[cache_key] = enrichment
        return enrichment
