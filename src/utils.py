"""
General utility functions for text cleaning, logging, and regex parsing.
"""

import logging
import re
import sys
from typing import Optional


def setup_logger(name: str = "ai_screening", level: str = "INFO") -> logging.Logger:
    """Configures and returns a structured logger."""
    logger = logging.getLogger(name)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def clean_text(text: str) -> str:
    """
    Cleans raw extracted text:
    - Normalizes unicode whitespace and special characters
    - Removes non-printable control characters
    - Collapses consecutive empty lines
    """
    if not text:
        return ""

    # Replace non-breaking spaces and unusual whitespace
    text = text.replace("\xa0", " ").replace("\u200b", "")

    # Replace common bullet points and symbols with standard text
    text = re.sub(r"[\u2022\u2023\u25E6\u2043\u2219\uf0b7\x83\x95]", " - ", text)

    # Normalize carriage returns and newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Clean multiple spaces within lines
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]

    # Remove runs of empty lines
    cleaned_lines = []
    prev_empty = False
    for line in lines:
        if not line:
            if not prev_empty:
                cleaned_lines.append("")
                prev_empty = True
        else:
            cleaned_lines.append(line)
            prev_empty = False

    return "\n".join(cleaned_lines).strip()


def extract_github_username(url_or_text: Optional[str]) -> Optional[str]:
    """
    Extracts a clean GitHub username from a URL or text string.
    Handles formats like:
    - https://github.com/username
    - http://github.com/username/
    - github.com/username
    - github: username
    Filters out invalid keywords (e.g. 'none', 'null', 'git', 'features').
    """
    if not url_or_text:
        return None

    # Pattern for github.com/username
    match = re.search(r"github\.com/([a-zA-Z0-9_\-]+)", url_or_text, re.IGNORECASE)
    if match:
        user = match.group(1).strip()
        user_lower = user.lower()
        if user_lower not in {"none", "null", "git", "github", "features", "topics", "pulls", "issues"}:
            return user

    # Pattern for 'GitHub: username'
    match_handle = re.search(r"github\s*[:\-]\s*([a-zA-Z0-9_\-]+)", url_or_text, re.IGNORECASE)
    if match_handle:
        user = match_handle.group(1).strip()
        user_lower = user.lower()
        if user_lower not in {"none", "null", "git", "github", "profile", "link"}:
            return user

    return None
