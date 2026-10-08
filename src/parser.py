"""
Document parser module for the AI Resume Screening & Ranking System.
Extracts clean plain text from PDF and TXT files using PyMuPDF.
Includes robust fault-tolerance to prevent corrupted files from crashing batches.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import pymupdf

from src.utils import clean_text, setup_logger

logger = setup_logger("parser")

# Suppress internal mupdf font/rendering warnings to keep terminal clean
try:
    pymupdf.TOOLS.mupdf_display_errors(False)
    pymupdf.TOOLS.mupdf_display_warnings(False)
except Exception:
    pass


@dataclass
class ParsedDocument:
    """Represents the raw parsed content of a resume document."""
    file_path: str
    file_name: str
    text: str
    is_success: bool
    error_message: Optional[str] = None


def discover_resumes(path_str: str) -> List[str]:
    """
    Finds all supported resume files (PDF, TXT) in the given directory or single file path.
    Returns a sorted list of absolute file paths.
    """
    target = Path(path_str)
    if not target.exists():
        logger.error(f"Input path does not exist: {path_str}")
        return []

    if target.is_file():
        if target.suffix.lower() in {".pdf", ".txt", ".docx"}:
            return [str(target.resolve())]
        else:
            logger.warning(f"Unsupported file format: {target.suffix}")
            return []

    supported_extensions = {".pdf", ".txt", ".docx"}
    found_files: List[str] = []

    for item in target.iterdir():
        if item.is_file() and item.suffix.lower() in supported_extensions:
            # Skip hidden or temporary files
            if not item.name.startswith((".", "~$")):
                found_files.append(str(item.resolve()))

    found_files.sort()
    logger.info(f"Discovered {len(found_files)} resume file(s) in {path_str}")
    return found_files


def parse_pdf(file_path: str) -> ParsedDocument:
    """Extracts text from a PDF file using PyMuPDF."""
    file_name = Path(file_path).name
    try:
        doc = pymupdf.open(file_path)
        extracted_pages = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text:
                extracted_pages.append(text)
        doc.close()

        combined_text = "\n".join(extracted_pages)
        cleaned = clean_text(combined_text)

        if not cleaned:
            logger.warning(f"Empty or non-extractable text in PDF: {file_name}")
            return ParsedDocument(
                file_path=file_path,
                file_name=file_name,
                text="",
                is_success=False,
                error_message="No extractable text found in PDF (possibly scanned/image-only)",
            )

        return ParsedDocument(
            file_path=file_path,
            file_name=file_name,
            text=cleaned,
            is_success=True,
            error_message=None,
        )
    except Exception as e:
        logger.error(f"Failed to parse PDF {file_name}: {str(e)}")
        return ParsedDocument(
            file_path=file_path,
            file_name=file_name,
            text="",
            is_success=False,
            error_message=f"PDF parsing error: {str(e)}",
        )


def parse_txt(file_path: str) -> ParsedDocument:
    """Extracts text from a plain text file."""
    file_name = Path(file_path).name
    try:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as f:
                content = f.read()

        cleaned = clean_text(content)
        return ParsedDocument(
            file_path=file_path,
            file_name=file_name,
            text=cleaned,
            is_success=bool(cleaned),
            error_message=None if cleaned else "Empty text file",
        )
    except Exception as e:
        logger.error(f"Failed to parse TXT {file_name}: {str(e)}")
        return ParsedDocument(
            file_path=file_path,
            file_name=file_name,
            text="",
            is_success=False,
            error_message=f"TXT read error: {str(e)}",
        )


def parse_resume_file(file_path: str) -> ParsedDocument:
    """
    Entry point for parsing any supported resume file.
    Catches all unexpected errors to ensure the batch never crashes.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    try:
        if suffix == ".pdf":
            return parse_pdf(file_path)
        elif suffix == ".txt":
            return parse_txt(file_path)
        elif suffix == ".docx":
            # Basic fallback for docx if python-docx available or unsupported note
            try:
                import docx  # type: ignore
                doc = docx.Document(file_path)
                full_text = "\n".join([p.text for p in doc.paragraphs if p.text])
                cleaned = clean_text(full_text)
                return ParsedDocument(
                    file_path=file_path,
                    file_name=path.name,
                    text=cleaned,
                    is_success=bool(cleaned),
                    error_message=None if cleaned else "Empty docx file",
                )
            except ImportError:
                return ParsedDocument(
                    file_path=file_path,
                    file_name=path.name,
                    text="",
                    is_success=False,
                    error_message="python-docx is not installed to process .docx files",
                )
        else:
            return ParsedDocument(
                file_path=file_path,
                file_name=path.name,
                text="",
                is_success=False,
                error_message=f"Unsupported file format: {suffix}",
            )
    except Exception as e:
        logger.error(f"Fatal parse error for {path.name}: {str(e)}")
        return ParsedDocument(
            file_path=file_path,
            file_name=path.name,
            text="",
            is_success=False,
            error_message=f"Unexpected parsing failure: {str(e)}",
        )
