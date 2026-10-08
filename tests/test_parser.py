"""
Unit tests for document parsing and resilience under malformed inputs.
Covers:
- PDF text extraction
- Plain text file extraction
- Corrupted / malformed resume handling (batch continues without crashing)
- Pipeline resilience
"""

import os
from pathlib import Path
import pytest
from src.parser import discover_resumes, parse_resume_file
from src.pipeline import ResumeScreeningPipeline


def test_pdf_parsing_real_file():
    """Verify PyMuPDF successfully parses a real resume from the dataset."""
    sample_pdf = "resumes/candidate_01.pdf"
    if os.path.exists(sample_pdf):
        parsed = parse_resume_file(sample_pdf)
        assert parsed.is_success is True
        assert len(parsed.text) > 500
        assert "Kartikay Sinha" in parsed.text or "React" in parsed.text


def test_txt_parsing(tmp_path):
    """Verify plain text file parsing."""
    txt_file = tmp_path / "resume_sample.txt"
    txt_file.write_text("John Doe\nPython and LangChain Engineer\nEmail: john@example.com", encoding="utf-8")

    parsed = parse_resume_file(str(txt_file))
    assert parsed.is_success is True
    assert "John Doe" in parsed.text
    assert "LangChain" in parsed.text


def test_malformed_resume_does_not_crash(tmp_path):
    """A corrupted binary PDF file must fail gracefully and not throw unhandled exceptions."""
    corrupted_pdf = tmp_path / "corrupt.pdf"
    # Write garbage bytes
    corrupted_pdf.write_bytes(b"NOT A VALID PDF FILE HEADER \x00\x01\x02 GARBAGE DATA")

    parsed = parse_resume_file(str(corrupted_pdf))
    assert parsed.is_success is False
    assert parsed.error_message is not None
    assert "PDF" in parsed.error_message or "error" in parsed.error_message.lower()


def test_pipeline_continues_with_malformed_resume(tmp_path):
    """The batch pipeline must process valid files even when a malformed file is present."""
    # Create 1 valid txt resume and 1 malformed file in a temporary folder
    valid_file = tmp_path / "candidate_valid.txt"
    valid_file.write_text(
        "Valid Candidate\nPython Engineer\nLangChain agent developer\nEmail: valid@example.com",
        encoding="utf-8",
    )

    bad_file = tmp_path / "candidate_broken.pdf"
    bad_file.write_bytes(b"INVALID PDF DATA")

    pipeline = ResumeScreeningPipeline()
    output = pipeline.run_batch(input_path=str(tmp_path))

    # Total discovered should be 2
    assert output.batch_summary.total_resumes == 2
    # One succeeded, one failed
    assert output.batch_summary.successfully_parsed == 1
    assert output.batch_summary.failed_unreadable == 1
    # 1 eligible candidate from valid_file
    assert output.batch_summary.eligible == 1
    # Batch did not crash and produced valid output
    assert len(output.candidates) == 2


def test_discover_resumes_ordering(tmp_path):
    """Discovered resumes must be sorted deterministically."""
    (tmp_path / "candidate_02.pdf").write_text("fake", encoding="utf-8")
    (tmp_path / "candidate_01.pdf").write_text("fake", encoding="utf-8")
    (tmp_path / "candidate_10.pdf").write_text("fake", encoding="utf-8")

    discovered = discover_resumes(str(tmp_path))
    names = [Path(p).name for p in discovered]
    assert names == ["candidate_01.pdf", "candidate_02.pdf", "candidate_10.pdf"]
