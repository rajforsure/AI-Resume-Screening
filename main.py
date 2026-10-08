"""
AI Resume Screening & Ranking System - Command-Line Interface.
Usage:
    python main.py --input ./resumes --output ./output/results.json
"""

import argparse
import sys
from pathlib import Path

from src.llm import get_llm_adapter
from src.pipeline import ResumeScreeningPipeline
from src.utils import setup_logger

logger = setup_logger("cli")


def print_banner():
    banner = r"""
================================================================================
   AI Resume Screening & Ranking System — SDE Intern Evaluation Engine
================================================================================
    """
    print(banner)


def format_progress(current: int, total: int, file_name: str):
    """Displays a clean single-line progress indicator."""
    percent = (current / total) * 100 if total > 0 else 0
    sys.stdout.write(f"\r  [{current:02d}/{total:02d}] ({percent:5.1f}%) Processing: {file_name:<25}")
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def print_summary_report(output_data, output_path: str):
    """Prints a clear, formatted batch summary report to stdout."""
    summary = output_data.batch_summary
    candidates = output_data.candidates
    eligible_list = [c for c in candidates if getattr(c, "eligible", False)]
    rejected_list = [c for c in candidates if not getattr(c, "eligible", False)]

    print("\n" + "=" * 80)
    print(" BATCH SCREENING SUMMARY")
    print("=" * 80)
    print(f"  Total Resumes Discovered : {summary.total_resumes}")
    print(f"  Successfully Parsed      : {summary.successfully_parsed}")
    print(f"  Eligible for Ranking     : {summary.eligible}")
    print(f"  Rejected (Ineligible)    : {summary.rejected}")
    print(f"  Failed / Unreadable      : {summary.failed_unreadable}")
    print("-" * 80)

    # Top Ranked Candidates Table
    if eligible_list:
        print("\nTOP RANKED ELIGIBLE CANDIDATES (Top 10 Preview):")
        print(f" {'Rank':<5} | {'Candidate Name':<26} | {'Total':<6} | {'AI (40)':<8} | {'Py (30)':<8} | {'Cloud':<6} | {'GH':<4}")
        print("-" * 75)
        for cand in eligible_list[:10]:
            sb = cand.score_breakdown
            print(
                f" {cand.rank:<5} | {cand.candidate_name[:26]:<26} | {cand.total_score:<6.1f} | "
                f"{sb.ai_project_depth:<8.1f} | {sb.python_backend:<8.1f} | {sb.cloud_fullstack:<6.1f} | {sb.github:<4.1f}"
            )
        if len(eligible_list) > 10:
            print(f" ... and {len(eligible_list) - 10} more eligible candidates (see full JSON).")

    # Sample Rejections
    if rejected_list:
        print("\nSAMPLE REJECTED CANDIDATES (Non-Competitive):")
        for cand in rejected_list[:5]:
            reasons = "; ".join(cand.rejection_reasons)
            print(f"  * {cand.candidate_name:<24} -> {reasons}")
        if len(rejected_list) > 5:
            print(f"  ... and {len(rejected_list) - 5} more rejected candidates.")

    print("\n" + "=" * 80)
    print(f" Machine-readable JSON output successfully saved to:")
    print(f" -> {Path(output_path).resolve()}")
    print("=" * 80 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Resume Screening & Ranking System - Batch Processor"
    )
    parser.add_argument(
        "--input",
        "-i",
        default="./resumes",
        help="Path to input directory containing resumes or a single resume file (default: ./resumes)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./output/results.json",
        help="Path to destination JSON file (default: ./output/results.json)",
    )
    parser.add_argument(
        "--provider",
        "-p",
        choices=["auto", "groq", "gemini", "openai", "heuristic"],
        default="auto",
        help="LLM provider choice: auto, groq, gemini, openai, heuristic (default: auto)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable detailed debug logging",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print_banner()

    # Configure LLM provider
    provider = get_llm_adapter(preferred_provider=args.provider)
    pipeline = ResumeScreeningPipeline(llm_provider=provider)

    # Execute batch
    output_data = pipeline.run_batch(
        input_path=args.input,
        output_file_path=args.output,
        progress_callback=format_progress,
    )

    # Print summary report
    print_summary_report(output_data, args.output)


if __name__ == "__main__":
    main()
