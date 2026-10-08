"""
Core pipeline orchestrator for the AI Resume Screening & Ranking System.
Coordinates document parsing, semantic extraction, hard eligibility filtering,
GitHub enrichment, 100-point rubric scoring, ranking, and batch reporting.
Guarantees uninterrupted batch processing even when individual resumes or external APIs fail.
"""

import json
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

from src.eligibility import evaluate_hard_eligibility
from src.extractor import extract_resume_data
from src.github import enrich_github_profile
from src.llm import BaseLLMProvider, get_llm_adapter
from src.models import (
    BatchSummary,
    EligibleCandidate,
    RejectedCandidate,
    RunOutput,
)
from src.parser import discover_resumes, parse_resume_file
from src.scorer import evaluate_eligible_candidate, rank_eligible_candidates
from src.utils import setup_logger

logger = setup_logger("pipeline")


class ResumeScreeningPipeline:
    """
    Production-grade batch resume screening and ranking orchestrator.
    """

    def __init__(self, llm_provider: Optional[BaseLLMProvider] = None):
        self.llm_provider = llm_provider or get_llm_adapter()

    def process_single_resume(
        self,
        file_path: str,
    ) -> Tuple[bool, Optional[Union[EligibleCandidate, RejectedCandidate]], Optional[str]]:
        """
        Processes an individual resume through the parsing, extraction,
        eligibility, enrichment, and scoring stages.
        Returns: (is_parsed_ok, candidate_result_or_none, error_message_or_none).
        """
        # 1. Parse document text
        parsed_doc = parse_resume_file(file_path)
        if not parsed_doc.is_success or not parsed_doc.text:
            return False, None, parsed_doc.error_message or "Failed to extract text"

        # 2. Extract semantic data
        try:
            extracted_data = extract_resume_data(
                text=parsed_doc.text,
                file_path=file_path,
                llm_provider=self.llm_provider,
            )
        except Exception as e:
            logger.warning(f"Semantic extraction error for {parsed_doc.file_name}: {str(e)}; fallback used.")
            # Resilient fallback extraction
            extracted_data = extract_resume_data(
                text=parsed_doc.text,
                file_path=file_path,
                llm_provider=None,
            )

        # 3. Apply Hard Eligibility Filtering BEFORE ranking
        eligibility = evaluate_hard_eligibility(extracted_data, parsed_doc.text)

        if not eligibility.eligible:
            # Candidate is rejected: no competitive ranking awarded
            rejected_candidate = RejectedCandidate(
                candidate_name=extracted_data.candidate_name,
                eligible=False,
                rejection_reasons=eligibility.rejection_reasons,
                matched_skills=eligibility.matched_skills,
            )
            return True, rejected_candidate, None

        # 4. Candidate is ELIGIBLE -> GitHub Enrichment
        github_enrichment = enrich_github_profile(extracted_data.github_url)

        # 5. Optional LLM project assessment if provider is active
        llm_assessment = None
        if self.llm_provider and self.llm_provider.is_available():
            try:
                assessment, _ = self.llm_provider.assess_project_quality(extracted_data.model_dump())
                llm_assessment = assessment
            except Exception as e:
                logger.warning(f"LLM project assessment error for {extracted_data.candidate_name}: {str(e)}")

        # 6. Score candidate across the 100-point rubric
        total_score, breakdown, proj_summary, gh_summary, strengths, concerns = evaluate_eligible_candidate(
            data=extracted_data,
            raw_text=parsed_doc.text,
            github_enrichment=github_enrichment,
            llm_assessment=llm_assessment,
        )

        # Rank will be assigned deterministically in batch ranking step
        eligible_candidate = EligibleCandidate(
            rank=0,
            candidate_name=extracted_data.candidate_name,
            eligible=True,
            total_score=total_score,
            score_breakdown=breakdown,
            matched_skills=extracted_data.skills,
            project_summary=proj_summary,
            github_summary=gh_summary,
            strengths=strengths,
            concerns=concerns,
        )

        return True, eligible_candidate, None

    def run_batch(
        self,
        input_path: str,
        output_file_path: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> RunOutput:
        """
        Executes batch resume screening on all discovered resumes in input_path.
        Compiles ranked eligible candidates, rejected candidates, and batch summary.
        Writes the results to output_file_path if provided.
        """
        resume_files = discover_resumes(input_path)
        total_resumes = len(resume_files)

        logger.info(f"Starting batch screening on {total_resumes} resume(s)...")

        parsed_count = 0
        failed_count = 0
        eligible_candidates: List[EligibleCandidate] = []
        rejected_candidates: List[RejectedCandidate] = []

        for idx, file_path in enumerate(resume_files, start=1):
            file_name = Path(file_path).name
            if progress_callback:
                progress_callback(idx, total_resumes, file_name)

            try:
                is_parsed, result, err = self.process_single_resume(file_path)
                if not is_parsed:
                    failed_count += 1
                    logger.warning(f"[{idx}/{total_resumes}] Failed parsing {file_name}: {err}")
                    # Also record as rejected/unreadable candidate with explicit reason
                    rejected_candidates.append(
                        RejectedCandidate(
                            candidate_name=Path(file_path).stem.replace("_", " ").title(),
                            eligible=False,
                            rejection_reasons=[f"File unreadable or corrupt: {err}"],
                            matched_skills=[],
                        )
                    )
                else:
                    parsed_count += 1
                    if isinstance(result, EligibleCandidate):
                        eligible_candidates.append(result)
                    elif isinstance(result, RejectedCandidate):
                        rejected_candidates.append(result)

            except Exception as e:
                # Catch-all to guarantee that an individual resume fault NEVER crashes the batch
                failed_count += 1
                logger.error(f"[{idx}/{total_resumes}] Unexpected error on {file_name}: {str(e)}")
                rejected_candidates.append(
                    RejectedCandidate(
                        candidate_name=Path(file_path).stem.replace("_", " ").title(),
                        eligible=False,
                        rejection_reasons=[f"Processing error: {str(e)}"],
                        matched_skills=[],
                    )
                )

        # Apply deterministic tie-breaking and assign ranks to eligible candidates
        ranked_eligible = rank_eligible_candidates(eligible_candidates)

        # Batch summary statistics
        batch_summary = BatchSummary(
            total_resumes=total_resumes,
            successfully_parsed=parsed_count,
            eligible=len(ranked_eligible),
            rejected=len(rejected_candidates),
            failed_unreadable=failed_count,
        )

        # Combine all candidates: ranked eligible candidates first, followed by rejected candidates
        all_candidates: List[Union[EligibleCandidate, RejectedCandidate]] = []
        all_candidates.extend(ranked_eligible)
        all_candidates.extend(rejected_candidates)

        output_data = RunOutput(
            batch_summary=batch_summary,
            candidates=all_candidates,
        )

        # Write results to output file if requested
        if output_file_path:
            out_path = Path(output_file_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(output_data.model_dump(), f, indent=2)
            logger.info(f"Machine-readable results written to {out_path.resolve()}")

        return output_data
