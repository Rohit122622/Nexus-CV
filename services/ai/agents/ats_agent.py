"""
ATS Agent — Wrapper around the ML ATS scorer. NO LLM calls.

Pure computation — delegates to calculate_ats_score() and returns
structured breakdown with confidence.
"""

import logging

logger = logging.getLogger(__name__)


class ATSAgent:
    """ATS scoring agent — deterministic ML/heuristic scoring."""

    def evaluate(self, parsed_data, role):
        """
        Score resume using the ATS engine.

        Args:
            parsed_data: dict from parse_resume()
            role: target role name

        Returns:
            dict with ats_score, breakdown, confidence, scoring_method
        """
        try:
            from services.ml.ats_scorer import calculate_ats_score
            result = calculate_ats_score(parsed_data, role)

            return {
                "ats_score": result.get("ats_score", 0),
                "breakdown": {
                    "skill_score": result.get("skill_score", 0),
                    "keyword_score": result.get("keyword_score", 0),
                    "completeness_score": result.get("completeness_score", 0),
                },
                "matched_skills": result.get("matched_skills", []),
                "missing_skills": result.get("missing_skills", []),
                "missing_classified": result.get("missing_classified", {}),
                "confidence": result.get("confidence", 50),
                "scoring_method": result.get("scoring_method", "heuristic"),
                "feature_importance": result.get("feature_importance", {}),
            }
        except Exception as e:
            logger.error("ATSAgent failed: %s", e)
            return {
                "ats_score": 0,
                "breakdown": {"skill_score": 0, "keyword_score": 0, "completeness_score": 0},
                "matched_skills": [],
                "missing_skills": [],
                "missing_classified": {},
                "confidence": 10,
                "scoring_method": "failed",
                "feature_importance": {},
            }
