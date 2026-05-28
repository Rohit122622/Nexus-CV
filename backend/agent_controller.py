"""
Agent Controller — Central orchestrator for Nexus CV Agentic AI Platform.
Coordinates all agents (Parser, Scoring, Matching, Generation, Comparison, Ranking)
through unified pipeline methods with error handling and fallbacks.

Usage:
    from agent_controller import AgentController
    controller = AgentController()
    result = controller.analyze_resume(pdf_path, role, jd_text)
"""

import os
import sys
import logging
import time

logger = logging.getLogger(__name__)

# ── Project root path setup ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ── Import all agents ──
from services.processing.resume_parser import parse_resume
from services.ml.ats_scorer import calculate_ats_score
from services.processing.multi_role_predictor import predict_multiple_roles
from services.processing.jd_matcher import match_jd
from services.processing.resume_insights import analyze_resume_insights
from services.processing.resume_suggestions import generate_suggestions
from services.processing.career_recommender import recommend_career

try:
    from services.ai import gemini_agent
except ImportError:
    gemini_agent = None

try:
    from services.ml import model_hub
    import numpy as np
except ImportError:
    model_hub = None
    np = None

try:
    from services.processing.bulk_screener import screen_resumes
except ImportError:
    screen_resumes = None


class AgentController:
    """Central orchestrator for the multi-agent resume analysis pipeline."""

    def __init__(self):
        self._pipeline_log = []

    def _log(self, agent, status, duration=0, details=""):
        """Log agent execution step."""
        entry = {
            "agent": agent,
            "status": status,
            "duration_ms": round(duration * 1000),
            "details": details
        }
        self._pipeline_log.append(entry)
        logger.info("Agent [%s] %s in %dms — %s", agent, status, entry["duration_ms"], details)

    def get_pipeline_log(self):
        """Return execution log for the last pipeline run."""
        return list(self._pipeline_log)

    # ────────────── FULL ANALYSIS PIPELINE ──────────────

    def analyze_resume(self, pdf_path, role=None, jd_text=None):
        """
        Full resume analysis pipeline:
          1. Parser Agent → structured data
          2. Role Predictor → multi-role prediction
          3. Scoring Agent → ML ATS score + XAI
          4. Matching Agent → JD matching (if JD provided)
          5. Generation Agent → insights, suggestions, roadmap, skill gap
        
        Returns dict with all results + pipeline_log.
        """
        self._pipeline_log = []
        result = {"success": False}

        # ── Step 1: Parser Agent ──
        t0 = time.time()
        try:
            parsed_data = parse_resume(pdf_path)
            resume_text = parsed_data.get("text", "")
            if not resume_text or len(resume_text) < 50:
                result["error"] = "Could not extract meaningful text from PDF"
                return result
            self._log("Parser", "success", time.time() - t0,
                      f"Extracted {parsed_data.get('word_count', 0)} words, "
                      f"{len(parsed_data.get('skills', []))} skills, "
                      f"{parsed_data.get('section_count', 0)} sections")
        except Exception as e:
            self._log("Parser", "failed", time.time() - t0, str(e))
            result["error"] = f"Parser failed: {e}"
            return result

        # ── Step 2: Role Predictor Agent ──
        t0 = time.time()
        try:
            multi_roles = predict_multiple_roles(parsed_data["skills"])
            if not role:
                role = multi_roles[0]["role"] if multi_roles else "Software Engineer"
            self._log("RolePredictor", "success", time.time() - t0,
                      f"Top role: {role}, {len(multi_roles)} predictions")
        except Exception as e:
            self._log("RolePredictor", "fallback", time.time() - t0, str(e))
            multi_roles = [{"role": "Software Engineer", "confidence": 0.5}]
            if not role:
                role = "Software Engineer"

        # ── Step 3: Scoring Agent ──
        t0 = time.time()
        try:
            ats_result = calculate_ats_score(parsed_data, role)
            self._log("Scorer", "success", time.time() - t0,
                      f"Score: {ats_result['ats_score']}/100 "
                      f"(method: {ats_result.get('scoring_method', 'unknown')}, "
                      f"confidence: {ats_result.get('confidence', 0)}%)")
        except Exception as e:
            self._log("Scorer", "failed", time.time() - t0, str(e))
            ats_result = {"ats_score": 0, "matched_skills": [], "missing_classified": {}}

        # ── Step 4: Matching Agent (if JD provided) ──
        jd_result = None
        if jd_text:
            t0 = time.time()
            try:
                jd_result = match_jd(parsed_data["skills"], jd_text)
                self._log("Matcher", "success", time.time() - t0,
                          f"Match: {jd_result.get('match_percentage', 0)}% "
                          f"(method: {jd_result.get('match_method', 'unknown')})")
            except Exception as e:
                self._log("Matcher", "failed", time.time() - t0, str(e))
                jd_result = {"match_percentage": 0, "matched_skills": [], "missing_keywords": []}

        # ── Step 5: Generation Agent (insights + suggestions + roadmap) ──
        t0 = time.time()
        try:
            insights = analyze_resume_insights(
                resume_text, parsed_data["skills"],
                ats_data=ats_result, role=role, jd_text=jd_text
            )
            missing_skills = ats_result.get("missing_skills", [])
            suggestions = generate_suggestions(insights, missing_skills, jd_result or {})
            roadmap = recommend_career(ats_result, role, parsed_data["skills"], insights=insights, jd_result=jd_result)
            self._log("Generator", "success", time.time() - t0,
                      f"Insights: {insights.get('strength', 'N/A')}, "
                      f"{len(suggestions)} suggestions")
        except Exception as e:
            self._log("Generator", "fallback", time.time() - t0, str(e))
            insights = {}
            suggestions = []
            roadmap = {}

        # ── Step 6: Skill Gap Analysis (Gemini) ──
        skill_gaps = None
        if gemini_agent:
            t0 = time.time()
            try:
                skill_gaps = gemini_agent.generate_skill_gap_analysis(
                    parsed_data["skills"], role, ats_result
                )
                self._log("SkillGap", "success", time.time() - t0,
                          f"{len(skill_gaps) if skill_gaps else 0} gaps identified")
            except Exception as e:
                self._log("SkillGap", "skipped", time.time() - t0, str(e))

        # ── Build safe result (NO None values) ──
        result = {
            "success": True,
            "parsed_data": parsed_data,
            "role": role or "Software Engineer",
            "multi_roles": multi_roles or [],
            "ats_result": ats_result or {},
            "ats_score": (ats_result or {}).get("ats_score", 10),
            "matched_skills": (ats_result or {}).get("matched_skills", []),
            "missing_skills": (ats_result or {}).get("missing_skills", []),
            "jd_result": jd_result or {"match_percentage": 0, "matched_skills": [], "missing_keywords": []},
            "insights": insights or {},
            "suggestions": suggestions or [],
            "roadmap": roadmap or {},
            "skill_gaps": skill_gaps,
            "pipeline_log": self.get_pipeline_log()
        }

        return result

    # ────────────── COMPARISON PIPELINE ──────────────

    def compare_resumes(self, pdf_path_a, pdf_path_b, role, jd_text=None):
        """
        Compare two resumes through the agent pipeline.
        Returns comparison dict + semantic_similarity + pipeline_log.
        """
        self._pipeline_log = []

        # Parse both
        t0 = time.time()
        parsed_a = parse_resume(pdf_path_a)
        parsed_b = parse_resume(pdf_path_b)
        self._log("Parser", "success", time.time() - t0, "Parsed both resumes")

        # Score both
        t0 = time.time()
        ats_a = calculate_ats_score(parsed_a, role)
        ats_b = calculate_ats_score(parsed_b, role)
        self._log("Scorer", "success", time.time() - t0,
                  f"A: {ats_a['ats_score']}/100, B: {ats_b['ats_score']}/100")

        # Semantic similarity between resumes
        semantic_similarity = 0.0
        if model_hub is not None and np is not None:
            t0 = time.time()
            try:
                emb_a = model_hub.embed_text(parsed_a["text"][:2000])
                emb_b = model_hub.embed_text(parsed_b["text"][:2000])
                if emb_a is not None and emb_b is not None:
                    semantic_similarity = round(float(np.dot(emb_a, emb_b)) * 100, 1)
                self._log("Embedder", "success", time.time() - t0,
                          f"Similarity: {semantic_similarity}%")
            except Exception as e:
                self._log("Embedder", "failed", time.time() - t0, str(e))

        # Gemini AI comparison
        ai_comparison = None
        if gemini_agent:
            t0 = time.time()
            try:
                ai_comparison = gemini_agent.compare_resumes(
                    parsed_a["text"], parsed_b["text"],
                    ats_a, ats_b, role, jd_text
                )
                self._log("GeminiCompare", "success", time.time() - t0, "AI comparison complete")
            except Exception as e:
                self._log("GeminiCompare", "skipped", time.time() - t0, str(e))

        return {
            "parsed_a": parsed_a,
            "parsed_b": parsed_b,
            "ats_a": ats_a,
            "ats_b": ats_b,
            "semantic_similarity": semantic_similarity,
            "ai_comparison": ai_comparison,
            "pipeline_log": self.get_pipeline_log()
        }

    # ────────────── BULK RANKING PIPELINE ──────────────

    def rank_resumes(self, pdf_paths, jd_text, role="Software Engineer"):
        """
        Rank multiple resumes against a JD.
        Delegates to bulk_screener.screen_resumes with agent logging.
        """
        self._pipeline_log = []

        if screen_resumes is None:
            return {"error": "Bulk screening module not available"}

        t0 = time.time()
        result = screen_resumes(pdf_paths, jd_text, role)
        self._log("Ranker", "success", time.time() - t0,
                  f"Processed {result.get('total_processed', 0)} resumes")

        result["pipeline_log"] = self.get_pipeline_log()
        return result
