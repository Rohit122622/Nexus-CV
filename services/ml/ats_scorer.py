"""
ATS Scoring Engine with ML-based scoring (XGBoost) + Explainable AI.
Uses model_hub for embeddings and XGBoost model.
Falls back to heuristic scoring if ML model not available.

Returns dict with:
  ats_score, matched_skills, missing_classified,
  feature_importance, confidence, confidence_reason,
  scoring_method (ml or heuristic)
"""

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

from utils.skill_normalizer import normalize as normalize_skills
from services.ml.skill_registry import (
    ROLE_SKILL_MAP, CRITICAL_SKILLS, NON_SKILL_WORDS, STANDARD_SECTIONS
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Model Hub for embeddings and ML model ──
try:
    import services.ml.model_hub as model_hub
    import numpy as np
except ImportError:
    model_hub = None
    np = None


def _semantic_match_skills(resume_skills, role_skills):
    """Use model_hub embeddings for semantic skill matching."""
    if model_hub is None:
        return _exact_match_skills(resume_skills, role_skills)

    try:
        resume_embeddings = model_hub.embed_batch(resume_skills)
        role_embeddings = model_hub.embed_batch(role_skills)

        if any(e is None for e in resume_embeddings) or any(e is None for e in role_embeddings):
            return _exact_match_skills(resume_skills, role_skills)

        matched = set()
        for i, r_emb in enumerate(role_embeddings):
            if r_emb is None:
                continue
            for j, s_emb in enumerate(resume_embeddings):
                if s_emb is None:
                    continue
                similarity = float(np.dot(r_emb, s_emb))
                if similarity > 0.70:
                    matched.add(role_skills[i].lower())
                    break

        return matched
    except Exception:
        return _exact_match_skills(resume_skills, role_skills)


def _exact_match_skills(resume_skills, role_skills):
    """Fallback: exact string matching."""
    resume_set = set(s.lower() for s in resume_skills)
    role_set = set(s.lower() for s in role_skills)
    return resume_set.intersection(role_set)


def classify_missing_skills(missing, role):
    """Classify missing skills as critical or nice-to-have."""
    critical_set = set(s.lower() for s in CRITICAL_SKILLS.get(role, []))
    critical_missing = [s for s in missing if s.lower() in critical_set]
    nice_to_have = [s for s in missing if s.lower() not in critical_set]
    return {
        "critical": critical_missing[:10],
        "nice_to_have": nice_to_have[:10]
    }


def _compute_features(parsed_data, role, matched_ratio):
    """
    Compute 7-feature vector for XGBoost prediction.

    Features:
        0: skill_match_ratio       (0.0–1.0)
        1: keyword_density         (role keywords / total words)
        2: section_completeness    (sections_found / 5)
        3: experience_years_norm   (capped at 15)
        4: has_education           (0 or 1)
        5: bullet_count_norm       (capped at 30)
        6: word_count_norm         (capped at 600)
    """
    text = parsed_data.get("text", "")
    word_count = parsed_data.get("word_count", len(text.split()))
    sections = parsed_data.get("sections_found", [])
    experience_years = parsed_data.get("experience_years", 0)
    education = parsed_data.get("education_detected", False)
    bullet_count = parsed_data.get("bullet_count", 0)

    # Keyword density: role-specific keywords in resume
    role_keywords = set(s.lower() for s in CRITICAL_SKILLS.get(role, []))
    text_lower = text.lower()
    kw_count = sum(1 for kw in role_keywords if kw in text_lower)
    keyword_density = kw_count / max(len(role_keywords), 1)

    # Section completeness
    section_ratio = len(sections) / len(STANDARD_SECTIONS) if sections else 0.3

    features = [
        max(0.0, min(matched_ratio, 1.0)),
        max(0.0, min(keyword_density, 1.0)),
        max(0.0, min(section_ratio, 1.0)),
        max(0.0, min(experience_years / 15.0, 1.0)),
        1.0 if education else 0.0,
        max(0.0, min(bullet_count / 30.0, 1.0)),
        max(0.0, min(word_count / 600.0, 1.0))
    ]
    return features


def _compute_confidence(features, scoring_method):
    """
    Compute confidence score (0-100) based on data quality.
    Higher confidence = more complete data, more reliable score.
    """
    non_zero = sum(1 for f in features if f > 0.05)
    completeness = non_zero / len(features)

    # ML gets higher base confidence than heuristic
    base = 0.80 if scoring_method == "ml" else 0.60

    confidence = base * 0.5 + completeness * 0.5
    confidence = min(1.0, max(0.2, confidence))
    confidence_pct = round(confidence * 100)

    if confidence_pct >= 80:
        reason = "High — resume is well-structured with strong data quality"
    elif confidence_pct >= 60:
        reason = "Moderate — some resume sections missing or limited content"
    else:
        reason = "Low — resume has insufficient data for reliable scoring"

    return confidence_pct, reason


def _get_feature_importance(xgb_model, features, feature_names):
    """
    Get per-prediction feature contributions using model importances.
    Returns dict of feature_name -> impact_percentage.
    """
    try:
        importances = xgb_model.feature_importances_
        weighted = importances * features
        total = sum(weighted)
        if total == 0:
            return {name: round(float(imp) * 100, 1) for name, imp in zip(feature_names, importances)}

        contributions = {}
        for name, w in zip(feature_names, weighted):
            contributions[name] = round(float(w / total) * 100, 1)
        return contributions
    except Exception:
        return {}


def calculate_ats_score(parsed_data, role="Software Engineer"):
    """
    Calculate ATS score using XGBoost ML model (preferred) or heuristic fallback.

    Returns dict:
        ats_score: int (0-100)
        matched_skills: list
        missing_classified: dict (critical, nice_to_have)
        feature_importance: dict (feature_name -> %)
        confidence: int (0-100)
        confidence_reason: str
        scoring_method: 'ml' or 'heuristic'
    """
    try:

        # ── Load job roles for skill matching ──
        roles_path = os.path.join(PROJECT_ROOT, "data", "job_roles.json")
        try:
            with open(roles_path, "r", encoding="utf-8") as f:
                job_roles = json.load(f)
        except Exception:
            job_roles = {}

        # ── Role-Aware Strict Filter ──
        raw_skills = parsed_data.get("skills", [])
        
        # Clean skill extraction: max 3 words, < 30 chars, no non-skill words
        skills = [
            s.lower().strip()
            for s in raw_skills
            if isinstance(s, str)
            and len(s.split()) <= 3
            and len(s) < 30
            and s.lower().strip() not in NON_SKILL_WORDS
            and not any(w in s.lower() for w in NON_SKILL_WORDS)
        ]
        skills = list(set(skills))

        # Apply role filter (relaxed: if role not in map, use all cleaned skills)
        allowed = set(ROLE_SKILL_MAP.get(role, []))
        if allowed:
            # Prefer role-matched skills, but keep all clean skills as fallback
            role_matched = [s for s in skills if s in allowed]
            resume_skills = role_matched[:20] if role_matched else skills[:20]
        else:
            resume_skills = skills[:20]

        # ── Build required_skills from job_roles + ROLE_SKILL_MAP ──
        jd_skills_raw = list(set(
            [s.lower() for s in job_roles.get(role, [])] +
            ROLE_SKILL_MAP.get(role, [])
        ))
        # Clean JD / required skills: normalize, max 3 words, no non-skill words
        required_skills = list(set([
            s.lower().strip()
            for s in jd_skills_raw
            if len(s.split()) <= 3
            and s.lower().strip() not in NON_SKILL_WORDS
        ]))

        # ── Skill matching: normalized intersection ──
        matched_set = _semantic_match_skills(resume_skills, required_skills)
        
        # Also do exact normalized intersection
        exact_matched = set([
            s for s in resume_skills
            if s in required_skills
        ])
        matched_set = matched_set | exact_matched
        
        # CLEAN MATCHED SKILLS: max 2 words, lowercase, deduplicated, no non-skill words
        matched_skills = list(set([
            s for s in matched_set
            if isinstance(s, str)
            and len(s.split()) <= 2
            and s == s.lower()
            and s.strip() not in NON_SKILL_WORDS
            and not any(w in s for w in NON_SKILL_WORDS)
        ]))[:20]
        
        all_missing = [
            s for s in required_skills
            if s.lower() not in matched_set
            and s.strip() not in NON_SKILL_WORDS
        ]
        missing_classified = classify_missing_skills(all_missing, role)
        matched_ratio = len(matched_skills) / max(len(required_skills), 1)

        # Apply global normalization
        matched_skills = normalize_skills(matched_skills)

        # ── Compute feature vector ──
        features = _compute_features(parsed_data, role, matched_ratio)
        feature_names = [
            "skill_match_ratio", "keyword_density", "section_completeness",
            "experience_years", "has_education", "bullet_count", "word_count"
        ]

        # ── Deterministic ATS Scoring ──
        section_count = parsed_data.get("section_count", len(parsed_data.get("sections_found", [])))
        jd_skills = required_skills  # already cleaned above

        matched = set([s for s in resume_skills if s in jd_skills])
        # Also count semantic matches from matched_skills
        matched = matched | set(matched_skills)
        skill_score = int((len(matched) / max(1, len(jd_skills))) * 50)

        # ── FIXED: keyword_score now measures keyword density in FULL TEXT ──
        # This is different from skill_score (which only checks skill list intersection)
        text_lower = parsed_data.get("text", "").lower()
        all_role_keywords = set(s.lower() for s in ROLE_SKILL_MAP.get(role, []) + CRITICAL_SKILLS.get(role, []))
        kw_found_in_text = sum(1 for kw in all_role_keywords if kw in text_lower)
        keyword_density = kw_found_in_text / max(len(all_role_keywords), 1)
        keyword_score = int(keyword_density * 30)

        # Completeness: 5 standard sections → 4 points each → 20 max
        sections_found = [s.lower() for s in parsed_data.get("sections_found", [])]
        has_education = "education" in sections_found or parsed_data.get("education_detected")
        has_experience = "experience" in sections_found
        has_projects = "projects" in sections_found
        has_skills = "skills" in sections_found or len(parsed_data.get("skills", [])) > 0
        has_summary = "summary" in sections_found or "objective" in sections_found or "profile" in sections_found
        
        sections = {
            "education": has_education,
            "experience": has_experience,
            "projects": has_projects,
            "skills": has_skills,
            "summary": has_summary
        }

        logger.debug("ATS sections detected: %s (raw: %s)", sections, sections_found)

        section_count_present = sum(1 for v in sections.values() if v)
        # If all 5 sections exist → full 20
        if section_count_present == 5:
            completeness_score = 20
        else:
            completeness_score = section_count_present * 4

        logger.debug("ATS completeness_score=%d/20", completeness_score)

        final_score = skill_score + keyword_score + completeness_score

        # ── Logical consistency: if matched_skills > 0 → score ≥ 40 ──
        if len(matched) > 0:
            final_score = max(final_score, 40)
        if len(matched) >= 3:
            final_score = max(final_score, 50)

        # ── Try ML scoring (minor adjustment only) ──
        scoring_method = "heuristic"
        feature_importance = {}
        ats_score = final_score

        # ── FIXED: Correct model path (model/ at project root) ──
        model_path = os.path.join(PROJECT_ROOT, "model", "ats_xgb.pkl")

        model = None

        if os.path.exists(model_path):
            try:
                from joblib import load
                model = load(model_path)
                logger.info("XGBoost model loaded from %s", model_path)
            except Exception as e:
                logger.warning("XGBoost load failed: %s", e)
                model = None
        else:
            logger.info("XGBoost model not found at %s — using heuristic ATS", model_path)
            model = None

        if model:
            try:
                import numpy as _np
                score = model.predict([_np.array(features, dtype=_np.float32)])
                
                raw_score = float(score[0])
                ml_adjustment = (raw_score - 50) / 5.0
                adjusted_score = final_score + ml_adjustment

                # Re-apply logical consistency after ML adjustment
                if len(matched) > 0:
                    adjusted_score = max(adjusted_score, 50)

                ats_score = int(max(5, min(100, adjusted_score)))
                scoring_method = "ml"
                feature_importance = _get_feature_importance(model, features, feature_names)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("XGBoost prediction failed: %s", e)
                # Fall back to heuristic rule_based_score
                scoring_method = "heuristic"
        else:
            # Ensures we fall through to rule_based_score handling below
            scoring_method = "heuristic"

        # ── Heuristic fallback ──
        if scoring_method == "heuristic":
            ats_score = int(max(10, min(100, final_score)))

            feature_importance = {
                "skill_match_ratio": round(matched_ratio * 50, 1),
                "keyword_density": round(features[1] * 30, 1),
                "section_completeness": round(features[2] * 20, 1),
                "experience_years": 0,
                "has_education": 0,
                "bullet_count": 0,
                "word_count": 0
            }

        # ── Confidence ──
        confidence, confidence_reason = _compute_confidence(features, scoring_method)

        # Build flat missing_skills list (backward compat)
        all_missing_flat = normalize_skills(
            missing_classified.get("critical", []) + missing_classified.get("nice_to_have", [])
        )

        return {
            "ats_score": ats_score,
            "matched_skills": matched_skills,
            "missing_classified": missing_classified,
            "feature_importance": feature_importance,
            "confidence": confidence,
            "confidence_reason": confidence_reason,
            "scoring_method": scoring_method,
            # ── Backward-compatible keys (used by app.py, career_recommender, pdf_generator) ──
            "missing_skills": all_missing_flat,
            "skill_score": skill_score,
            "keyword_score": keyword_score,
            "completeness_score": completeness_score
        }
    except Exception as e:
        import traceback
        import logging
        logging.getLogger(__name__).error(f"Global ATS calculation error: {e}\n{traceback.format_exc()}")
        return {
            "ats_score": 10,
            "matched_skills": [],
            "missing_classified": {"critical": ["Try improving role-specific skills"], "nice_to_have": []},
            "feature_importance": {},
            "confidence": 0,
            "confidence_reason": "Try improving role-specific skills",
            "scoring_method": "error_fallback",
            "missing_skills": ["Try improving role-specific skills"],
            "skill_score": 0,
            "keyword_score": 0,
            "completeness_score": 0
        }

