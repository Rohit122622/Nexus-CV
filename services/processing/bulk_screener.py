"""
Bulk Resume Screener — Hybrid Agentic RAG Hiring System.

Multi-Agent ML scoring + semantic matching + ReAct agent reasoning + XGBoost.
Uses model_hub for embeddings, ats_scorer for ML-based scoring,
multi-agent pipeline (Skill, Experience, ATS, Decision) for evaluation,
and gemini_agent for AI explanations.

Supports: individual PDFs and ZIP archives (auto-extracted).
Limit: 50 resumes max. Uses batching and embedding cache.

Ranking formula (5-signal Hybrid):
  Final Score = 0.30 * semantic_score
              + 0.20 * ATS_score
              + 0.20 * agent_score
              + 0.15 * skill_score
              + 0.15 * xgboost_score
"""

import os
import zipfile
import logging
import hashlib

from utils.skill_normalizer import normalize as normalize_skills

logger = logging.getLogger(__name__)

try:
    import services.ml.model_hub as model_hub
    import numpy as np
except ImportError:
    model_hub = None
    np = None

from services.processing.resume_parser import parse_resume
from services.ml.ats_scorer import calculate_ats_score

try:
    import services.ai.gemini_agent as gemini_agent
except ImportError:
    gemini_agent = None

# ── Multi-Agent Reasoner ──
try:
    from services.ai.agent_reasoner import run_agent, run_agent_batch
except ImportError:
    run_agent = None
    run_agent_batch = None

# ── XGBoost ML scoring ──
_xgb_model = None
_XGB_FEATURES = [
    "skill_match_ratio", "keyword_density", "experience_years",
    "project_count", "ats_score", "resume_length", "section_completeness"
]


def _load_or_train_xgboost():
    """Load XGBoost model with feature validation. Retrain if mismatch."""
    global _xgb_model
    if _xgb_model is not None:
        return _xgb_model

    import os as _os
    PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    model_path = _os.path.join(PROJECT_ROOT, "model", "ats_xgb.pkl")

    # Try loading existing model
    if _os.path.exists(model_path):
        try:
            import joblib
            model = joblib.load(model_path)
            # Feature validation: check if model expects correct number of features
            if hasattr(model, 'n_features_in_') and model.n_features_in_ == len(_XGB_FEATURES):
                _xgb_model = model
                print("[OK] XGBoost model loaded")
                return _xgb_model
            else:
                expected = getattr(model, 'n_features_in_', 'unknown')
                print(f"[WARN] XGBoost feature mismatch: expected {len(_XGB_FEATURES)}, model has {expected}")
        except Exception as e:
            print(f"[WARN] XGBoost load failed: {e}")

    # Retrain with synthetic data
    try:
        _xgb_model = _retrain_xgboost(model_path)
        return _xgb_model
    except Exception as e:
        print(f"[WARN] XGBoost retrain failed: {e}")
        return None


def _retrain_xgboost(model_path):
    """Train a fresh XGBoost model using synthetic data."""
    try:
        from xgboost import XGBRegressor
        import joblib
        import numpy as _np
    except ImportError:
        print("[WARN] xgboost/joblib not installed, ML scoring unavailable")
        return None

    # Generate synthetic training data
    _np.random.seed(42)
    n_samples = 500

    # Features: skill_match_ratio, keyword_density, experience_years,
    #           project_count, ats_score, resume_length, section_completeness
    X = _np.column_stack([
        _np.random.uniform(0, 1, n_samples),         # skill_match_ratio
        _np.random.uniform(0, 1, n_samples),         # keyword_density
        _np.random.uniform(0, 15, n_samples),        # experience_years
        _np.random.randint(0, 8, n_samples),         # project_count
        _np.random.uniform(20, 100, n_samples),      # ats_score
        _np.random.uniform(100, 1500, n_samples),    # resume_length (word count)
        _np.random.uniform(0, 1, n_samples),         # section_completeness
    ])

    # Target: weighted combination reflecting realistic scoring
    y = (
        X[:, 0] * 25 +  # skill_match_ratio
        X[:, 1] * 20 +  # keyword_density
        _np.clip(X[:, 2] / 10, 0, 1) * 15 +  # experience_years (normalized)
        _np.clip(X[:, 3] / 5, 0, 1) * 10 +   # project_count (normalized)
        X[:, 4] * 0.2 +  # ats_score contribution
        _np.clip(X[:, 5] / 800, 0, 1) * 10 +  # resume_length
        X[:, 6] * 20     # section_completeness
    ) + _np.random.normal(0, 3, n_samples)  # noise
    y = _np.clip(y, 0, 100)

    model = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X, y)

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print("[WARN] XGBoost retrained due to mismatch (synthetic data)")
    print(f"[OK] XGBoost model saved to {model_path}")

    return model


def _compute_xgboost_score(parsed_data, ats_result, skill_match_ratio):
    """Compute XGBoost prediction for a single candidate."""
    model = _load_or_train_xgboost()
    if model is None:
        return None

    try:
        import numpy as _np
        from services.ml.skill_registry import ROLE_SKILL_MAP, STANDARD_SECTIONS

        text = parsed_data.get("text", "")
        sections_found = [s.lower() for s in parsed_data.get("sections_found", [])]
        section_completeness = len([s for s in STANDARD_SECTIONS if s.lower() in sections_found]) / len(STANDARD_SECTIONS)

        features = _np.array([[
            skill_match_ratio,
            ats_result.get("keyword_score", 0) / 30.0,  # keyword_density normalized
            parsed_data.get("experience_years", 0),
            len([s for s in sections_found if s == "projects"]),  # rough project section indicator
            ats_result.get("ats_score", 50),
            parsed_data.get("word_count", len(text.split())),
            section_completeness,
        ]])

        prediction = float(model.predict(features)[0])
        return max(0, min(100, round(prediction, 1)))
    except Exception as e:
        logger.warning("XGBoost prediction failed: %s", e)
        return None


# ── Progress callback for SSE ──
_progress_callback = None


def set_progress_callback(callback):
    """Set a callback function for progress updates."""
    global _progress_callback
    _progress_callback = callback


def _emit_progress(step, detail=""):
    """Emit progress update if callback is set."""
    if _progress_callback:
        try:
            _progress_callback(step, detail)
        except Exception:
            pass


def extract_pdfs_from_zip(zip_path, output_dir):
    """Extract all PDF files from a ZIP archive."""
    extracted = []
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            for name in z.namelist():
                if name.lower().endswith('.pdf') and not name.startswith('__MACOSX'):
                    z.extract(name, output_dir)
                    extracted.append(os.path.join(output_dir, name))
    except zipfile.BadZipFile:
        logger.warning("Invalid ZIP file: %s", zip_path)
    return extracted


def _generate_reason(candidate, rank):
    """Generate a human-readable reason for ranking."""
    parts = []

    sem = candidate.get("semantic_score", 0)
    ats = candidate.get("ats_score", 0)
    skill = candidate.get("skill_match_pct", 0)

    if sem >= 70:
        parts.append("strong semantic alignment with JD")
    elif sem >= 50:
        parts.append("moderate semantic relevance")

    if ats >= 70:
        parts.append(f"excellent ATS score ({ats}/100)")
    elif ats >= 50:
        parts.append(f"solid ATS score ({ats}/100)")

    if skill >= 60:
        parts.append(f"high skill match ({skill}%)")
    elif skill >= 40:
        parts.append(f"adequate skill coverage ({skill}%)")

    matched_count = len(candidate.get("matched_skills", []))
    if matched_count > 0:
        parts.append(f"{matched_count} matched skills")

    if not parts:
        parts.append("general profile relevance")

    reason = f"Ranked #{rank}: " + ", ".join(parts) + "."
    return reason


def screen_resumes(pdf_paths, jd_text, role="Software Engineer", top_n=3):
    """
    Screens multiple PDFs against a JD and ranks by hybrid agentic score.

    Pipeline:
      1. Parse each resume (ML)
      2. Compute ATS score (ML + heuristic)
      3. Compute semantic similarity (embedding)
      4. Pre-rank by ML score
      5. Run ReAct agent on top candidates (LLM)
      6. Compute final hybrid score
      7. Return ranked results with evidence

    Returns dict:
      top_candidates: list of dicts with rank, filename, scores, skills, reason, evidence, confidence
      all_results: list of dicts with rank, filename, scores
      total_processed: int
      top_n: int
      scoring_details: str (formula used)
      agent_enabled: bool
      error: str (only if failed)
    """
    if not pdf_paths or not jd_text:
        return {"error": "No PDFs or JD provided"}

    # ── Encode JD for semantic similarity (once) ──
    jd_embedding = None
    if model_hub is not None:
        jd_embedding = model_hub.embed_text(jd_text)

    results = []
    seen_hashes = set()  # Deduplicate identical resumes

    for path in pdf_paths[:50]:
        try:
            filename = os.path.basename(path)

            # Full structured parse
            parsed = parse_resume(path)
            resume_text = parsed.get("text", "")
            if not resume_text or len(resume_text) < 100:
                continue

            # Deduplicate by text hash
            text_hash = hash(resume_text[:500])
            if text_hash in seen_hashes:
                continue
            seen_hashes.add(text_hash)

            # ML-based ATS scoring
            ats = calculate_ats_score(parsed, role)

            # Semantic similarity: resume embedding vs JD embedding
            similarity = 0.0
            if model_hub is not None and jd_embedding is not None:
                try:
                    resume_embedding = model_hub.embed_text(resume_text[:2000])
                    if resume_embedding is not None and np is not None:
                        similarity = float(np.dot(resume_embedding, jd_embedding))
                        similarity = max(0.0, min(1.0, similarity))
                except Exception:
                    pass

            # ── FIX 4: Normalize + filter matched/missing skills ──
            matched_skills = normalize_skills(ats.get("matched_skills", []))
            jd_skills_set = set(normalize_skills(ats.get("missing_skills", []) + matched_skills))
            matched_skills = [
                s for s in matched_skills
                if s in jd_skills_set
            ][:15]
            missing_skills = normalize_skills(ats.get("missing_skills", []))
            total_role_skills = len(matched_skills) + len(missing_skills)
            skill_match_ratio = len(matched_skills) / max(total_role_skills, 1)

            # ── Pre-agent combined score (for initial sorting) ──
            ml_ats_norm = ats["ats_score"] / 100.0
            pre_score = round(
                (similarity * 0.5 + ml_ats_norm * 0.3 + skill_match_ratio * 0.2) * 100, 1
            )

            # Extract candidate name
            candidate_name = (
                parsed.get("name", "")
                or filename.replace(".pdf", "").replace("_", " ").replace("-", " ").title()
            )

            # ── XGBoost score ──
            xgb_score = _compute_xgboost_score(parsed, ats, skill_match_ratio)

            results.append({
                "filename": filename,
                "name": candidate_name,
                "path": path,
                "resume_text": resume_text,
                "parsed_data": parsed,
                "skills": parsed.get("skills", []),
                "ats_score": ats["ats_score"],
                "ats_data": ats,
                "semantic_score": round(similarity * 100, 1),
                "semantic_raw": similarity,
                "skill_match_pct": round(skill_match_ratio * 100, 1),
                "skill_match_raw": skill_match_ratio,
                "xgboost_score": xgb_score,
                "xgboost_raw": (xgb_score or 50) / 100.0,
                "pre_score": pre_score,
                "combined_score": pre_score,
                "matched_skills": matched_skills,
                "missing_skills": missing_skills[:8],
                "confidence": ats.get("confidence", 50),
                "scoring_method": ats.get("scoring_method", "heuristic"),
                "feature_importance": ats.get("feature_importance", {}),
                # Agent fields (populated later)
                "agent_result": None,
                "agent_score": None,
                "agent_confidence": "pending",
                "agent_reason": "",
                "agent_evidence": [],
            })
        except Exception as e:
            logger.warning("Error processing %s: %s", path, e)
            continue

    if not results:
        return {"error": "Could not process any resumes"}

    _emit_progress("embedding", f"Processed {len(results)} resumes")

    # ── Pre-sort by ML score (for agent prioritization) ──
    results.sort(key=lambda x: x["pre_score"], reverse=True)

    _emit_progress("agents", f"Running multi-agent analysis on top {min(5, len(results))} candidates")

    # ── Run ReAct Agent on top candidates ──
    agent_enabled = False
    top_candidates = results[:5]

    if run_agent_batch is not None:
        try:
            agent_results = run_agent_batch(
                candidates=top_candidates,
                jd_text=jd_text,
                role=role,
                max_agent_calls=5
            )
            agent_enabled = True

            for i, agent_res in enumerate(agent_results):
                if agent_res is None:
                    continue

                # ── FIX 2/3/4/5: Normalize agent missing skills + strip noise ──
                agent_missing_raw = normalize_skills(agent_res.get("missing_skills", []))
                agent_missing_clean = []
                for skill in agent_missing_raw:
                    skill = skill.lower().strip()
                    skill = skill.replace("experience in", "")
                    skill = skill.replace("knowledge of", "")
                    skill = skill.replace("familiar with", "")
                    skill = skill.strip()
                    if skill:
                        agent_missing_clean.append(skill)

                resume_skills_norm = normalize_skills(results[i].get("matched_skills", []))
                jd_lower = jd_text.lower() if jd_text else ""
                valid_skills = []
                for skill in agent_missing_clean:
                    if skill not in resume_skills_norm and skill in jd_lower:
                        valid_skills.append(skill)
                agent_missing = valid_skills

                # ── FIX 6: Agent score cap if many missing ──
                agent_score = agent_res.get("agent_score", 5.0)
                if len(agent_missing) > 3:
                    agent_score = min(agent_score, 6)

                # ── FIX 7: Confidence from agent_score ──
                if agent_score >= 8:
                    agent_confidence = "high"
                elif agent_score >= 6:
                    agent_confidence = "medium"
                else:
                    agent_confidence = "low"

                results[i]["agent_result"] = agent_res
                results[i]["agent_score"] = round(agent_score, 1)
                results[i]["agent_confidence"] = agent_confidence
                results[i]["agent_reason"] = agent_res.get("reason", "")
                results[i]["agent_evidence"] = agent_res.get("evidence", [])
                results[i]["missing_skills"] = agent_missing[:8]

        except Exception as e:
            logger.error("Agent batch failed: %s", e)
            agent_enabled = False

    _emit_progress("ranking", "Computing final hybrid scores")

    # ── Compute final 5-signal hybrid score ──
    for r in results:
        semantic = r["semantic_raw"]
        ats_norm = r["ats_score"] / 100.0
        skill_match = r["skill_match_raw"]
        xgb_norm = r.get("xgboost_raw", 0.5)

        if r["agent_score"] is not None:
            agent_norm = r["agent_score"] / 10.0
            # 5-signal: 0.25 semantic + 0.20 ATS + 0.25 agent + 0.15 skill + 0.15 xgboost
            r["combined_score"] = round(
                (semantic * 0.25 + ats_norm * 0.20 + agent_norm * 0.25 +
                 skill_match * 0.15 + xgb_norm * 0.15) * 100, 1
            )
        else:
            # No agent: redistribute agent weight
            r["combined_score"] = round(
                (semantic * 0.35 + ats_norm * 0.25 + skill_match * 0.20 +
                 xgb_norm * 0.20) * 100, 1
            )

        r["combined_score"] = round(r["combined_score"], 1)

    # ── Final sort by hybrid score ──
    results.sort(key=lambda x: x["combined_score"], reverse=True)

    top_candidates = results[:top_n]

    # ── FIX 10: Sanity assertions ──
    for c in top_candidates:
        assert isinstance(c.get("matched_skills", []), list), "matched_skills must be a list"
        assert isinstance(c.get("missing_skills", []), list), "missing_skills must be a list"

    # ── Generate reasons & AI explanations for top candidates ──
    for i, candidate in enumerate(top_candidates):
        # Use agent reason if available, otherwise fallback to heuristic
        if candidate.get("agent_reason"):
            candidate["reason"] = candidate["agent_reason"]
        else:
            candidate["reason"] = _generate_reason(candidate, i + 1)

        # AI explanation (optional, via Gemini — only if agent didn't provide one)
        explanation = candidate.get("agent_reason", "")
        if not explanation and gemini_agent:
            try:
                explanation = gemini_agent.explain_top_candidate(
                    resume_text=candidate["resume_text"],
                    jd_text=jd_text,
                    role=role,
                    rank=i + 1,
                    ats_data=candidate["ats_data"]
                )
            except Exception:
                pass

        candidate["ai_explanation"] = explanation or candidate["reason"]

    _emit_progress("complete", f"Ranked {len(results)} candidates")

    # ── Build scoring details string ──
    xgb_active = any(r.get("xgboost_score") is not None for r in results)
    if agent_enabled:
        scoring_details = (
            "5-Signal Hybrid: 0.25 × Semantic + 0.20 × ATS + 0.25 × Multi-Agent + "
            "0.15 × Skill Match + 0.15 × XGBoost"
        )
    else:
        scoring_details = (
            "ML Scoring: 0.35 × Semantic + 0.25 × ATS + 0.20 × Skill + 0.20 × XGBoost"
        )

    return {
        "top_candidates": [
            {
                "rank": i + 1,
                "filename": c["filename"],
                "name": c["name"],
                "score": c["combined_score"],
                "ats_score": c["ats_score"],
                "semantic_score": c["semantic_score"],
                "skill_match_pct": c["skill_match_pct"],
                "combined_score": c["combined_score"],
                "matched_skills": c["matched_skills"],
                "missing_skills": c["missing_skills"],
                "reason": c["reason"],
                "ai_explanation": c["ai_explanation"],
                "confidence": c["confidence"],
                "scoring_method": c["scoring_method"],
                "feature_importance": c["feature_importance"],
                # ── Agent fields ──
                "agent_score": c.get("agent_score"),
                "agent_confidence": c.get("agent_confidence", "pending"),
                "agent_evidence": c.get("agent_evidence", []),
                "agent_reason": c.get("agent_reason", ""),
            }
            for i, c in enumerate(top_candidates)
        ],
        "all_results": [
            {
                "rank": i + 1,
                "filename": r["filename"],
                "name": r.get("name", r["filename"]),
                "ats_score": r["ats_score"],
                "combined_score": r["combined_score"],
                "confidence": r["confidence"],
                "agent_score": r.get("agent_score"),
            }
            for i, r in enumerate(results)
        ],
        "total_processed": len(results),
        "top_n": top_n,
        "scoring_details": scoring_details,
        "agent_enabled": agent_enabled,
    }
