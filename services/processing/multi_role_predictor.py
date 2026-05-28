"""
Multi-role predictor using zero-shot classification (BART-large-MNLI).
Falls back to rule-based logic if transformers/torch not available.
"""

import json
import os

from services.ml.skill_registry import ROLE_SKILL_MAP

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Try to load zero-shot classifier ──
_classifier = None
try:
    from transformers import pipeline as hf_pipeline
    _classifier = hf_pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1   # CPU
    )
except Exception:
    _classifier = None


# ── Role labels (15 roles from expanded job_roles.json) ──
def _get_role_labels():
    roles_path = os.path.join(PROJECT_ROOT, "data", "job_roles.json")
    try:
        with open(roles_path, "r", encoding="utf-8") as f:
            return list(json.load(f).keys())
    except Exception:
        return [
            "Web Developer", "Backend Developer", "Data Analyst",
            "Software Engineer", "ML Engineer"
        ]


# ── Backward-compatible alias ──
ROLE_SKILLS = ROLE_SKILL_MAP


def get_confidence_reason(role, matched_skills):
    matched = set(skill.lower() for skill in matched_skills)

    if role == "Web Developer" and matched.intersection({"html", "css", "javascript", "react"}):
        return "Strong frontend skill overlap"
    if role == "Backend Developer" and matched.intersection({"python", "flask", "api", "sql"}):
        return "Backend development skill alignment"
    if role == "ML Engineer" and matched.intersection({"python", "machine learning", "numpy", "pandas"}):
        return "Programming + Python dominance"
    if role == "Data Analyst" and matched.intersection({"python", "sql", "pandas", "statistics"}):
        return "Data analysis and statistics focus"
    if role == "DevOps Engineer" and matched.intersection({"docker", "kubernetes", "aws", "ci/cd"}):
        return "DevOps infrastructure alignment"
    if role == "Data Scientist" and matched.intersection({"python", "machine learning", "statistics"}):
        return "Data science skill alignment"
    if role == "Full Stack Developer" and matched.intersection({"javascript", "react", "node.js"}):
        return "Full stack technology coverage"
    if role == "Frontend Developer" and matched.intersection({"html", "css", "javascript", "react"}):
        return "Frontend development focus"
    if role == "Mobile Developer" and matched.intersection({"android", "ios", "flutter", "react native"}):
        return "Mobile development expertise"
    if role == "Cloud Engineer" and matched.intersection({"aws", "azure", "gcp", "docker"}):
        return "Cloud platform proficiency"
    if role == "Cybersecurity Analyst" and matched.intersection({"cybersecurity", "networking", "linux"}):
        return "Security domain alignment"
    if role == "Database Administrator" and matched.intersection({"sql", "postgresql", "mysql"}):
        return "Database management expertise"
    if role == "Product Manager" and matched.intersection({"agile", "scrum", "roadmap"}):
        return "Product management alignment"
    if role == "QA Engineer" and matched.intersection({"testing", "selenium", "automation"}):
        return "Quality assurance focus"
    if role == "Software Engineer":
        return "General engineering skill alignment"

    return "Relevant skill match"


def predict_multiple_roles(resume_skills):
    """
    Predict top 3 job roles for given skills.
    Uses BART zero-shot classification if available, else rule-based matching.
    Returns same format: list of dicts with role, score, matched_skills,
    missing_skills, reason.
    """
    if _classifier is not None:
        return _predict_zero_shot(resume_skills)
    else:
        return _predict_rule_based(resume_skills)


def _predict_zero_shot(resume_skills):
    """Zero-shot classification using facebook/bart-large-mnli."""
    try:
        candidate_labels = _get_role_labels()
        skills_text = ", ".join(resume_skills) if resume_skills else "general skills"

        result = _classifier(skills_text, candidate_labels, multi_label=False)

        scores = []
        for label, score_val in zip(result["labels"], result["scores"]):
            role_skills = ROLE_SKILLS.get(label, [])
            resume_lower = set(s.lower() for s in resume_skills)
            role_lower = set(s.lower() for s in role_skills)

            matched = list(resume_lower & role_lower)
            missing = list(role_lower - resume_lower)

            scores.append({
                "role": label,
                "score": int(score_val * 100),
                "matched_skills": matched,
                "missing_skills": missing,
                "reason": get_confidence_reason(label, matched) + f" (AI confidence: {score_val:.0%})"
            })

        return scores[:3]

    except Exception:
        return _predict_rule_based(resume_skills)


def _predict_rule_based(resume_skills):
    """Fallback: original rule-based matching."""
    scores = []
    resume_skills_lower = [skill.lower() for skill in resume_skills]

    for role, skills in ROLE_SKILLS.items():
        matched = set(resume_skills_lower).intersection(
            set(skill.lower() for skill in skills)
        )
        score = int((len(matched) / len(skills)) * 100) if skills else 0

        scores.append({
            "role": role,
            "score": score,
            "matched_skills": list(matched),
            "missing_skills": list(set(skills) - matched),
            "reason": get_confidence_reason(role, matched)
        })

    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores[:3]
