"""
JD Matcher with embedding-based semantic similarity + confidence scoring.
Uses model_hub for centralized embeddings (BGE-large with MiniLM fallback).
Falls back to keyword matching if embedding models unavailable.
"""

import re
import os

from utils.skill_normalizer import normalize as normalize_skills

# ── Model Hub ──
try:
    import services.ml.model_hub as model_hub
    import numpy as np
except ImportError:
    model_hub = None
    np = None

# ── spaCy for noun chunk extraction ──
try:
    import spacy
    try:
        _nlp = spacy.load("en_core_web_trf")
    except OSError:
        _nlp = spacy.load("en_core_web_sm")
except Exception:
    _nlp = None


_NON_SKILL_WORDS = {
    "engineer", "experience", "knowledge", "development", "software",
    "engineering", "skills", "tools", "systems", "platform", "management",
    "responsibility", "requirements", "qualifications", "team", "work",
    "ability", "understanding", "proficiency", "familiarity", "working",
    "strong", "excellent", "good", "basic", "advanced", "candidate",
    "position", "role", "company", "opportunity", "looking", "ideal",
    "required", "preferred", "minimum", "years", "degree", "bachelor"
}


def _extract_keywords_nlp(text):
    """Extract meaningful keywords using spaCy noun chunks + entities.
    Returns only clean 1-2 word technical keywords, no sentence fragments."""
    keywords = set()
    if _nlp is not None:
        doc = _nlp(text[:5000])
        for chunk in doc.noun_chunks:
            phrase = chunk.text.strip().lower()
            words = phrase.split()
            # Only keep 1-2 word technical phrases
            if len(words) <= 2 and len(phrase) > 2 and len(phrase) < 25 and not phrase.isdigit():
                if not any(w in _NON_SKILL_WORDS for w in words):
                    keywords.add(phrase)
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PRODUCT", "WORK_OF_ART", "LANGUAGE"):
                ent_text = ent.text.strip().lower()
                if len(ent_text.split()) <= 2 and len(ent_text) < 25:
                    keywords.add(ent_text)
    return list(keywords)


def _extract_keywords_basic(text):
    """Fallback: extract keywords using regex for technical terms."""
    words = re.findall(r'\b[a-zA-Z][a-zA-Z+#.]{1,30}\b', text.lower())
    stop = {"the", "and", "for", "with", "you", "are", "will", "this", "that",
            "from", "have", "has", "had", "our", "your", "they", "been", "being",
            "were", "was", "not", "but", "can", "all", "would", "should", "could",
            "more", "about", "which", "what", "when", "where", "who", "how", "also"}
    stop.update(_NON_SKILL_WORDS)
    return list(set(w for w in words if w not in stop and len(w) > 2))


def match_jd(resume_skills, jd_text):
    """
    Match resume skills against job description using semantic embedding similarity.
    Falls back to keyword matching if embeddings unavailable.

    Returns dict:
        match_percentage: int (0-100)
        matched_skills: list[str]
        missing_keywords: list[str]
        confidence: int (0-100)
        confidence_reason: str
        match_method: 'semantic' or 'keyword'
    """
    if not jd_text or not resume_skills:
        return {
            "match_percentage": 0,
            "matched_skills": [],
            "missing_keywords": [],
            "confidence": 20,
            "confidence_reason": "Insufficient data for matching",
            "match_method": "none"
        }

    # ── Extract JD keywords ──
    jd_keywords = _extract_keywords_nlp(jd_text)
    if not jd_keywords:
        jd_keywords = _extract_keywords_basic(jd_text)

    # Clean JD keywords: max 2 words, no non-skill words
    jd_keywords = [
        s.lower().strip()
        for s in jd_keywords
        if len(s.split()) <= 2
        and len(s) < 25
        and s.lower().strip() not in _NON_SKILL_WORDS
    ]

    # ── Try semantic matching ──
    if model_hub is not None and np is not None:
        try:
            resume_skill_list = list(set(s.lower() for s in resume_skills))
            resume_embeddings = model_hub.embed_batch(resume_skill_list)
            jd_embeddings = model_hub.embed_batch(jd_keywords[:50])

            if all(e is not None for e in resume_embeddings[:3]) and \
               all(e is not None for e in jd_embeddings[:3]):

                matched = set()
                missing = []

                for j, jd_emb in enumerate(jd_embeddings):
                    if jd_emb is None:
                        continue

                    best_sim = 0.0
                    for r, res_emb in enumerate(resume_embeddings):
                        if res_emb is None:
                            continue
                        sim = float(np.dot(jd_emb, res_emb))
                        if sim > best_sim:
                            best_sim = sim

                    if best_sim >= 0.65:
                        matched.add(jd_keywords[j])
                    elif best_sim < 0.55:
                        missing.append(jd_keywords[j])

                total = len(jd_keywords[:50])
                match_pct = int((len(matched) / max(total, 1)) * 100) if total > 0 else 0

                # ── Confidence: based on data quality ──
                has_enough_skills = len(resume_skills) >= 3
                has_enough_jd_kw = len(jd_keywords) >= 5
                confidence = 85 if (has_enough_skills and has_enough_jd_kw) else 55
                conf_reason = "High — sufficient skills and JD keywords for reliable matching" \
                    if confidence >= 80 else "Moderate — limited data for matching"

                return {
                    "match_percentage": match_pct,
                    "matched_skills": normalize_skills(sorted(matched)),
                    "missing_keywords": normalize_skills(missing[:15]),
                    "confidence": confidence,
                    "confidence_reason": conf_reason,
                    "match_method": "semantic"
                }
        except Exception:
            pass

    # ── Keyword fallback ──
    resume_set = set(s.lower() for s in resume_skills)
    jd_set = set(k.lower() for k in jd_keywords)
    matched = resume_set.intersection(jd_set)
    missing = jd_set.difference(resume_set)
    match_pct = int((len(matched) / max(len(jd_set), 1)) * 100)

    return {
        "match_percentage": match_pct,
        "matched_skills": normalize_skills(sorted(matched)),
        "missing_keywords": normalize_skills(sorted(missing)[:15]),
        "confidence": 45,
        "confidence_reason": "Low — using keyword matching only (no semantic analysis)",
        "match_method": "keyword"
    }
