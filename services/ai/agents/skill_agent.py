"""
Skill Agent — Rule-based skill extraction and matching. NO LLM calls.

Extracts skills from resume text, compares against JD requirements,
computes overlap score. Pure computation — fast and deterministic.
"""

import re
from services.ml.skill_registry import ROLE_SKILL_MAP, CRITICAL_SKILLS, NON_SKILL_WORDS
from utils.skill_normalizer import normalize as normalize_skills


class SkillAgent:
    """Rule-based skill extraction and matching agent."""

    def evaluate(self, resume_chunks, jd_text, role, parsed_skills=None):
        """
        Extract skills from resume chunks and match against JD + role requirements.

        Args:
            resume_chunks: list of chunk dicts from semantic_chunker
            jd_text: job description text
            role: target role name
            parsed_skills: pre-extracted skills from resume parser (optional)

        Returns:
            dict with skill_score, matched, missing, reason
        """
        # Use pre-parsed skills if available
        if parsed_skills:
            resume_skills = set(s.lower().strip() for s in parsed_skills)
        else:
            resume_skills = self._extract_skills_from_chunks(resume_chunks)

        # Get role-required skills
        role_skills = set(s.lower() for s in ROLE_SKILL_MAP.get(role, []))
        critical = set(s.lower() for s in CRITICAL_SKILLS.get(role, []))

        # Extract JD skills
        jd_skills = self._extract_skills_from_text(jd_text)

        # Combine role + JD required skills
        all_required = role_skills | jd_skills

        # Match
        matched = resume_skills & all_required
        missing = all_required - resume_skills
        critical_matched = resume_skills & critical
        critical_missing = critical - resume_skills

        # Score: weighted — critical skills worth more
        if not all_required:
            score = 5.0
        else:
            base_ratio = len(matched) / max(len(all_required), 1)
            critical_ratio = len(critical_matched) / max(len(critical), 1)
            score = round((base_ratio * 0.6 + critical_ratio * 0.4) * 10, 1)

        # Reason
        if score >= 8:
            reason = f"Strong skill match: {len(matched)}/{len(all_required)} required skills found, {len(critical_matched)}/{len(critical)} critical skills present"
        elif score >= 5:
            reason = f"Moderate skill match: {len(matched)}/{len(all_required)} skills. Missing critical: {', '.join(list(critical_missing)[:3])}"
        else:
            reason = f"Weak skill match: only {len(matched)}/{len(all_required)} skills found. Critical gaps: {', '.join(list(critical_missing)[:4])}"

        return {
            "skill_score": min(score, 10.0),
            "matched": normalize_skills(sorted(matched)),
            "missing": normalize_skills(sorted(missing)[:15]),
            "critical_matched": normalize_skills(sorted(critical_matched)),
            "critical_missing": normalize_skills(sorted(critical_missing)),
            "total_required": len(all_required),
            "reason": reason
        }

    def _extract_skills_from_chunks(self, chunks):
        """Extract skills from chunk texts using dictionary matching."""
        combined_text = " ".join(c["text"] for c in chunks)
        return self._extract_skills_from_text(combined_text)

    def _extract_skills_from_text(self, text):
        """Dictionary-based skill extraction from text."""
        from services.processing.resume_parser import _all_skills

        text_lower = text.lower()
        text_clean = re.sub(r'[^\w\s\-\+\#]', ' ', text_lower)
        tokens = text_clean.split()

        # Generate unigrams, bigrams, trigrams
        phrases = set(tokens)
        for i in range(len(tokens) - 1):
            phrases.add(f"{tokens[i]} {tokens[i+1]}")
        for i in range(len(tokens) - 2):
            phrases.add(f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}")

        found = set()
        for skill in _all_skills:
            skill_clean = skill.strip().lower()
            if skill_clean in NON_SKILL_WORDS or len(skill_clean) < 2:
                continue
            if skill_clean in phrases:
                found.add(skill_clean)

        # Clean: max 2 words, < 25 chars
        return set(
            s for s in found
            if len(s.split()) <= 2 and len(s) < 25
        )
