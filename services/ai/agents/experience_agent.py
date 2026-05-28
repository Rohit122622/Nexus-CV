"""
Experience Agent — Evaluates candidate experience depth. Max 1 LLM call.

Extracts years of experience, project count, impact metrics using
regex patterns first, then uses 1 LLM call for qualitative assessment.
"""

import re
import logging

logger = logging.getLogger(__name__)


# Impact patterns for evidence extraction
_IMPACT_PATTERNS = [
    re.compile(r"(?:increased|improved|boosted|grew|enhanced).*?(?:\d+%|\d+x)", re.IGNORECASE),
    re.compile(r"(?:reduced|decreased|cut|eliminated|saved).*?(?:\d+%|\d+x|\$[\d,]+)", re.IGNORECASE),
    re.compile(r"(?:led|managed|supervised|mentored).*?(?:team|\d+\s*(?:people|engineers|developers))", re.IGNORECASE),
    re.compile(r"(?:built|developed|created|launched|shipped).*?(?:product|platform|system|service|application)", re.IGNORECASE),
    re.compile(r"\d+[+]?\s*(?:years?|months?)\s*(?:of\s+)?(?:experience|expertise)", re.IGNORECASE),
]

# Date patterns
_YEAR_RANGE = re.compile(r"(\d{4})\s*[-–—to]+\s*(\d{4}|present|current|now)", re.IGNORECASE)
_PROJECT_INDICATORS = re.compile(
    r"(?:project|built|developed|created|launched|designed|implemented|architected)",
    re.IGNORECASE
)


class ExperienceAgent:
    """Experience evaluation agent with optional single LLM call."""

    def evaluate(self, resume_chunks, resume_text, jd_text, role, parsed_data=None):
        """
        Evaluate experience depth from resume.

        Args:
            resume_chunks: list of chunk dicts from semantic_chunker
            resume_text: full resume text
            jd_text: job description text
            role: target role name
            parsed_data: pre-parsed resume data (optional)

        Returns:
            dict with experience_score, years, project_count, impact_evidence, reason
        """
        # ── STEP 1: Regex-based extraction (no LLM) ──
        years = self._estimate_years(resume_text, parsed_data)
        project_count = self._count_projects(resume_text, parsed_data)
        impact_evidence = self._extract_impact(resume_text)
        leadership = self._detect_leadership(resume_text)

        # ── STEP 2: Rule-based scoring ──
        base_score = 0.0

        # Years contribution (0-3 points)
        if years >= 5:
            base_score += 3.0
        elif years >= 3:
            base_score += 2.0
        elif years >= 1:
            base_score += 1.0

        # Project count contribution (0-2.5 points)
        if project_count >= 4:
            base_score += 2.5
        elif project_count >= 2:
            base_score += 1.5
        elif project_count >= 1:
            base_score += 0.5

        # Impact evidence (0-2.5 points)
        impact_count = len(impact_evidence)
        if impact_count >= 3:
            base_score += 2.5
        elif impact_count >= 1:
            base_score += 1.5

        # Leadership bonus (0-2 points)
        if leadership:
            base_score += 2.0

        base_score = min(base_score, 10.0)

        # ── STEP 3: Single LLM call for qualitative assessment ──
        llm_assessment = None
        try:
            llm_assessment = self._llm_assess(resume_chunks, jd_text, role, years, project_count)
            if llm_assessment and isinstance(llm_assessment, dict):
                llm_score = llm_assessment.get("score", base_score)
                # Blend: 60% rule-based + 40% LLM
                final_score = round(base_score * 0.6 + float(llm_score) * 0.4, 1)
            else:
                final_score = base_score
        except Exception as e:
            logger.warning("ExperienceAgent LLM call failed: %s", e)
            final_score = base_score

        final_score = min(max(final_score, 0), 10.0)

        # Build reason
        parts = []
        if years > 0:
            parts.append(f"{years} years experience")
        if project_count > 0:
            parts.append(f"{project_count} projects")
        if impact_count > 0:
            parts.append(f"{impact_count} quantified impacts")
        if leadership:
            parts.append("leadership evidence found")
        reason = ", ".join(parts) if parts else "Limited experience data found"

        return {
            "score": round(final_score, 1),
            "reason": reason,
            "evidence": impact_evidence[:5],
            "experience_score": round(final_score, 1),
            "years": years,
            "project_count": project_count,
            "impact_evidence": impact_evidence[:5],
            "leadership": leadership,
            "llm_used": llm_assessment is not None
        }

    def _estimate_years(self, text, parsed_data=None):
        """Estimate years of experience."""
        if parsed_data and parsed_data.get("experience_years", 0) > 0:
            return parsed_data["experience_years"]

        total_months = 0
        for match in _YEAR_RANGE.finditer(text):
            start = int(match.group(1))
            end_str = match.group(2).lower()
            if end_str in ("present", "current", "now"):
                from datetime import datetime
                end = datetime.now().year
            else:
                try:
                    end = int(end_str)
                except ValueError:
                    continue
            if 1990 <= start <= 2030 and 1990 <= end <= 2030:
                total_months += max(0, (end - start) * 12)

        return round(min(total_months / 12.0, 30.0), 1)

    def _count_projects(self, text, parsed_data=None):
        """Count project mentions."""
        if parsed_data and parsed_data.get("sections_found"):
            if "Projects" in parsed_data["sections_found"]:
                # Count bullet points in project section
                lines = text.split("\n")
                in_project = False
                count = 0
                for line in lines:
                    stripped = line.strip()
                    if re.search(r"(?i)\b(projects?)\b", stripped) and len(stripped) < 30:
                        in_project = True
                        continue
                    if in_project and re.match(r"(?i)^[\s]*(education|experience|skills|certification)", stripped):
                        break
                    if in_project and stripped and not stripped.startswith(("•", "-", "–")):
                        count += 1
                return max(count, 1)

        # Fallback: count project indicators
        matches = _PROJECT_INDICATORS.findall(text)
        return min(len(set(matches)), 10)

    def _extract_impact(self, text):
        """Extract quantified impact statements."""
        evidence = []
        for pattern in _IMPACT_PATTERNS:
            for match in pattern.finditer(text):
                snippet = match.group(0).strip()
                if len(snippet) > 15 and len(snippet) < 200:
                    evidence.append(snippet)
        return list(set(evidence))[:5]

    def _detect_leadership(self, text):
        """Detect leadership/management evidence."""
        leadership_patterns = [
            r"(?i)\b(?:led|managed|supervised|mentored|headed|directed)\b.*\b(?:team|group|department)\b",
            r"(?i)\b(?:senior|lead|principal|staff|manager|director|head)\b",
        ]
        for pattern in leadership_patterns:
            if re.search(pattern, text):
                return True
        return False

    def _llm_assess(self, chunks, jd_text, role, years, project_count):
        """Single LLM call for qualitative experience assessment."""
        from services.ai.multi_llm import call_llm
        import time

        # Use only Experience-tagged chunks
        exp_chunks = [c for c in chunks if c.get("section") in ("Experience", "Projects")]
        if not exp_chunks:
            exp_chunks = chunks[:3]

        context = "\n---\n".join(c["text"][:250] for c in exp_chunks[:3])

        prompt = f"""Evaluate candidate experience for {role}.

EXPERIENCE:
{context}

JD: {jd_text[:400]}

Facts: {years}yr exp, {project_count} projects.
Score 0-10 for relevance, depth, impact.

Return ONLY JSON: {{"score": <float>, "assessment": "<1 sentence>"}}"""

        for attempt in range(2):
            try:
                response = call_llm(prompt)
                if isinstance(response, dict) and "score" in response:
                    return response
                # call_llm already returns a dict; if it doesn't have "score",
                # check if nested or raw
                if isinstance(response, dict):
                    return response
            except Exception as e:
                logger.warning("ExperienceAgent LLM attempt %d: %s", attempt, e)
            time.sleep(1)

        return None
