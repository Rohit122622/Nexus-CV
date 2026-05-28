"""
Decision Agent — Final ReAct reasoning loop. STRICT 2 LLM iterations.

Synthesizes outputs from Skill, Experience, and ATS agents.
Uses Thought -> Action -> Observation pattern with scratchpad.

STRICT RULES:
  - Minimum 2 iterations enforced
  - Evidence MUST include 2+ real quotes from chunks
  - Confidence computed from signal strength (NOT random)
  - Scores normalized: each sub-agent contributes uniquely
  - NO generic reasoning allowed
"""

import re
import json
import logging

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 2


def _compute_confidence(sub_scores, final_score):
    """
    Compute confidence from signal strength — NOT random.

    HIGH:   strong skill match (>=7) AND strong experience (>=6)
    MEDIUM: mixed signals OR moderate scores
    LOW:    weak signals across the board
    """
    skill_score = sub_scores.get("skill", {}).get("skill_score", 0)
    exp_score = sub_scores.get("experience", {}).get("experience_score", 0)
    ats_score = sub_scores.get("ats", {}).get("ats_score", 0)
    matched_count = len(sub_scores.get("skill", {}).get("matched", []))
    years = sub_scores.get("experience", {}).get("years", 0)

    strong_skill = skill_score >= 7 and matched_count >= 4
    strong_exp = exp_score >= 6 and years >= 3
    strong_ats = ats_score >= 60

    if strong_skill and strong_exp:
        return "high"
    elif strong_skill or strong_exp or strong_ats:
        return "medium"
    else:
        # Check if at least some signals present
        if skill_score >= 4 and exp_score >= 4:
            return "medium"
        return "low"


def _extract_evidence_from_chunks(chunks, min_evidence=2):
    """Extract real quotes from resume chunks for evidence."""
    evidence = []
    patterns = [
        re.compile(r"(?:developed|built|created|designed|implemented|deployed|led|managed|optimized|automated|architected).*?(?:\.|$)", re.IGNORECASE),
        re.compile(r"(?:improved|increased|reduced|decreased|boosted|saved|achieved|delivered).*?\d+.*?(?:\.|$)", re.IGNORECASE),
        re.compile(r"\b(?:python|java|react|node|sql|aws|docker|kubernetes|tensorflow).*?(?:\.|$)", re.IGNORECASE),
        re.compile(r"(?:led|managed|supervised|mentored).*?(?:team|engineers|developers).*?(?:\.|$)", re.IGNORECASE),
    ]

    for chunk in (chunks or []):
        text = chunk.get("text", "")
        for pattern in patterns:
            for match in pattern.finditer(text):
                snippet = match.group(0).strip()
                if 20 < len(snippet) < 200 and snippet not in evidence:
                    evidence.append(snippet)
                    if len(evidence) >= 5:
                        return evidence

    return evidence


def _build_react_prompt(chunks_context, jd_text, role, sub_scores, scratchpad, iteration):
    """Build high-quality ReAct prompt for strong reasoning."""
    skill_data = sub_scores.get("skill", {})
    exp_data = sub_scores.get("experience", {})
    ats_data = sub_scores.get("ats", {})

    matched = skill_data.get("matched", [])
    missing = skill_data.get("missing", [])

    # Identify strongest and weakest signals
    scores = {
        "skill": skill_data.get("skill_score", 0),
        "experience": exp_data.get("experience_score", 0),
        "ats": ats_data.get("ats_score", 0) / 10.0  # normalize to 0-10
    }
    strongest = max(scores, key=scores.get)
    weakest = min(scores, key=scores.get)

    return f"""You are a senior technical hiring evaluator for {role}. This is iteration {iteration}/{MAX_ITERATIONS}.

AGENT ASSESSMENT SUMMARY:
- Skill Match: {skill_data.get('skill_score', 0)}/10 (matched: {', '.join(matched[:6])}) (missing: {', '.join(missing[:4])})
- Experience: {exp_data.get('experience_score', 0)}/10 ({exp_data.get('years', 0)} years, {exp_data.get('project_count', 0)} projects)
- ATS Score: {ats_data.get('ats_score', 0)}/100
- STRONGEST signal: {strongest} ({scores[strongest]:.1f}/10)
- WEAKEST signal: {weakest} ({scores[weakest]:.1f}/10)

RESUME EXCERPTS:
{chunks_context}

JOB REQUIREMENTS:
{jd_text[:400]}

{f'PREVIOUS ANALYSIS NOTES: {scratchpad}' if scratchpad else ''}

INSTRUCTIONS:
1. Compare ALL three sub-scores and identify alignment or conflicts
2. Focus on the STRONGEST and WEAKEST signals
3. MUST include at least 2 DIRECT QUOTES from the resume excerpts as evidence.
4. MUST justify your final score explicitly discussing: skills matched, years of experience, and ATS alignment.
5. If strong evidence is present that matches the JD, set confidence to "high".

{"Return your FINAL assessment as JSON:" if iteration >= 2 else "Perform initial analysis. Return JSON:"}
{{"thought":"<detailed reasoning comparing sub-scores>","action":"{"final_answer" if iteration >= 2 else "need_more_analysis"}","final_score":<0-100>,"confidence":"<high|medium|low>","reasoning":"<Explicit justification covering skills, experience, ATS>","evidence":["<direct quote 1 from resume>","<direct quote 2 from resume>"],"conflicts_found":"<any score conflicts>"{'"observation":"<what needs further checking>","focus_area":"<skills|experience|ats>"' if iteration < 2 else ''}}}"""


class DecisionAgent:
    """ReAct-based final decision agent with STRICT 2 iterations and evidence enforcement."""

    def evaluate(self, resume_chunks, jd_text, role, sub_scores):
        """
        Run ReAct loop to produce final candidate evaluation.

        ENFORCES:
          - Minimum 2 LLM iterations
          - At least 2 evidence quotes
          - Confidence computed from signal strength
          - Normalized, non-overlapping sub-scores

        Returns:
            dict with final_score, confidence, reasoning, evidence, thought_trace, sub_scores
        """
        from services.ai.multi_llm import call_llm
        import time

        # Build context from top chunks
        chunks_context = "\n---\n".join(
            f"[{c.get('section', 'Unknown')}] {c['text'][:400]}"
            for c in (resume_chunks or [])[:6]
        )

        scratchpad = ""
        thought_trace = []
        final_result = None

        for iteration in range(1, MAX_ITERATIONS + 1):
            prompt = _build_react_prompt(
                chunks_context, jd_text, role, sub_scores, scratchpad, iteration
            )

            # Retry once if LLM fails
            response = None
            for attempt in range(2):
                try:
                    response = call_llm(prompt)
                    if response:
                        break
                except Exception as e:
                    logger.warning("DecisionAgent: LLM try %d failed on iteration %d: %s", attempt, iteration, e)
                    time.sleep(1)

            if not response:
                logger.warning("DecisionAgent: LLM returned empty on iteration %d", iteration)
                continue

            # Parse JSON
            parsed = self._parse_response(response)
            if not parsed:
                logger.warning("DecisionAgent: Failed to parse JSON on iteration %d", iteration)
                continue

            action = parsed.get("action", "final_answer")
            # Force minimum 2 iterations by overriding final_answer on iteration 1
            if iteration == 1 and action == "final_answer":
                action = "need_more_analysis"

            thought = parsed.get("thought", "")
            thought_trace.append(f"Iteration {iteration}: {thought}")

            if action == "final_answer":
                final_result = parsed
                break
            elif action == "need_more_analysis":
                observation = parsed.get("observation", "")
                focus = parsed.get("focus_area", "general")
                scratchpad += f"\n[Iter {iteration}] Focus: {focus}. Observation: {observation}"
            else:
                final_result = parsed
                break

        # If no final result from LLM, use rule-based scoring
        if final_result is None:
            final_result = self._rule_based_scoring(sub_scores)
            logger.info("DecisionAgent: Using rule-based computation.")
            thought_trace.append("Rule-based: Computed from sub-agent scores")

        # Cross-validate score
        final_score = self._cross_validate(final_result, sub_scores)

        # ── Evidence enforcement: ensure at least 2 real quotes ──
        evidence = final_result.get("evidence", [])
        if isinstance(evidence, list):
            # Validate evidence against actual chunks
            validated = []
            combined_text = " ".join(c["text"] for c in (resume_chunks or []))
            for ev in evidence:
                if isinstance(ev, str) and len(ev) > 10:
                    ev_lower = ev.lower().strip().strip('"\'')
                    if ev_lower[:30] in combined_text.lower():
                        validated.append(ev)
            evidence = validated[:4]
        else:
            evidence = []

        # If insufficient evidence, extract directly from chunks
        if len(evidence) < 2:
            extracted = _extract_evidence_from_chunks(resume_chunks)
            for ev in extracted:
                if ev not in evidence:
                    evidence.append(ev)
                    if len(evidence) >= 2:
                        break

        # ── Smart confidence (NOT random) ──
        confidence = _compute_confidence(sub_scores, final_score)
        # Allow LLM to upgrade confidence if it returned high and signals support it
        llm_confidence = final_result.get("confidence", "medium")
        if llm_confidence == "high" and confidence != "low":
            confidence = "high"

        return {
            "score": round(final_score, 1),
            "reason": final_result.get("reasoning", "Evaluation completed."),
            "evidence": evidence,
            "final_score": round(final_score, 1),
            "confidence": confidence,
            "reasoning": final_result.get("reasoning", "Evaluation completed."),
            "thought_trace": thought_trace,
            "sub_scores": {
                "skill": sub_scores.get("skill", {}).get("skill_score", 0),
                "experience": sub_scores.get("experience", {}).get("experience_score", 0),
                "ats": sub_scores.get("ats", {}).get("ats_score", 0),
            },
            "conflicts_found": final_result.get("conflicts_found", "none"),
        }

    def _parse_response(self, response):
        """Extract JSON from LLM response (handles dict or string)."""
        if isinstance(response, dict):
            return response
        from utils.json_utils import safe_json_parse
        try:
            parsed = safe_json_parse(response)
            if isinstance(parsed, dict):
                return parsed
            return None
        except Exception:
            return None

    def _rule_based_scoring(self, sub_scores):
        """
        Compute score from sub-agents without LLM.
        Normalized: each signal contributes uniquely with non-overlapping weights.
        """
        skill_score = sub_scores.get("skill", {}).get("skill_score", 5) * 10  # 0-100
        exp_score = sub_scores.get("experience", {}).get("experience_score", 5) * 10  # 0-100
        ats_score = sub_scores.get("ats", {}).get("ats_score", 50)  # 0-100

        # Normalized weights: skill=35%, experience=35%, ATS=30%
        final = round(skill_score * 0.35 + exp_score * 0.35 + ats_score * 0.30, 1)

        # Compute evidence string from sub-scores
        matched = sub_scores.get("skill", {}).get("matched", [])
        years = sub_scores.get("experience", {}).get("years", 0)

        reasoning = (
            f"Candidate shows {len(matched)} skill matches "
            f"with {years} years of experience. "
            f"ATS compatibility score is {ats_score}/100."
        )

        confidence = _compute_confidence(sub_scores, final)

        return {
            "final_score": final,
            "confidence": confidence,
            "reasoning": reasoning,
            "evidence": [],
        }

    def _cross_validate(self, result, sub_scores):
        """Sanity-check LLM score against sub-agent scores."""
        llm_score = result.get("final_score", 50)
        try:
            llm_score = float(llm_score)
        except (ValueError, TypeError):
            llm_score = 50.0

        skill = sub_scores.get("skill", {}).get("skill_score", 5) * 10
        exp = sub_scores.get("experience", {}).get("experience_score", 5) * 10
        ats = sub_scores.get("ats", {}).get("ats_score", 50)
        ml_avg = (skill + exp + ats) / 3.0

        # If LLM score deviates more than 50 points from ML average, clamp it
        if abs(llm_score - ml_avg) > 50:
            logger.warning(
                "DecisionAgent cross-validation: LLM=%.1f vs ML_avg=%.1f -- clamping",
                llm_score, ml_avg
            )
            llm_score = ml_avg * 0.4 + llm_score * 0.6

        return max(0, min(100, llm_score))
