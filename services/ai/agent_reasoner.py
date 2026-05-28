"""
Multi-Agent Resume Reasoner — Orchestrates 4 specialized agents.

Pipeline: SkillAgent → ExperienceAgent → ATSAgent → DecisionAgent (ReAct loop)
Falls back to single-agent behavior if multi-agent pipeline fails.

Backward-compatible: run_agent() and run_agent_batch() signatures unchanged.

Output format (unchanged):
  {
    "agent_score": float (0-10),
    "confidence": "high" | "medium" | "low",
    "reason": str,
    "evidence": [str, ...],
    "missing_skills": [str, ...],
    "thought_trace": [str, ...]
  }
"""

import os
import re
import json
import logging
import time

from utils.skill_normalizer import normalize as normalize_skills

logger = logging.getLogger(__name__)

# ── LLM call ──
try:
    from services.ai.multi_llm import call_llm
except ImportError:
    call_llm = None

# ── RAG store for context retrieval ──
try:
    from utils.rag_store import rag_store as _rag
except ImportError:
    _rag = None

# ── Embeddings for chunk relevance ──
try:
    import services.ml.model_hub as model_hub
    import numpy as np
except ImportError:
    model_hub = None
    np = None

# ── Multi-Agent imports ──
try:
    from services.ai.agents.skill_agent import SkillAgent
    from services.ai.agents.experience_agent import ExperienceAgent
    from services.ai.agents.ats_agent import ATSAgent
    from services.ai.agents.decision_agent import DecisionAgent
    _MULTI_AGENT_AVAILABLE = True
except ImportError as e:
    logger.warning("Multi-agent system unavailable: %s", e)
    _MULTI_AGENT_AVAILABLE = False

# ── Semantic chunker ──
try:
    from services.processing.semantic_chunker import chunk_text, get_priority_chunks
    _SEMANTIC_CHUNKER = True
except ImportError:
    _SEMANTIC_CHUNKER = False

# ── Bias filter ──
try:
    from utils.bias_filter import anonymize, strip_name_from_text
    _BIAS_FILTER = True
except ImportError:
    _BIAS_FILTER = False


# ─────────────── EVIDENCE EXTRACTION (SHARED) ───────────────

_EVIDENCE_PATTERNS = [
    re.compile(r"(?:developed|built|created|designed|implemented|deployed|led|managed|optimized|automated|architected).*?(?:\.|$)", re.IGNORECASE),
    re.compile(r"(?:improved|increased|reduced|decreased|boosted|saved|achieved|delivered).*?\d+.*?(?:\.|$)", re.IGNORECASE),
    re.compile(r"\b(?:python|java|react|node|sql|aws|docker|kubernetes|tensorflow|pytorch)\b.*?(?:\.|$)", re.IGNORECASE),
]


def _extract_evidence(resume_text, jd_text=""):
    """Extract evidence snippets from resume using regex patterns."""
    evidence = []
    for pattern in _EVIDENCE_PATTERNS:
        for match in pattern.finditer(resume_text):
            snippet = match.group(0).strip()
            if 20 < len(snippet) < 200:
                evidence.append(snippet)
    return list(set(evidence))[:5]


# ─────────────── LEGACY CHUNKER (FALLBACK) ───────────────

def _chunk_resume_legacy(resume_text, chunk_size=500, overlap=100):
    """Legacy word-based chunker — used if semantic chunker unavailable."""
    if not resume_text:
        return []
    words = resume_text.split()
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        if len(chunk_words) < 20:
            continue
        chunks.append({
            "text": " ".join(chunk_words),
            "section": "Unknown",
            "char_count": len(" ".join(chunk_words)),
            "index": len(chunks)
        })
    return chunks


# ─────────────── MULTI-AGENT PIPELINE ───────────────

def _run_multi_agent(resume_text, jd_text, role, matched_skills=None,
                     ats_score=0, parsed_data=None, candidate_name=""):
    """
    Execute the 4-agent pipeline:
      1. SkillAgent (NO LLM)
      2. ExperienceAgent (1 LLM call max)
      3. ATSAgent (NO LLM)
      4. DecisionAgent (max 2 LLM iterations)

    Total LLM calls per candidate: ≤ 3
    """
    thought_trace = []
    t_total = time.time()

    # ── Apply bias filter ──
    anon_text = resume_text
    if _BIAS_FILTER and candidate_name:
        anon_text = anonymize(resume_text, candidate_id="Candidate")
        anon_text = strip_name_from_text(anon_text, candidate_name)
    elif _BIAS_FILTER:
        anon_text = anonymize(resume_text, candidate_id="Candidate")

    # ── Create chunks ──
    if _SEMANTIC_CHUNKER:
        chunks = chunk_text(anon_text)
        priority_chunks = get_priority_chunks(chunks, jd_text, top_k=8)
        thought_trace.append(f"Semantic chunker: {len(chunks)} chunks, {len(priority_chunks)} priority")
    else:
        chunks = _chunk_resume_legacy(anon_text)
        priority_chunks = chunks[:8]
        thought_trace.append(f"Legacy chunker: {len(chunks)} chunks")

    # ── Agent 1: Skill Agent (NO LLM) ──
    t0 = time.time()
    skill_agent = SkillAgent()
    skill_result = skill_agent.evaluate(
        resume_chunks=priority_chunks,
        jd_text=jd_text,
        role=role,
        parsed_skills=matched_skills
    )
    thought_trace.append(
        f"SkillAgent ({int((time.time()-t0)*1000)}ms): "
        f"score={skill_result['skill_score']}/10, "
        f"matched={len(skill_result['matched'])}, "
        f"missing={len(skill_result['missing'])}"
    )

    # ── Agent 2: Experience Agent (1 LLM call max) ──
    t0 = time.time()
    exp_agent = ExperienceAgent()
    exp_result = exp_agent.evaluate(
        resume_chunks=priority_chunks,
        resume_text=anon_text,
        jd_text=jd_text,
        role=role,
        parsed_data=parsed_data
    )
    thought_trace.append(
        f"ExperienceAgent ({int((time.time()-t0)*1000)}ms): "
        f"score={exp_result['experience_score']}/10, "
        f"years={exp_result['years']}, projects={exp_result['project_count']}, "
        f"llm={'yes' if exp_result.get('llm_used') else 'no'}"
    )

    # ── Agent 3: ATS Agent (NO LLM) ──
    t0 = time.time()
    ats_agent = ATSAgent()
    if parsed_data:
        ats_result_agent = ats_agent.evaluate(parsed_data, role)
    else:
        # Build minimal parsed_data from text
        ats_result_agent = {
            "ats_score": ats_score,
            "breakdown": {"skill_score": 0, "keyword_score": 0, "completeness_score": 0},
            "matched_skills": matched_skills or [],
            "missing_skills": [],
            "confidence": 50,
            "scoring_method": "pre-computed"
        }
    thought_trace.append(
        f"ATSAgent ({int((time.time()-t0)*1000)}ms): "
        f"score={ats_result_agent['ats_score']}/100, "
        f"method={ats_result_agent.get('scoring_method', 'unknown')}"
    )

    # ── Agent 4: Decision Agent (max 2 LLM iterations) ──
    t0 = time.time()
    decision_agent = DecisionAgent()
    sub_scores = {
        "skill": skill_result,
        "experience": exp_result,
        "ats": ats_result_agent,
    }
    decision_result = decision_agent.evaluate(
        resume_chunks=priority_chunks,
        jd_text=jd_text,
        role=role,
        sub_scores=sub_scores
    )
    thought_trace.extend(decision_result.get("thought_trace", []))
    thought_trace.append(
        f"DecisionAgent ({int((time.time()-t0)*1000)}ms): "
        f"final={decision_result['final_score']}, "
        f"confidence={decision_result['confidence']}"
    )

    total_ms = int((time.time() - t_total) * 1000)
    thought_trace.append(f"Total multi-agent pipeline: {total_ms}ms")

    # ── Convert to backward-compatible format ──
    # agent_score is 0-10, final_score from decision is 0-100
    agent_score_10 = round(decision_result["final_score"] / 10.0, 1)
    agent_score_10 = max(0, min(10, agent_score_10))

    # Combine missing skills
    all_missing = list(set(
        skill_result.get("missing", []) +
        ats_result_agent.get("missing_skills", [])
    ))

    return {
        "agent_score": agent_score_10,
        "confidence": decision_result["confidence"],
        "reason": decision_result["reasoning"],
        "evidence": decision_result["evidence"],
        "missing_skills": normalize_skills(all_missing[:8]),
        "thought_trace": thought_trace,
        # ── Extended fields (new) ──
        "multi_agent": True,
        "sub_scores": decision_result.get("sub_scores", {}),
        "skill_agent_score": skill_result["skill_score"],
        "experience_agent_score": exp_result["experience_score"],
        "ats_agent_score": ats_result_agent["ats_score"],
        "decision_score": decision_result["final_score"],
        "bias_filtered": _BIAS_FILTER,
    }


# ─────────────── LEGACY SINGLE-AGENT (FALLBACK) ───────────────

def _run_single_agent(resume_text, jd_text, role, matched_skills=None,
                      ats_score=0, parsed_data=None):
    """
    Original single-agent ReAct loop (used as fallback if multi-agent fails).
    """
    fallback = {
        "agent_score": 5.0,
        "confidence": "low",
        "reason": "Unable to perform deep agent analysis.",
        "evidence": [],
        "missing_skills": [],
        "thought_trace": ["Single-agent fallback mode"],
        "multi_agent": False,
    }

    if call_llm is None:
        return fallback

    try:
        # Chunk resume
        if _SEMANTIC_CHUNKER:
            chunks = chunk_text(resume_text)
            priority = get_priority_chunks(chunks, jd_text, top_k=5)
        else:
            chunks = _chunk_resume_legacy(resume_text)
            priority = chunks[:5]

        chunk_context = "\n---\n".join(
            c["text"][:300] for c in priority
        )

        # RAG context
        rag_context = ""
        if _rag and _rag.index_size() > 0:
            rag_results = _rag.query(f"{role} {jd_text[:200]}", top_k=2)
            if rag_results:
                rag_context = "\n".join(r["text"][:200] for r in rag_results)

        prompt = f"""You are a ReAct hiring evaluator for: {role}

RESUME CHUNKS:
{chunk_context}

JOB DESCRIPTION:
{jd_text[:500]}

{f'CONTEXT: {rag_context}' if rag_context else ''}

KNOWN DATA:
- ATS score: {ats_score}/100
- Matched skills: {', '.join((matched_skills or [])[:10])}

Evaluate this candidate. Return ONLY valid JSON:
{{
    "action": "final_answer",
    "score": <float 0-10>,
    "confidence": "<high|medium|low>",
    "reason": "<2-3 sentence assessment>",
    "evidence": ["<quote from resume>"],
    "missing_skills": ["<skill1>", "<skill2>"]
}}"""

        response = call_llm(prompt)
        if response:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                parsed = json.loads(json_match.group())
                score = float(parsed.get("score", 5.0))
                score = max(0, min(10, score))

                # Cross-validate with ATS
                ats_normalized = ats_score / 10.0
                if abs(score - ats_normalized) > 3:
                    score = score * 0.6 + ats_normalized * 0.4

                return {
                    "agent_score": round(score, 1),
                    "confidence": parsed.get("confidence", "medium"),
                    "reason": parsed.get("reason", "Evaluation completed."),
                    "evidence": parsed.get("evidence", [])[:4],
                    "missing_skills": normalize_skills(parsed.get("missing_skills", [])[:8]),
                    "thought_trace": ["Single-agent ReAct (1 iteration)"],
                    "multi_agent": False,
                }

    except Exception as e:
        logger.error("Single-agent fallback error: %s", e)

    fallback["evidence"] = _extract_evidence(resume_text, jd_text)
    return fallback


# ─────────────── PUBLIC API (BACKWARD COMPATIBLE) ───────────────

def run_agent(resume_text, jd_text, role="Software Engineer",
              matched_skills=None, ats_score=0, parsed_data=None,
              candidate_name=""):
    """
    Run agentic evaluation on a single candidate.

    Tries multi-agent pipeline first, falls back to single-agent if it fails.

    Args:
        resume_text: Full resume text
        jd_text: Job description
        role: Target role
        matched_skills: Pre-matched skill list
        ats_score: Pre-computed ATS score
        parsed_data: Full parsed resume dict (optional)
        candidate_name: Name for bias filtering (optional)

    Returns:
        dict with agent_score, confidence, reason, evidence, missing_skills, thought_trace
    """
    if _MULTI_AGENT_AVAILABLE and call_llm is not None:
        try:
            result = _run_multi_agent(
                resume_text=resume_text,
                jd_text=jd_text,
                role=role,
                matched_skills=matched_skills,
                ats_score=ats_score,
                parsed_data=parsed_data,
                candidate_name=candidate_name
            )
            if result and result.get("agent_score", 0) > 0:
                logger.info("Multi-agent pipeline succeeded: score=%.1f", result["agent_score"])
                return result
        except Exception as e:
            logger.warning("Multi-agent pipeline failed, falling back: %s", e)

    # Fallback to single-agent
    return _run_single_agent(
        resume_text=resume_text,
        jd_text=jd_text,
        role=role,
        matched_skills=matched_skills,
        ats_score=ats_score,
        parsed_data=parsed_data
    )


def run_agent_batch(candidates, jd_text, role="Software Engineer", max_agent_calls=5):
    """
    Run the agent on top candidates (to save LLM calls).
    Candidates should already be pre-sorted by ML score.

    Args:
        candidates: list of dicts with resume_text, matched_skills, ats_score
        jd_text: Job description
        role: Target role
        max_agent_calls: Max number of candidates to evaluate with agents

    Returns:
        list of agent results (same order as input candidates)
    """
    results = []

    for i, candidate in enumerate(candidates[:max_agent_calls]):
        try:
            agent_result = run_agent(
                resume_text=candidate.get("resume_text", ""),
                jd_text=jd_text,
                role=role,
                matched_skills=candidate.get("matched_skills", []),
                ats_score=candidate.get("ats_score", 0),
                parsed_data=candidate.get("parsed_data"),
                candidate_name=candidate.get("name", "")
            )
            results.append(agent_result)
            logger.info(
                "Agent batch [%d/%d]: score=%.1f, multi_agent=%s",
                i + 1, min(len(candidates), max_agent_calls),
                agent_result.get("agent_score", 0),
                agent_result.get("multi_agent", False)
            )
        except Exception as e:
            logger.warning("Agent batch item %d failed: %s", i, e)
            results.append({
                "agent_score": 5.0,
                "confidence": "low",
                "reason": "Agent evaluation incomplete.",
                "evidence": [],
                "missing_skills": [],
                "thought_trace": [f"Error: {str(e)[:100]}"],
                "multi_agent": False,
            })

    # Pad remaining candidates with None (they don't get agent evaluation)
    while len(results) < len(candidates):
        results.append(None)

    return results
