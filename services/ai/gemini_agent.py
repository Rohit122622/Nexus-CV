"""
Gemini AI Agent for Nexus CV — Multi-model Gemini with fallback chain.
Provides LLM-powered resume insights, suggestions, rewriting, comparison,
skill gap analysis, and candidate explanations.
Falls back gracefully to None if GEMINI_API_KEY is not set or any call fails.
"""

import os
import json

try:
    from services.ml.skill_registry import ROLE_SKILL_MAP, ROLE_SKILL_SCOPE
except ImportError:
    ROLE_SKILL_MAP = {}
    ROLE_SKILL_SCOPE = {}

from google import genai
import time
from utils.json_utils import safe_json_parse

# ── Gemini model priority (same as multi_llm.py) ──
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3-flash",
]

def call_gemini(prompt, retries=2):
    try:
        from google.genai import types
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        prompt = prompt + "\nReturn ONLY valid JSON."
        
        config = types.GenerateContentConfig(
            max_output_tokens=1024,
            temperature=0.2
        )
        
        last_error = None
        for model_name in GEMINI_MODELS:
            for attempt in range(retries + 1):
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=config
                    )

                    if hasattr(response, "text") and response.text:
                        print(f"[Gemini] {model_name} response received")
                        return response.text

                    if hasattr(response, "candidates") and response.candidates:
                        print(f"[Gemini] {model_name} candidate response received")
                        return response.candidates[0].content.parts[0].text
                    
                    raise Exception("Empty Gemini response")
                except Exception as e:
                    last_error = e
                    err_str = str(e)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        print(f"[Gemini] {model_name} rate-limited, trying next")
                        break
                    if "not found" in err_str.lower() or "invalid" in err_str.lower():
                        print(f"[Gemini] {model_name} not available, trying next")
                        break
                    if attempt < retries:
                        time.sleep(1.5)
                        continue
                    break

        raise last_error or Exception("All Gemini models exhausted")

    except Exception as e:
        print(f"[Gemini] Error: {type(e).__name__}")
        raise e

_model = True  # Mocking _model True so further script conditions still pass

# Try to import RAG store for context injection
try:
    from utils.rag_store import rag_store as _rag
except ImportError:
    _rag = None

# ── Role-specific skill scopes (prevents irrelevant suggestions) ──
ROLE_SKILL_SCOPE = {
    "Software Engineer": "data structures, algorithms, Python, Java, C++, Go, system design, Git, Docker, REST APIs, SQL, testing, CI/CD",
    "Web Developer": "HTML, CSS, JavaScript, TypeScript, React, Vue.js, Angular, Node.js, REST APIs, responsive design, Git, Webpack, accessibility",
    "Backend Developer": "Python, Java, Node.js, Go, SQL, PostgreSQL, MongoDB, Redis, Docker, REST APIs, microservices, message queues",
    "Frontend Developer": "HTML, CSS, JavaScript, TypeScript, React, Vue.js, Angular, Redux, Tailwind CSS, Webpack, Vite, testing, accessibility",
    "Full Stack Developer": "JavaScript, TypeScript, React, Node.js, Python, SQL, MongoDB, Docker, REST APIs, Git, authentication",
    "Data Scientist": "Python, R, pandas, NumPy, scikit-learn, TensorFlow, PyTorch, statistics, machine learning, deep learning, SQL, data visualization, Jupyter",
    "Data Analyst": "SQL, Python, R, Excel, Tableau, Power BI, pandas, statistics, data visualization, data cleaning, A/B testing",
    "ML Engineer": "Python, TensorFlow, PyTorch, scikit-learn, MLOps, Docker, Kubernetes, model serving, ONNX, MLflow, data pipelines, feature engineering",
    "DevOps Engineer": "Docker, Kubernetes, AWS, Azure, GCP, Terraform, CI/CD, Jenkins, GitHub Actions, Linux, Ansible, monitoring, Prometheus",
    "Cloud Engineer": "AWS, Azure, GCP, Terraform, CloudFormation, Docker, Kubernetes, serverless, IAM, networking, security",
    "Mobile Developer": "Android, iOS, Kotlin, Swift, React Native, Flutter, Dart, mobile UI/UX, REST APIs, Firebase, app deployment",
    "Cybersecurity Analyst": "cybersecurity, networking, Linux, Python, SIEM, penetration testing, vulnerability assessment, firewalls, IDS/IPS, compliance",
    "Database Administrator": "SQL, PostgreSQL, MySQL, Oracle, MongoDB, query optimization, indexing, replication, backup, performance tuning",
    "Product Manager": "Agile, Scrum, roadmap, stakeholder management, data analysis, user research, A/B testing, OKRs, Jira, communication",
    "QA Engineer": "testing, Selenium, Cypress, Playwright, API testing, CI/CD, test automation, performance testing, ISTQB, quality assurance"
}


def _call_gemini(prompt):
    """Internal helper: call Gemini and parse JSON response. Returns dict or None."""
    try:
        text_response = call_gemini(prompt)
        data = safe_json_parse(text_response)
        if isinstance(data, dict) and data:
            return data
        return None
    except Exception as e:
        print(f"[Gemini] Internal call error: {type(e).__name__}")
        return None


def _get_rag_context(role, resume_text=None):
    """Get RAG context for a given role. Returns string or empty."""
    if _rag is None:
        return ""
    try:
        context = _rag.get_context_for_role(role, resume_text, top_k=3)
        if context:
            return f"\n\nReference Examples (for context, do not copy directly):\n{context}\n"
        return ""
    except Exception:
        return ""


def _default_insights(role):
    return {
        "insights": {
            "resume_strength": "Moderate",
            "strength_reason": "Resume shows a moderate baseline but lacks explicitly quantified achievements and strictly role-specific keywords. Consider expanding your bullet points with measurable impact.",
            "word_count_assessment": "Good Length",
            "sections_found": ["Experience", "Skills"],
            "sections_missing": [],
            "skill_density": "Medium"
        },
        "suggestions": [
            f"Tailor your resume specifically for {role} positions.",
            "Add more quantified achievements to your experience section.",
            "Ensure you have covered all critical skills mapping strictly to the role.",
            "Remove generic phrasing like 'worked on' and use strong action verbs."
        ],
        "career_roadmap": []
    }

def get_resume_insights_and_suggestions(
    resume_text, ats_data, role, jd_text=None
):
    """
    Returns dict with keys:
      insights: dict (resume_strength, strength_reason, word_count_assessment,
                      sections_found, sections_missing, skill_density)
      suggestions: list of str (5-7 actionable suggestions)
      career_roadmap: list of dicts (month, focus, action, resource)
    Falls back to valid default dict if API key missing or call fails.
    """
    if not os.environ.get("GEMINI_API_KEY") or not _model:
        return _default_insights(role)

    jd_section = f"\nJob Description:\n{jd_text[:1500]}" if jd_text else ""
    rag_context = _get_rag_context(role, resume_text)

    # Get role-specific skill scope to prevent irrelevant suggestions
    role_scope = ROLE_SKILL_SCOPE.get(role, "")

    prompt = f"""You are an expert ATS resume coach specializing in {role} positions.

Resume Text (first 3000 chars):
{resume_text[:3000]}

Target Role: {role}
ATS Score: {ats_data.get('ats_score', 0)}/100
Scoring Method: {ats_data.get('scoring_method', 'unknown')}
Confidence: {ats_data.get('confidence', 0)}%
Matched Skills: {', '.join(ats_data.get('matched_skills', []))}
Missing Critical Skills: {', '.join(ats_data.get('missing_classified', {}).get('critical', []))}
{jd_section}{rag_context}

CRITICAL RULES:
1. ONLY suggest skills relevant to {role}. Allowed skill scope: {role_scope}
2. Do NOT suggest skills from unrelated domains (e.g., do NOT suggest Laravel for Data Scientist)
3. Every suggestion must be actionable and specific to {role}
4. Do NOT invent or hallucinate tools, frameworks, or certifications that don't exist
5. Career roadmap skills MUST come from the allowed skill scope above

Respond ONLY with a valid JSON object, no markdown, no explanation:
{{
  "insights": {{
    "resume_strength": "Strong|Moderate|Weak",
    "strength_reason": "one sentence explanation",
    "word_count_assessment": "Too Short|Good Length|Too Long",
    "sections_found": ["Education", "Experience", "Skills"],
    "sections_missing": ["Projects", "Certifications"],
    "skill_density": "High|Medium|Low"
  }},
  "suggestions": [
    "Specific actionable suggestion 1. Do not invent tools.",
    "Specific actionable suggestion 2. Validate facts."
  ],
  "career_roadmap": [
    {{"month": 1, "focus": "Must be a SINGLE standardized tool from the {role} domain", "action": "what to do", "resource": "course/platform"}},
    {{"month": 2, "focus": "Another strict tool from {role} scope", "action": "what to do", "resource": "course/platform"}},
    {{"month": 3, "focus": "skill name from allowed scope", "action": "what to do", "resource": "course/platform"}},
    {{"month": 4, "focus": "portfolio", "action": "what to build", "resource": "platform"}},
    {{"month": 5, "focus": "job search", "action": "what to do", "resource": "platform"}},
    {{"month": 6, "focus": "interview prep", "action": "what to do", "resource": "resource"}}
  ]
}}"""

    result = _call_gemini(prompt)
    if not result or not isinstance(result, dict):
        return _default_insights(role)

    # ── Post-LLM Filtering: Remove irrelevant roadmap skills ──
    if "career_roadmap" in result and isinstance(result["career_roadmap"], list):
        if hasattr(ROLE_SKILL_MAP, "get"):
            allowed = set(ROLE_SKILL_MAP.get(role, ROLE_SKILL_MAP.get("Software Engineer", [])))
        else:
            allowed = set()
            
        filtered_roadmap = []
        for item in result["career_roadmap"]:
            focus = str(item.get("focus", "")).lower()
            if focus in ["job search", "portfolio", "interview prep", "networking", "soft skills"]:
                filtered_roadmap.append(item)
                continue
            if allowed:
                if any(a in focus for a in allowed) or any(focus in a for a in allowed):
                    filtered_roadmap.append(item)
            else:
                filtered_roadmap.append(item)
        result["career_roadmap"] = filtered_roadmap

    return result


def rewrite_resume_content(
    experience_bullets, objective, target_role, skills, ats_score
):
    """
    Returns dict with keys:
      rewritten_bullets: list of str (same length as input)
      rewritten_objective: str
      skill_suggestions: list of str (up to 5 new skills to add)
    Falls back to None if API unavailable.
    """
    if not os.environ.get("GEMINI_API_KEY") or not _model:
        return None

    bullets_text = "\n".join([f"- {b}" for b in experience_bullets[:20]])
    rag_context = _get_rag_context(target_role)

    prompt = f"""You are an expert resume writer. Rewrite these resume bullet points to be
stronger, more impactful, and ATS-optimized for a {target_role} role.

Current ATS Score: {ats_score}/100
Target Role: {target_role}
Current Skills: {', '.join(skills[:20])}

Current Objective:
{objective}

Current Bullet Points:
{bullets_text}
{rag_context}

Rules:
- Start every bullet with a strong action verb (Led, Built, Developed, Engineered, etc.)
- Be specific and quantify where data exists in the original — do NOT invent percentages
  or numbers that aren't already there. If no numbers exist, use strong verbs only.
- Keep bullets concise: 1-2 lines max
- Optimize for ATS keywords relevant to {target_role}
- Do not change the meaning or fabricate experience

Respond ONLY with valid JSON, no markdown:
{{
  "rewritten_bullets": ["bullet 1", "bullet 2"],
  "rewritten_objective": "rewritten objective statement",
  "skill_suggestions": ["Only use strictly relevant tools to {target_role}. Do not invent tools. Max 10."]
}}"""

    try:
        result = _call_gemini(prompt)
        if not result or not isinstance(result, dict):
            return {
                "rewritten_bullets": experience_bullets,
                "rewritten_objective": objective,
                "skill_suggestions": []
            }

        if len(result.get("rewritten_bullets", [])) != len(experience_bullets):
            result["rewritten_bullets"] = experience_bullets

        # Post LLM Filtering on skill_suggestions
        if hasattr(ROLE_SKILL_MAP, "get"):
            allowed = set(ROLE_SKILL_MAP.get(target_role, ROLE_SKILL_MAP.get("Software Engineer", [])))
        else:
            allowed = set()

        suggestions = result.get("skill_suggestions", [])
        if allowed and isinstance(suggestions, list):
            result["skill_suggestions"] = list(dict.fromkeys(s for s in suggestions if isinstance(s, str) and s.lower() in allowed))[:15]

        return result
    except Exception as e:
        import traceback
        print(f"Gemini rewrite error: {e}\n{traceback.format_exc()}")
        return {
            "rewritten_bullets": experience_bullets,
            "rewritten_objective": objective,
            "skill_suggestions": []
        }


def _default_comparison():
    return {
        "verdict": "Both versions are roughly equal",
        "verdict_reason": "Both versions present similar baseline strengths, but further refinement with role-specific keywords and quantified achievements is strongly recommended.",
        "version_a_strengths": ["Original baseline."],
        "version_b_strengths": ["Updated baseline incorporates some changes."],
        "version_a_weaknesses": ["Lacks clearly quantified achievements."],
        "version_b_weaknesses": ["Still missing critical role ecosystem skills."],
        "recommendation": "Review ATS scores and skill overlap to determine the better version."
    }

def compare_resumes(
    resume_a_text, resume_b_text, ats_a, ats_b, role, jd_text=None
):
    """
    Returns dict with keys:
      verdict, verdict_reason, version_a_strengths, version_b_strengths,
      version_a_weaknesses, version_b_weaknesses, recommendation
    Falls back to default comparison dict if API unavailable.
    """
    if not os.environ.get("GEMINI_API_KEY") or not _model:
        return _default_comparison()

    jd_section = f"\nJob Description:\n{jd_text[:800]}" if jd_text else ""
    rag_context = _get_rag_context(role)

    prompt = f"""You are an expert recruiter comparing two versions of a resume for a {role} position.

VERSION A (ATS Score: {ats_a.get('ats_score', 0)}/100, Confidence: {ats_a.get('confidence', 0)}%):
{resume_a_text[:1500]}

VERSION B (ATS Score: {ats_b.get('ats_score', 0)}/100, Confidence: {ats_b.get('confidence', 0)}%):
{resume_b_text[:1500]}

Version A matched skills: {', '.join(ats_a.get('matched_skills', []))}
Version B matched skills: {', '.join(ats_b.get('matched_skills', []))}
{jd_section}{rag_context}

Respond ONLY with valid JSON, no markdown:
{{
  "verdict": "Version A is stronger|Version B is stronger|Both are roughly equal",
  "verdict_reason": "2-3 sentence explanation of why",
  "version_a_strengths": ["strength 1", "strength 2", "strength 3"],
  "version_b_strengths": ["strength 1", "strength 2", "strength 3"],
  "version_a_weaknesses": ["weakness 1", "weakness 2"],
  "version_b_weaknesses": ["weakness 1", "weakness 2"],
  "recommendation": "Specific advice on what to do next to improve the stronger version"
}}"""

    result = _call_gemini(prompt)
    if not result or not isinstance(result, dict):
        return _default_comparison()
    return result


def explain_top_candidate(
    resume_text, jd_text, role, rank, ats_data
):
    """Returns a 2-3 sentence explanation of why this candidate ranked in top 3."""
    if not os.environ.get("GEMINI_API_KEY") or not _model:
        return None

    prompt = f"""You are a senior recruiter. Explain in 2-3 sentences why this resume
is rank #{rank} match for the {role} position. Be specific about skills and experience.

Resume (first 1000 chars): {resume_text[:1000]}
Job Description (first 500 chars): {jd_text[:500]}
ATS Score: {ats_data.get('ats_score', 0)}/100
Scoring Method: {ats_data.get('scoring_method', 'unknown')}
Confidence: {ats_data.get('confidence', 0)}%
Matched Skills: {', '.join(ats_data.get('matched_skills', [])[:10])}

Respond ONLY with a valid JSON object matching EXACTLY this structure:
{{
  "insights": "2-3 sentence explanation of why this resume is rank #{rank} match.",
  "suggestions": [],
  "analysis": "Explanation reasoning",
  "score_reason": "ATS Evaluation"
}}"""

    try:
        from services.ai.multi_llm import call_llm
        raw = call_llm(prompt)
        if isinstance(raw, dict): return raw.get("insights", "Strong candidate.")
        return "Strong candidate."
    except Exception:
        return "Strong candidate."


def generate_skill_gap_analysis(skills, role, ats_data):
    """
    Analyze missing skills and predict their impact on ATS score.
    Returns list of dicts: [{"skill": str, "impact": int, "reason": str}]
    Falls back to None if API unavailable.
    """
    if not os.environ.get("GEMINI_API_KEY") or not _model:
        return []

    missing_critical = ats_data.get("missing_classified", {}).get("critical", [])
    missing_nice = ats_data.get("missing_classified", {}).get("nice_to_have", [])
    all_missing = missing_critical + missing_nice[:5]

    if not all_missing:
        return []

    rag_context = _get_rag_context(role)

    role_scope = ROLE_SKILL_SCOPE.get(role, "")

    prompt = f"""You are a career advisor analyzing skill gaps for a {role} position.

Current Skills: {', '.join(skills[:20])}
Missing Skills (ONLY analyze these): {', '.join(all_missing)}
Current ATS Score: {ats_data.get('ats_score', 0)}/100
{rag_context}

CRITICAL RULES:
1. ONLY include skills from the Missing Skills list above. Do NOT add skills that are not in that list.
2. All skills must be relevant to {role}. Allowed scope: {role_scope}
3. Do NOT suggest skills from unrelated domains.
4. Impact score must be between 1 and 15.

For each missing skill, estimate how much adding it would improve the ATS score
(1-15 points) and explain WHY it matters for {role}.

Respond ONLY with a valid JSON object matching EXACTLY this structure:
{{
  "insights": "Overall summary of missing skills",
  "suggestions": [
    "Skill Name|12|Why it matters for {role}",
    "Another Skill|8|Complementary skill"
  ],
  "analysis": "Analysis of the gaps",
  "score_reason": "Impact on ATS score"
}}"""

    try:
        from services.ai.multi_llm import call_llm
        raw = call_llm(prompt)
        if not raw or not isinstance(raw, dict): return []
        
        parsed_gaps = []
        for s in raw.get("suggestions", []):
            parts = str(s).split("|")
            if len(parts) >= 3:
                try:
                    parsed_gaps.append({
                        "skill": parts[0].strip(),
                        "impact": int(parts[1].strip()),
                        "reason": "|".join(parts[2:]).strip()
                    })
                except ValueError:
                    pass
        
        # Post-LLM Filter
        if hasattr(ROLE_SKILL_MAP, "get"):
            allowed = set(ROLE_SKILL_MAP.get(role, ROLE_SKILL_MAP.get("Software Engineer", [])))
        else:
            allowed = set()
            
        if allowed:
            return [r for r in parsed_gaps if str(r.get("skill", "")).lower() in allowed][:10]
        return parsed_gaps[:10]
    except Exception as e:
        import traceback
        print(f"Gemini skill gap error: {e}\n{traceback.format_exc()}")
        return []
