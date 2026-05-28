"""
Resume insights analyzer with optional Gemini AI enhancement.
Falls back to heuristic analysis if Gemini unavailable.
"""

try:
    import services.ai.gemini_agent as gemini_agent
except ImportError:
    gemini_agent = None


def analyze_resume_insights(resume_text, skills, ats_data=None, role=None, jd_text=None):
    """
    Analyze resume for insights. Tries Gemini AI first, falls back to heuristic.
    Accepts optional ats_data, role, jd_text for AI-enhanced analysis.
    Original signature (resume_text, skills) still works.
    """
    insights = {}

    # ── Heuristic analysis (always runs as baseline) ──

    # 1. Resume Length Analysis
    word_count = len(resume_text.split())

    if word_count < 150:
        insights["length"] = "Too Short – Resume lacks detail"
    elif 150 <= word_count <= 400:
        insights["length"] = "Good Length – Ideal for ATS"
    else:
        insights["length"] = "Too Long – Consider shortening"

    # 2. Section Presence Check
    sections = {
        "education": "Education",
        "experience": "Experience",
        "project": "Projects",
        "skill": "Skills"
    }

    missing_sections = []
    for key in sections:
        if key not in resume_text.lower():
            missing_sections.append(sections[key])

    insights["missing_sections"] = missing_sections

    # 3. Skill Density
    skill_count = len(skills)
    insights["skill_density"] = f"{skill_count} skills detected"

    # 4. Overall Resume Strength
    score = 0

    if word_count >= 150:
        score += 1
    if len(missing_sections) <= 1:
        score += 1
    if skill_count >= 5:
        score += 1

    if score == 3:
        insights["strength"] = "Strong Resume"
    elif score == 2:
        insights["strength"] = "Moderate Resume"
    else:
        insights["strength"] = "Weak Resume"

    # ── Gemini AI enhancement (optional overlay) ──
    if gemini_agent and ats_data and role:
        try:
            ai_result = gemini_agent.get_resume_insights_and_suggestions(
                resume_text, ats_data, role, jd_text
            )
            if ai_result and isinstance(ai_result, dict):
                ai_insights = ai_result.get("insights", {})
                if ai_insights:
                    # Merge AI insights without overwriting heuristic keys
                    insights["ai_strength"] = ai_insights.get("resume_strength", "")
                    insights["ai_strength_reason"] = ai_insights.get("strength_reason", "")
                    insights["ai_word_assessment"] = ai_insights.get("word_count_assessment", "")
                    insights["ai_skill_density"] = ai_insights.get("skill_density", "")
                    insights["ai_sections_missing"] = ai_insights.get("sections_missing", [])
                    insights["ai_enhanced"] = True

                # Store full AI result for use by suggestions and career recommender
                insights["_ai_full_result"] = ai_result
        except Exception:
            pass

    return insights
