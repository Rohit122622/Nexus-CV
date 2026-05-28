"""
Resume suggestion generator with optional Gemini AI enhancement.
Falls back to template-based suggestions if Gemini unavailable.
"""


def generate_suggestions(insights, missing_skills, jd_result):
    """
    Generate actionable resume improvement suggestions.
    If Gemini AI provided suggestions via insights, use those.
    Otherwise, generate template-based suggestions.
    """
    # ── Check for AI suggestions from Gemini (passed via insights) ──
    ai_result = insights.get("_ai_full_result") if isinstance(insights, dict) else None
    if ai_result and isinstance(ai_result, dict):
        ai_suggestions = ai_result.get("suggestions", [])
        if ai_suggestions and isinstance(ai_suggestions, list) and len(ai_suggestions) >= 3:
            return ai_suggestions

    # ── Template-based suggestions (original logic) ──
    suggestions = []

    # Resume length
    if "Too Short" in insights.get("length", ""):
        suggestions.append("Add more project details and responsibilities to increase resume depth.")

    if "Too Long" in insights.get("length", ""):
        suggestions.append("Shorten descriptions and remove less relevant content.")

    # Missing sections
    for sec in insights.get("missing_sections", []):
        suggestions.append(f"Add a dedicated {sec} section to improve ATS score.")

    # Missing skills
    if isinstance(missing_skills, list):
        for skill in missing_skills[:3]:
            suggestions.append(f"Consider learning or mentioning {skill} if you have experience.")

    # JD match improvement
    if isinstance(jd_result, dict) and jd_result.get("match_percentage", 100) < 70:
        suggestions.append("Customize your resume keywords according to the job description.")

    if not suggestions:
        suggestions.append("Your resume is well optimized. Minor improvements can make it even stronger.")

    return suggestions
