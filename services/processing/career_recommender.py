import json
import os

try:
    import services.ai.gemini_agent as gemini_agent
except ImportError:
    gemini_agent = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_career_paths():
    path = os.path.join(PROJECT_ROOT, "data", "career_paths.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def recommend_career(ats_result, job_role, resume_skills=None, insights=None, jd_result=None):
    """
    Dynamic AI-driven career roadmap using ATS breakdown, missing critical
    skills, JD gaps, resume insights, and real data references.
    """
    career_paths = load_career_paths()

    if job_role not in career_paths:
        return {}

    path = career_paths[job_role]

    # ------- Existing base data (unchanged) -------
    recommendations = {
        "recommended_role": job_role,
        "skills_to_learn": path.get("skills_to_learn", []),
        "recommended_projects": path.get("projects", []),
        "roadmap": {}
    }

    # ------- Priority logic (unchanged) -------
    if ats_result["ats_score"] < 60:
        recommendations["priority"] = "Low – build fundamentals first"
    elif ats_result["ats_score"] < 80:
        recommendations["priority"] = "Medium – skill gaps to fill"
    else:
        recommendations["priority"] = "High – ready for applications"

    # ------- Normalize resume skills -------
    resume_skills = set(s.lower() for s in (resume_skills or []))

    # ------- Handle categorized skills -------
    skills_block = path.get("skills_to_learn", [])
    if isinstance(skills_block, dict):
        all_skills = (
            skills_block.get("core", []) +
            skills_block.get("performance", []) +
            skills_block.get("growth", [])
        )
    else:
        all_skills = skills_block

    all_skills = [s.lower() for s in all_skills]

    # ------- Find missing skills -------
    missing_skills = [s for s in all_skills if s not in resume_skills]

    # ------- Extract real data for dynamic roadmap -------
    missing_classified = ats_result.get("missing_classified", {})
    critical_missing = missing_classified.get("critical", [])

    ats_breakdown = {
        "skill_score": ats_result.get("skill_score", 0),
        "keyword_score": ats_result.get("keyword_score", 0),
        "completeness_score": ats_result.get("completeness_score", 0),
    }

    # Find the weakest ATS component
    breakdown_labels = {
        "skill_score": ("Skills Match", 50),
        "keyword_score": ("Keyword Coverage", 30),
        "completeness_score": ("Resume Completeness", 20),
    }
    weakest_area = min(
        breakdown_labels.keys(),
        key=lambda k: ats_breakdown[k] / breakdown_labels[k][1]
    )
    weakest_label = breakdown_labels[weakest_area][0]
    weakest_val = ats_breakdown[weakest_area]
    weakest_max = breakdown_labels[weakest_area][1]

    missing_sections = []
    if insights and isinstance(insights, dict):
        missing_sections = insights.get("missing_sections", [])

    jd_missing_keywords = []
    if jd_result and isinstance(jd_result, dict):
        jd_missing_keywords = jd_result.get("missing_keywords", [])

    # ============================================
    # 🔥 DYNAMIC AI-DRIVEN ROADMAP
    # ============================================
    month = 1

    # Month 1: Fix highest-impact missing critical skill
    if critical_missing:
        top_critical = critical_missing[0]
        recommendations["roadmap"][f"Month {month}"] = (
            f"Master {top_critical} — the highest-impact critical skill gap "
            f"for the {job_role} role. Build hands-on projects and earn certification."
        )
    elif missing_skills:
        recommendations["roadmap"][f"Month {month}"] = (
            f"Learn and practice {missing_skills[0]} — your top missing skill "
            f"for {job_role} readiness."
        )
    else:
        recommendations["roadmap"][f"Month {month}"] = (
            f"Deepen expertise in your strongest skills. Focus on advanced "
            f"techniques and real-world application for {job_role}."
        )
    month += 1

    # Month 2: Address ATS breakdown weakness
    recommendations["roadmap"][f"Month {month}"] = (
        f"Improve your {weakest_label} (currently {weakest_val}/{weakest_max}) — "
        f"{'add more role-specific skills to your resume' if weakest_area == 'skill_score' else 'increase keyword density with industry-standard terminology' if weakest_area == 'keyword_score' else 'add missing resume sections (contact info, objective, skills)'}"
        f" to boost your ATS score."
    )
    month += 1

    # Month 3: Fix resume structural gaps
    if missing_sections:
        sections_str = ", ".join(missing_sections[:3])
        recommendations["roadmap"][f"Month {month}"] = (
            f"Add missing resume sections: {sections_str}. "
            f"A complete resume structure significantly improves ATS parsing."
        )
    else:
        recommendations["roadmap"][f"Month {month}"] = (
            "Polish resume formatting and structure — your sections are complete. "
            "Focus on quantifying achievements and using action verbs."
        )
    month += 1

    # Month 4: JD gap remediation
    if jd_missing_keywords:
        kw_str = ", ".join(jd_missing_keywords[:4])
        recommendations["roadmap"][f"Month {month}"] = (
            f"Bridge job description gaps by learning: {kw_str}. "
            f"Tailor your resume for each application to maximize JD match."
        )
    else:
        recommendations["roadmap"][f"Month {month}"] = (
            "Your resume aligns well with job descriptions. "
            "Practice mock interviews and refine your portfolio presentation."
        )
    month += 1

    # Month 5+: Portfolio projects referencing actual skill gaps
    remaining_critical = critical_missing[1:3] if len(critical_missing) > 1 else []
    remaining_missing = [s for s in missing_skills if s not in [c.lower() for c in critical_missing]][:2]
    project_skills = remaining_critical + remaining_missing

    if project_skills:
        skills_str = ", ".join(project_skills[:3])
        recommendations["roadmap"][f"Month {month}"] = (
            f"Build portfolio projects demonstrating {skills_str}. "
            f"Showcase real-world problem-solving for {job_role} roles."
        )
    else:
        for project in path.get("projects", [])[:2]:
            recommendations["roadmap"][f"Month {month}"] = project
            month += 1

    # Final month: Application readiness
    month += 1
    if ats_result["ats_score"] >= 80:
        recommendations["roadmap"][f"Month {month}"] = (
            "Apply to target roles — your profile is strong. "
            "Focus on networking, interview prep, and continuous skill sharpening."
        )
    else:
        recommendations["roadmap"][f"Month {month}"] = (
            "Iterate on your resume, re-run ATS analysis, and begin applying "
            "to roles while continuing to close remaining skill gaps."
        )

    # ── Try Gemini AI roadmap overlay ──
    if insights and isinstance(insights, dict):
        ai_result = insights.get("_ai_full_result")
        if ai_result and isinstance(ai_result, dict):
            ai_roadmap = ai_result.get("career_roadmap", [])
            if ai_roadmap and isinstance(ai_roadmap, list) and len(ai_roadmap) >= 3:
                try:
                    ai_formatted = {}
                    for item in ai_roadmap:
                        m = item.get("month", 0)
                        focus = item.get("focus", "")
                        action = item.get("action", "")
                        resource = item.get("resource", "")
                        ai_formatted[f"Month {m}"] = (
                            f"{focus}: {action} (Resource: {resource})"
                        )
                    if ai_formatted:
                        recommendations["roadmap"] = ai_formatted
                        recommendations["ai_enhanced"] = True
                except Exception:
                    pass  # Keep rule-based roadmap

    return recommendations
