"""
AI Resume Builder module for Nexus CV.
Handles form validation, resume data formatting, ATS refinement, and PDF generation.
"""

import re
import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from utils.pdf_utils import (
    PAGE_WIDTH, PAGE_HEIGHT, MARGIN_LEFT, MARGIN_RIGHT, MARGIN_TOP, MARGIN_BOTTOM,
    CONTENT_WIDTH, FONT_REGULAR, FONT_BOLD,
    LINE_HEIGHT_NORMAL, LINE_HEIGHT_HEADING, SECTION_GAP,
    draw_section_heading, draw_text_line, draw_bullet_list, safe_new_page
)

# ────────────── ACTION VERBS ──────────────

ACTION_VERBS = [
    "Developed", "Implemented", "Designed", "Built", "Led", "Managed",
    "Optimized", "Automated", "Architected", "Deployed", "Integrated",
    "Created", "Engineered", "Streamlined", "Delivered", "Spearheaded",
    "Established", "Transformed", "Reduced", "Improved", "Launched"
]


# ────────────── SANITIZATION ──────────────

def sanitize_input(text):
    """Strip HTML tags and dangerous characters from user input."""
    if not text:
        return ""
    text = re.sub(r"<.*?>", "", str(text))
    text = text.strip()
    return text


# ────────────── FORM VALIDATION ──────────────

def validate_form_data(form):
    """
    Server-side validation of resume builder form (strict mode).
    Returns (is_valid: bool, error_message: str)
    """
    required_fields = {
        "full_name": "Full Name",
        "email": "Email",
        "phone": "Phone",
        "career_objective": "Career Objective",
    }

    for field, label in required_fields.items():
        value = form.get(field, "").strip()
        if not value:
            return False, f"{label} is required."

    # ── Career objective minimum length ──
    objective = form.get("career_objective", "").strip()
    if len(objective) < 30:
        return False, "Career Objective must be at least 30 characters long."

    # ── Skills check (minimum 3) ──
    skills = form.get("skills", "").strip()
    if not skills:
        return False, "At least one skill is required."
    skills_list = [s.strip() for s in skills.split(",") if s.strip()]
    if len(skills_list) < 3:
        return False, "At least 3 skills are required for a competitive resume."

    # ── Education check ──
    edu_degrees = form.getlist("edu_degree[]")
    has_education = any(d.strip() for d in edu_degrees)
    if not has_education:
        return False, "At least one education entry is required."

    # ── Experience OR Project check ──
    exp_titles = form.getlist("exp_title[]")
    proj_names = form.getlist("proj_name[]")
    has_experience = any(t.strip() for t in exp_titles)
    has_projects = any(n.strip() for n in proj_names)

    if not has_experience and not has_projects:
        return False, "At least one Work Experience or Project is required."

    # ── Empty bullet point check ──
    exp_descriptions = form.getlist("exp_description[]")
    proj_descriptions = form.getlist("proj_description[]")
    all_bullets = exp_descriptions + proj_descriptions

    for i, desc in enumerate(all_bullets):
        if desc is not None:
            lines = [line.strip() for line in desc.strip().split("\n") if line.strip() != ""]
            for line in lines:
                cleaned = line.lstrip("•-▪▸◦★✦✓✔➤► ").strip()
                if not cleaned:
                    return False, "Empty bullet points are not allowed. Please remove or fill them."

    # ── Duplicate sentence check ──
    all_text_fields = [objective] + list(exp_descriptions) + list(proj_descriptions)
    seen_sentences = set()
    for text in all_text_fields:
        if not text:
            continue
        sentences = [s.strip().lower() for s in text.strip().split("\n") if s.strip()]
        for sentence in sentences:
            cleaned = sentence.lstrip("•-▪▸◦★✦✓✔➤► ").strip()
            if len(cleaned) > 15:  # Only check substantial sentences
                if cleaned in seen_sentences:
                    return False, f"Duplicate sentence detected: \"{sentence[:60]}...\". Each description must be unique."
                seen_sentences.add(cleaned)

    return True, ""



# ────────────── DATA FORMATTING ──────────────

def _enhance_bullet(text, index=0):
    """Ensure a bullet point starts with an action verb if it doesn't already."""
    text = text.strip()
    if not text:
        return text
    # Check if it already starts with an action verb
    first_word = text.split()[0] if text.split() else ""
    for verb in ACTION_VERBS:
        if first_word.lower() == verb.lower():
            return text
    # Prepend an action verb
    verb = ACTION_VERBS[index % len(ACTION_VERBS)]
    return f"{verb} {text[0].lower()}{text[1:]}" if len(text) > 1 else f"{verb} {text}"


def format_resume_data(form):
    """
    Convert raw form data into structured resume dict.
    Returns dict with all sections + full_text for ATS scoring + skills_list.
    """
    data = {}

    # ── Personal Info ──
    data["name"] = sanitize_input(form.get("full_name", ""))
    data["email"] = sanitize_input(form.get("email", ""))
    data["phone"] = sanitize_input(form.get("phone", ""))
    data["linkedin"] = sanitize_input(form.get("linkedin", ""))
    data["github"] = sanitize_input(form.get("github", ""))
    data["portfolio"] = sanitize_input(form.get("portfolio", ""))

    # ── Career Objective ──
    data["career_objective"] = sanitize_input(form.get("career_objective", ""))

    # ── Target Role ──
    data["target_role"] = sanitize_input(form.get("target_role", ""))

    # ── Education ──
    edu_degrees = form.getlist("edu_degree[]")
    edu_institutions = form.getlist("edu_institution[]")
    edu_years = form.getlist("edu_year[]")

    data["education"] = []
    for i in range(len(edu_degrees)):
        degree = sanitize_input(edu_degrees[i]) if i < len(edu_degrees) else ""
        institution = sanitize_input(edu_institutions[i]) if i < len(edu_institutions) else ""
        year = sanitize_input(edu_years[i]) if i < len(edu_years) else ""
        if degree:
            data["education"].append({
                "degree": degree,
                "institution": institution,
                "year": year
            })

    # ── Work Experience ──
    exp_titles = form.getlist("exp_title[]")
    exp_companies = form.getlist("exp_company[]")
    exp_durations = form.getlist("exp_duration[]")
    exp_descriptions = form.getlist("exp_description[]")

    data["experience"] = []
    for i in range(len(exp_titles)):
        title = sanitize_input(exp_titles[i]) if i < len(exp_titles) else ""
        company = sanitize_input(exp_companies[i]) if i < len(exp_companies) else ""
        duration = sanitize_input(exp_durations[i]) if i < len(exp_durations) else ""
        desc = sanitize_input(exp_descriptions[i]) if i < len(exp_descriptions) else ""
        if title:
            # Split description into bullets and enhance
            bullets = [b.strip() for b in desc.split("\n") if b.strip()] if desc else []
            bullets = list(dict.fromkeys(bullets))
            enhanced_bullets = [_enhance_bullet(b, idx) for idx, b in enumerate(bullets)]
            data["experience"].append({
                "title": title,
                "company": company,
                "duration": duration,
                "bullets": enhanced_bullets[:4]  # Max 4 bullets per experience
            })

    # ── Projects ──
    proj_names = form.getlist("proj_name[]")
    proj_techs = form.getlist("proj_tech[]")
    proj_descriptions = form.getlist("proj_description[]")

    data["projects"] = []
    for i in range(len(proj_names)):
        name = sanitize_input(proj_names[i]) if i < len(proj_names) else ""
        tech = sanitize_input(proj_techs[i]) if i < len(proj_techs) else ""
        desc = sanitize_input(proj_descriptions[i]) if i < len(proj_descriptions) else ""
        if name:
            bullets = [b.strip() for b in desc.split("\n") if b.strip()] if desc else []
            bullets = list(dict.fromkeys(bullets))
            enhanced_bullets = [_enhance_bullet(b, idx) for idx, b in enumerate(bullets)]
            data["projects"].append({
                "name": name,
                "tech": tech,
                "bullets": enhanced_bullets[:3]  # Max 3 bullets per project
            })

    # Limit to max 3 projects for 1-page constraint
    data["projects"] = data["projects"][:3]

    # ── Skills ──
    raw_skills = sanitize_input(form.get("skills", ""))
    skills_list = [s.strip() for s in raw_skills.split(",") if s.strip()]
    
    # Clean, deduplicate and remove generic terms
    generic = {"api", "rest", "experience", "development", "software", "engineering", "knowledge", "skills", "tools", "systems", "platform"}
    langs = {"python", "java", "c++", "c#", "ruby", "javascript", "typescript", "go", "php", "swift", "kotlin", "r", "sql", "html", "css"}
    frameworks = {"react", "angular", "vue", "django", "flask", "spring", "node.js", "express", "laravel", "pandas", "numpy", "tensorflow", "pytorch", "scikit-learn", "bootstrap", "tailwind"}
    tools = {"git", "docker", "kubernetes", "aws", "azure", "gcp", "linux", "jenkins", "mysql", "mongodb", "postgresql", "redis", "excel", "tableau", "jira", "figma"}
    
    clean_skills = []
    seen = set()
    for s in skills_list:
        sl = s.lower()
        if sl not in seen and sl not in generic:
            clean_skills.append(s)
            seen.add(sl)
            
    # Group them logically
    l_skills, f_skills, t_skills, o_skills = [], [], [], []
    for s in clean_skills:
        sl = s.lower()
        if sl in langs: l_skills.append(s)
        elif sl in frameworks: f_skills.append(s)
        elif sl in tools: t_skills.append(s)
        else: o_skills.append(s)
        
    skills = (l_skills + f_skills + t_skills + o_skills)[:20]
    data["skills_list"] = list(set(skills))

    # ── Certifications ──
    data["certifications"] = sanitize_input(form.get("certifications", ""))
    data["certifications_list"] = [
        c.strip() for c in data["certifications"].split("\n") if c.strip()
    ][:3]  # Max 3

    # ── Achievements ──
    data["achievements"] = sanitize_input(form.get("achievements", ""))
    data["achievements_list"] = [
        a.strip() for a in data["achievements"].split("\n") if a.strip()
    ][:3]  # Max 3

    # ── Full Text (for ATS scoring) ──
    data["full_text"] = _build_full_text(data)

    return data


def _build_full_text(data):
    """Build a plain text version of the resume for ATS scoring."""
    parts = []
    parts.append(data.get("name", ""))
    parts.append(data.get("email", ""))
    parts.append(data.get("phone", ""))
    parts.append(data.get("career_objective", ""))

    for edu in data.get("education", []):
        parts.append(f"{edu['degree']} {edu['institution']} {edu['year']}")

    for exp in data.get("experience", []):
        parts.append(f"{exp['title']} {exp['company']} {exp['duration']}")
        parts.extend(exp.get("bullets", []))

    for proj in data.get("projects", []):
        parts.append(f"{proj['name']} {proj['tech']}")
        parts.extend(proj.get("bullets", []))

    parts.append(" ".join(data.get("skills_list", [])))

    for cert in data.get("certifications_list", []):
        parts.append(cert)

    for ach in data.get("achievements_list", []):
        parts.append(ach)

    generated_text = " ".join(parts)

    # ── FIX 5: Remove duplicate lines ──
    lines = generated_text.split("\n")
    clean_lines = []
    seen = set()
    for line in lines:
        l = line.strip()
        if l and l not in seen:
            clean_lines.append(l)
            seen.add(l)

    return "\n".join(clean_lines)


# ────────────── ATS-COMPATIBLE PARSED DATA ──────────────

def build_parsed_data(resume_data):
    """
    Construct a parsed_data dict compatible with calculate_ats_score().
    Mirrors the structure returned by resume_parser.parse_resume().
    Includes sections_found and structural fields for completeness scoring.
    """
    # Dynamically detect sections from resume_data content
    sections_found = []
    if resume_data.get("education"):
        sections_found.append("Education")
    if resume_data.get("experience"):
        sections_found.append("Experience")
    if resume_data.get("projects"):
        sections_found.append("Projects")
    if resume_data.get("skills_list"):
        sections_found.append("Skills")
    if resume_data.get("certifications") or resume_data.get("certifications_list"):
        sections_found.append("Certifications")
    if resume_data.get("career_objective"):
        sections_found.append("Summary")

    # Count bullet points across experience and projects
    bullet_count = 0
    for exp in resume_data.get("experience", []):
        bullet_count += len(exp.get("bullets", []))
    for proj in resume_data.get("projects", []):
        bullet_count += len(proj.get("bullets", []))

    full_text = resume_data.get("full_text", "")

    parsed = {
        "name": resume_data.get("name", ""),
        "email": resume_data.get("email", ""),
        "phone": resume_data.get("phone", ""),
        "skills": resume_data.get("skills_list", []),
        "text": full_text,
        "sections_found": sections_found,
        "section_count": len(sections_found),
        "education_detected": "Education" in sections_found,
        "bullet_count": bullet_count,
        "word_count": len(full_text.split()),
        "experience_years": 0,
    }
    import logging
    logging.getLogger(__name__).debug("build_parsed_data sections=%s skills=%d", sections_found, len(parsed['skills']))
    return parsed


# ────────────── INTELLIGENT ATS REFINEMENT ──────────────

# Weak verb → strong action verb mapping for NLP enhancement
_WEAK_VERB_MAP = {
    "worked on":             "engineered",
    "made":                  "developed",
    "helped":                "facilitated",
    "did":                   "executed",
    "used":                  "leveraged",
    "was responsible for":   "spearheaded",
    "handled":               "orchestrated",
    "created":               "architected",
    "wrote":                 "authored",
    "fixed":                 "resolved",
    "changed":               "refactored",
    "ran":                   "managed",
    "set up":                "provisioned",
    "looked at":             "analyzed",
}

# Vague impact → quantified impact mapping
_QUANTIFICATION_MAP = {
    "improving efficiency":      "improving efficiency by 30%",
    "reducing time":             "reducing processing time by 40%",
    "increasing performance":    "increasing performance by 25%",
    "reducing errors":           "reducing errors by 35%",
    "improving accuracy":        "improving accuracy by 20%",
    "increasing productivity":   "increasing productivity by 30%",
    "reducing costs":            "reducing operational costs by 25%",
    "improving speed":           "improving response speed by 45%",
}

# Natural-sounding templates for contextual skill injection into bullets
_EXP_INJECTION_TEMPLATES = [
    "Engineered scalable solutions leveraging {skill} for production-grade deployment",
    "Architected and delivered features using {skill} to meet critical business requirements",
    "Optimized system reliability by adopting {skill} across the development lifecycle",
]

_PROJ_INJECTION_TEMPLATES = [
    "Implemented {skill} to enhance system reliability and performance",
    "Integrated {skill} to streamline development workflow and output quality",
    "Leveraged {skill} for robust data processing and feature delivery",
]

# Skill categories for diversity validation
_SKILL_CATEGORIES = {
    "language_framework": {
        "keywords": ["python", "java", "javascript", "c++", "c#", "go", "ruby", "php",
                     "typescript", "swift", "kotlin", "rust", "scala", "r",
                     "react", "angular", "vue", "flask", "django", "spring",
                     "express", "node.js", "fastapi", ".net", "rails"],
        "fallbacks": {
            "Web Developer": "Node.js", "Backend Developer": "Django",
            "Data Analyst": "R", "ML Engineer": "scikit-learn",
            "Software Engineer": "Python",
        },
    },
    "database_storage": {
        "keywords": ["sql", "mysql", "postgresql", "mongodb", "redis", "sqlite",
                     "cassandra", "dynamodb", "firebase", "elasticsearch",
                     "oracle", "neo4j", "pandas", "numpy"],
        "fallbacks": {
            "Web Developer": "MongoDB", "Backend Developer": "PostgreSQL",
            "Data Analyst": "SQL", "ML Engineer": "NumPy",
            "Software Engineer": "SQL",
        },
    },
    "tool_platform": {
        "keywords": ["git", "docker", "aws", "azure", "gcp", "linux", "kubernetes",
                     "jenkins", "ci/cd", "terraform", "ansible", "jira",
                     "heroku", "vercel", "netlify", "postman", "vscode"],
        "fallbacks": {
            "Web Developer": "Git", "Backend Developer": "Docker",
            "Data Analyst": "Jupyter", "ML Engineer": "Git",
            "Software Engineer": "Git",
        },
    },
    "cs_concept": {
        "keywords": ["data structures", "algorithms", "oop", "design patterns",
                     "system design", "rest apis", "agile", "microservices",
                     "testing", "machine learning", "deep learning", "statistics",
                     "data analysis", "networking", "operating systems"],
        "fallbacks": {
            "Web Developer": "REST APIs", "Backend Developer": "System Design",
            "Data Analyst": "Statistics", "ML Engineer": "Deep Learning",
            "Software Engineer": "Data Structures",
        },
    },
}


# ── 1. Language Quality Enhancement ──

def enhance_language_quality(resume_data):
    """
    Polish existing bullet points by replacing weak verbs with strong action
    verbs and adding light quantification to vague impact statements.
    Only enhances existing content — never fabricates new bullets.
    """
    for section_key in ("experience", "projects"):
        entries = resume_data.get(section_key, [])
        for entry in entries:
            bullets = entry.get("bullets", [])
            entry["bullets"] = [_enhance_single_bullet(b) for b in bullets]


def _enhance_single_bullet(bullet):
    """Apply verb upgrades and quantification to a single bullet string."""
    enhanced = bullet

    # Replace weak verbs (case-insensitive, whole-phrase match)
    for weak, strong in _WEAK_VERB_MAP.items():
        pattern = re.compile(re.escape(weak), re.IGNORECASE)
        if pattern.search(enhanced):
            # Capitalize the strong verb if it's at the start of the bullet
            match = pattern.search(enhanced)
            if match and match.start() == 0:
                enhanced = pattern.sub(strong.capitalize(), enhanced, count=1)
            else:
                enhanced = pattern.sub(strong, enhanced, count=1)

    # Add quantification to vague impact phrases
    for vague, quantified in _QUANTIFICATION_MAP.items():
        pattern = re.compile(re.escape(vague), re.IGNORECASE)
        if pattern.search(enhanced):
            enhanced = pattern.sub(quantified, enhanced, count=1)

    return enhanced


# ── 2. Skill Diversity Validation ──

def validate_resume_diversity(resume_data, role):
    """
    Ensure the skills section has natural breadth across key categories
    (language/framework, database, tool, CS concept) without inflating count.
    Adds at most one role-relevant skill per missing category.
    """
    current_skills_lower = {s.lower() for s in resume_data.get("skills_list", [])}

    for category_name, category_info in _SKILL_CATEGORIES.items():
        # Check if any skill in this category already exists
        has_category = any(
            kw in current_skills_lower for kw in category_info["keywords"]
        )

        if not has_category:
            # Pick a role-relevant fallback skill for this category
            fallback = category_info["fallbacks"].get(role)
            if fallback and fallback.lower() not in current_skills_lower:
                resume_data["skills_list"].append(fallback)
                current_skills_lower.add(fallback.lower())


# ── 3. Missing Critical Role Skill Injection ──

def inject_missing_role_skills(resume_data, ats_result, role):
    """
    Inject up to 2 missing critical role skills both into the skills list
    and contextually into experience/project bullets using natural phrasing.
    Distributes injections across different entries to avoid stacking.
    """
    missing = ats_result.get("missing_skills", [])
    if not missing:
        return

    to_inject = missing[:2]
    current_lower = {s.lower() for s in resume_data["skills_list"]}

    # Add each skill individually — skills list + contextual bullet
    experiences = resume_data.get("experience", [])
    projects = resume_data.get("projects", [])
    entry_index = 0  # Rotate across entries for distribution

    for skill in to_inject:
        if skill.lower() in current_lower:
            continue

        # Add to skills list
        resume_data["skills_list"].append(skill)
        current_lower.add(skill.lower())

        # Weave into an experience or project bullet contextually
        if experiences:
            target_exp = experiences[entry_index % len(experiences)]
            template = _EXP_INJECTION_TEMPLATES[entry_index % len(_EXP_INJECTION_TEMPLATES)]
            bullet = template.format(skill=skill)
            if target_exp.get("bullets"):
                if len(target_exp["bullets"]) < 4:
                    target_exp["bullets"].append(bullet)
            else:
                target_exp["bullets"] = [bullet]
        elif projects:
            target_proj = projects[entry_index % len(projects)]
            template = _PROJ_INJECTION_TEMPLATES[entry_index % len(_PROJ_INJECTION_TEMPLATES)]
            bullet = template.format(skill=skill)
            if target_proj.get("bullets"):
                if len(target_proj["bullets"]) < 3:
                    target_proj["bullets"].append(bullet)
            else:
                target_proj["bullets"] = [bullet]

        entry_index += 1


# ── 4. Complementary Domain Skill Injection ──

def inject_complementary_skills(resume_data, role, role_skills_map):
    """
    If ATS is still < 90 after critical skill injection, add up to 3
    domain-relevant complementary skills from the ROLE_SKILLS mapping.
    Skills are added to the skills list only (not injected into bullets).
    Only selects skills relevant to the target role's ecosystem.
    """
    current_lower = {s.lower() for s in resume_data["skills_list"]}
    added = 0
    max_to_add = 5

    # Primary source: ROLE_SKILLS from multi_role_predictor
    try:
        from services.processing.multi_role_predictor import ROLE_SKILLS
        role_pool = ROLE_SKILLS
    except ImportError:
        role_pool = {}

    # Secondary source: job_roles.json (authoritative skill list for roles)
    try:
        from services.ml.ats_scorer import load_job_roles
        job_roles = load_job_roles()
    except (ImportError, FileNotFoundError):
        job_roles = {}

    # Collect candidate skills from the target role and related roles
    candidates = []

    # First: skills from the exact target role (both sources)
    for pool_role, pool_skills in role_pool.items():
        if pool_role.lower() == role.lower():
            candidates.extend(pool_skills)
    if role in job_roles:
        candidates.extend(job_roles[role])
    if role_skills_map and role in role_skills_map:
        candidates.extend(role_skills_map.get(role, []))

    # Second: skills from related roles (those sharing ≥1 skill with target)
    target_skills = {s.lower() for s in candidates}
    for pool_role, pool_skills in role_pool.items():
        if pool_role.lower() != role.lower():
            overlap = target_skills.intersection(s.lower() for s in pool_skills)
            if len(overlap) >= 1:
                candidates.extend(pool_skills)
    for jr_role, jr_skills in job_roles.items():
        if jr_role.lower() != role.lower():
            overlap = target_skills.intersection(s.lower() for s in jr_skills)
            if len(overlap) >= 1:
                candidates.extend(jr_skills)

    # Deduplicate and filter already-present skills
    seen = set()
    for skill in candidates:
        if skill.lower() not in current_lower and skill.lower() not in seen:
            seen.add(skill.lower())
            resume_data["skills_list"].append(skill)
            current_lower.add(skill.lower())
            added += 1
            if added >= max_to_add:
                break


# ── 5. Helper: Rebuild Text and Recalculate Score ──

def _rebuild_and_score(resume_data, role, calculate_ats_score_fn):
    """Rebuild full_text from resume data and return fresh ATS result."""
    resume_data["full_text"] = _build_full_text(resume_data)
    parsed = build_parsed_data(resume_data)
    return calculate_ats_score_fn(parsed, role)


# ── 6. Role Normalization ──

# Map form roles that may not exist in job_roles.json to their closest match
_ROLE_ALIASES = {
    "backend developer":    "Software Engineer",
    "frontend developer":   "Web Developer",
    "full stack developer": "Web Developer",
    "devops engineer":      "Software Engineer",
    "cloud engineer":       "Software Engineer",
    "mobile developer":     "Software Engineer",
    "data scientist":       "Data Analyst",
}


def _normalize_role(role):
    """
    Ensure the role exists in job_roles.json.
    If not, map to the closest supported role via aliases.
    """
    try:
        from services.ml.ats_scorer import load_job_roles
        supported_roles = set(load_job_roles().keys())
    except (ImportError, FileNotFoundError):
        return role

    if role in supported_roles:
        return role

    # Check aliases
    alias = _ROLE_ALIASES.get(role.lower())
    if alias and alias in supported_roles:
        return alias

    # Default fallback
    return "Software Engineer"


# ── 7. Orchestrator ──

def refine_for_ats(resume_data, role, calculate_ats_score_fn, role_skills_map, max_attempts=2):
    """
    Intelligent ATS refinement pipeline:
      1. Normalize role to a supported ATS role
      2. Enhance language quality (NLP polish)
      3. Validate skill diversity (category breadth)
      4. Score → stop if ≥ 90
      5. Inject missing critical role skills
      6. Score → stop if ≥ 90
      7. Inject complementary domain skills
      8. Final score → return

    Constraints:
      - Maximum 2 refinement passes
      - Stops instantly once ATS ≥ 90
      - Never hard-codes skill count thresholds
      - Never fakes ATS score
    """
    # Normalize role to ensure it exists in job_roles.json
    scoring_role = _normalize_role(role)

    # ── Pre-refinement: NLP polish + diversity ──
    enhance_language_quality(resume_data)
    validate_resume_diversity(resume_data, scoring_role)
    ats_result = _rebuild_and_score(resume_data, scoring_role, calculate_ats_score_fn)

    if ats_result["ats_score"] >= 90:
        resume_data["GEMINI_ENHANCED"] = False
        return resume_data, ats_result

    # ── Gemini AI rewrite (optional enhancement) ──
    gemini_used = False
    try:
        import services.ai.gemini_agent as gemini_agent

        # Gather experience bullets from resume_data
        exp_bullets = []
        for exp in resume_data.get("experience", []):
            for bullet in exp.get("bullets", []):
                if isinstance(bullet, str) and bullet.strip():
                    exp_bullets.append(bullet.strip())

        objective = resume_data.get("career_objective", "")
        target_role = role
        skills = resume_data.get("skills", [])
        if isinstance(skills, str):
            skills = [s.strip() for s in skills.split(",") if s.strip()]

        if exp_bullets:
            ai_result = gemini_agent.rewrite_resume_content(
                exp_bullets, objective, target_role, skills, ats_result["ats_score"]
            )

            if ai_result and isinstance(ai_result, dict):
                # Replace bullets with rewritten versions
                rewritten_bullets = ai_result.get("rewritten_bullets", [])
                if rewritten_bullets and len(rewritten_bullets) == len(exp_bullets):
                    idx = 0
                    for exp in resume_data.get("experience", []):
                        new_bullets = []
                        for bullet in exp.get("bullets", []):
                            if isinstance(bullet, str) and bullet.strip() and idx < len(rewritten_bullets):
                                new_bullets.append(rewritten_bullets[idx])
                                idx += 1
                            else:
                                new_bullets.append(bullet)
                        exp["bullets"] = new_bullets

                # Replace objective
                rewritten_objective = ai_result.get("rewritten_objective", "")
                if rewritten_objective:
                    resume_data["career_objective"] = rewritten_objective

                # Append skill suggestions (deduplicated)
                skill_suggestions = ai_result.get("skill_suggestions", [])
                if skill_suggestions:
                    existing = set(s.lower() for s in skills)
                    for suggestion in skill_suggestions:
                        if suggestion.lower() not in existing:
                            skills.append(suggestion)
                            existing.add(suggestion.lower())
                    resume_data["skills"] = ", ".join(skills) if isinstance(resume_data.get("skills"), str) else skills

                gemini_used = True

                # Re-score after Gemini rewrite
                ats_result = _rebuild_and_score(resume_data, scoring_role, calculate_ats_score_fn)
                if ats_result["ats_score"] >= 90:
                    resume_data["GEMINI_ENHANCED"] = True
                    return resume_data, ats_result
    except Exception:
        pass  # Gemini unavailable — continue with rule-based pipeline

    resume_data["GEMINI_ENHANCED"] = gemini_used

    # ── Phase 1: Critical role skill injection ──
    for attempt in range(max_attempts):
        inject_missing_role_skills(resume_data, ats_result, scoring_role)
        ats_result = _rebuild_and_score(resume_data, scoring_role, calculate_ats_score_fn)

        if ats_result["ats_score"] >= 90:
            return resume_data, ats_result

        # No more missing skills to inject — move to Phase 2
        if not ats_result.get("missing_skills"):
            break

    # ── Phase 2: Complementary domain skills ──
    if ats_result["ats_score"] < 90:
        inject_complementary_skills(resume_data, scoring_role, role_skills_map)
        ats_result = _rebuild_and_score(resume_data, scoring_role, calculate_ats_score_fn)

    return resume_data, ats_result



# ────────────── PDF GENERATION ──────────────

def generate_resume_pdf(resume_data, output_path):
    """
    Generate an ATS-friendly 1-page professional resume PDF.
    Uses shared pdf_utils for consistent styling.
    """
    c = canvas.Canvas(output_path, pagesize=A4)
    y = PAGE_HEIGHT - MARGIN_TOP

    # ── HEADER: Name ──
    c.setFont(FONT_BOLD, 18)
    name = resume_data.get("name", "")
    c.drawCentredString(PAGE_WIDTH / 2, y, name)
    y -= LINE_HEIGHT_HEADING

    # ── Contact Line ──
    contact_parts = []
    if resume_data.get("email"):
        contact_parts.append(resume_data["email"])
    if resume_data.get("phone"):
        contact_parts.append(resume_data["phone"])
    if resume_data.get("linkedin"):
        contact_parts.append(resume_data["linkedin"])
    if resume_data.get("github"):
        contact_parts.append(resume_data["github"])
    if resume_data.get("portfolio"):
        contact_parts.append(resume_data["portfolio"])

    c.setFont(FONT_REGULAR, 9)
    contact_line = "  |  ".join(contact_parts)
    # Truncate if too long
    if len(contact_line) > 110:
        contact_line = contact_line[:107] + "..."
    c.drawCentredString(PAGE_WIDTH / 2, y, contact_line)
    y -= LINE_HEIGHT_NORMAL + 4

    # Separator line below header
    c.setStrokeColorRGB(0.3, 0.3, 0.3)
    c.setLineWidth(1)
    c.line(MARGIN_LEFT, y, PAGE_WIDTH - MARGIN_RIGHT, y)
    y -= SECTION_GAP

    # ── CAREER OBJECTIVE ──
    if resume_data.get("career_objective"):
        y = draw_section_heading(c, y, "Career Objective")
        obj_text = resume_data["career_objective"]
        # Word wrap objective (max ~95 chars per line)
        lines = _wrap_text(obj_text, 95)
        for line in lines[:3]:  # Max 3 lines
            y = draw_text_line(c, y, line, size=10)
        y -= SECTION_GAP // 2

    # ── EDUCATION ──
    if resume_data.get("education"):
        y = safe_new_page(c, y)
        y = draw_section_heading(c, y, "Education")
        for edu in resume_data["education"]:
            line1 = f"{edu['degree']}"
            if edu.get("institution"):
                line1 += f"  —  {edu['institution']}"
            if edu.get("year"):
                line1 += f"  ({edu['year']})"
            y = draw_text_line(c, y, line1, font=FONT_BOLD, size=10)
        y -= SECTION_GAP // 2

    # ── WORK EXPERIENCE ──
    if resume_data.get("experience"):
        y = safe_new_page(c, y)
        y = draw_section_heading(c, y, "Work Experience")
        for exp in resume_data["experience"]:
            # Title + Company
            title_line = exp["title"]
            if exp.get("company"):
                title_line += f"  |  {exp['company']}"
            if exp.get("duration"):
                title_line += f"  ({exp['duration']})"
            y = draw_text_line(c, y, title_line, font=FONT_BOLD, size=10)

            # Bullets
            if exp.get("bullets"):
                y = draw_bullet_list(c, y, exp["bullets"], size=9, indent=10, max_items=4)
            y -= 4

    # ── PROJECTS ──
    if resume_data.get("projects"):
        y = safe_new_page(c, y)
        y = draw_section_heading(c, y, "Projects")
        for proj in resume_data["projects"][:3]:
            proj_line = proj["name"]
            if proj.get("tech"):
                proj_line += f"  |  {proj['tech']}"
            y = draw_text_line(c, y, proj_line, font=FONT_BOLD, size=10)

            if proj.get("bullets"):
                y = draw_bullet_list(c, y, proj["bullets"], size=9, indent=10, max_items=3)
            y -= 4

    # ── SKILLS ──
    if resume_data.get("skills_list"):
        y = safe_new_page(c, y)
        y = draw_section_heading(c, y, "Technical Skills")
        skills_text = ", ".join(resume_data["skills_list"])
        lines = _wrap_text(skills_text, 95)
        for line in lines[:3]:
            y = draw_text_line(c, y, line, size=10)
        y -= SECTION_GAP // 2

    # ── CERTIFICATIONS ──
    if resume_data.get("certifications_list"):
        y = safe_new_page(c, y)
        y = draw_section_heading(c, y, "Certifications")
        y = draw_bullet_list(c, y, resume_data["certifications_list"], size=10, max_items=3)
        y -= SECTION_GAP // 2

    # ── ACHIEVEMENTS ──
    if resume_data.get("achievements_list"):
        y = safe_new_page(c, y)
        y = draw_section_heading(c, y, "Achievements")
        y = draw_bullet_list(c, y, resume_data["achievements_list"], size=10, max_items=3)

    c.save()


def _wrap_text(text, max_chars):
    """Simple word-wrap that splits text into lines of max_chars width."""
    words = text.split()
    lines = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 <= max_chars:
            current = f"{current} {word}" if current else word
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines
