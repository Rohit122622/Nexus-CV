"""
Input Validation Module for Nexus CV.
Provides resume text validation, job description validation, and AI-based
content quality checks using NLP heuristics.  Runs BEFORE any ATS scoring or
ML inference to reject invalid/random uploads early.
"""

import re
import logging

logger = logging.getLogger(__name__)

# ────────────── RESUME KEYWORDS ──────────────

_RESUME_SECTION_KEYWORDS = [
    "experience", "education", "skills", "projects", "summary",
    "objective", "career objective", "work experience", "internship",
    "certifications", "achievements", "professional experience",
    "technical skills", "qualifications", "publications", "awards",
    "contact", "references", "volunteer", "extracurricular",
]

# ────────────── TECHNICAL KEYWORDS (for JD validation) ──────────────

_TECHNICAL_KEYWORDS = [
    "python", "java", "javascript", "react", "angular", "node", "sql",
    "api", "aws", "docker", "kubernetes", "machine learning", "data",
    "cloud", "devops", "ci/cd", "git", "linux", "agile", "testing",
    "frontend", "backend", "full stack", "database", "html", "css",
    "flask", "django", "spring", "tensorflow", "pytorch", "mongodb",
    "postgresql", "rest", "microservices", "typescript", "c++", "c#",
    "go", "rust", "scala", "redis", "elasticsearch", "kafka",
    "development", "engineering", "software", "programming", "coding",
    "design", "architecture", "system", "network", "security",
    "analytics", "deep learning", "nlp", "computer vision",
]

# ────────────── JD STRUCTURAL KEYWORDS ──────────────

_JD_STRUCTURAL_KEYWORDS = [
    "required", "responsibilities", "skills", "qualifications",
    "requirements", "role", "duties", "job description",
    "must have", "preferred", "experience",
]

# ────────────── COMMON ENGLISH STOPWORDS (subset) ──────────────

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "out", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "because",
    "but", "and", "or", "if", "while", "about", "up", "it", "its",
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "they", "them", "their", "this", "that", "these",
    "those", "what", "which", "who", "whom",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. RESUME TEXT VALIDATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def validate_resume_text(text):
    """
    Validate extracted PDF text to ensure it looks like a real resume.

    Checks:
      - Minimum 300 characters of extractable text
      - Contains at least 2 resume section keywords
      - Not mostly unreadable/garbled content

    Returns:
        (is_valid: bool, error_message: str)
    """
    if not text or not text.strip():
        return False, (
            "Uploaded file does not contain any readable text. "
            "Please upload a text-based PDF resume."
        )

    clean = text.strip()

    # ── Length check ──
    if len(clean) < 500:
        return False, (
            "Uploaded file does not appear to be a valid resume. "
            "The document contains too little text (minimum 500 characters required). "
            "Please upload a professional resume in text-based PDF format."
        )

    # ── Resume section keyword check ──
    text_lower = clean.lower()
    matched_sections = [
        kw for kw in _RESUME_SECTION_KEYWORDS if kw in text_lower
    ]
    if len(matched_sections) < 2:
        return False, (
            "Uploaded file does not appear to be a valid resume. "
            "No recognizable resume sections (Experience, Education, Skills, etc.) were found. "
            "Please upload a professional resume in text-based PDF format."
        )

    # ── Garbled / unreadable content check ──
    # If >40% of chars are non-ASCII/non-printable, likely image-based or corrupted
    printable_count = sum(1 for c in clean if c.isprintable() or c in "\n\r\t")
    if len(clean) > 0 and (printable_count / len(clean)) < 0.60:
        return False, (
            "Uploaded file contains mostly unreadable content. "
            "Please upload a text-based PDF resume (not a scanned image)."
        )

    return True, ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. JOB DESCRIPTION VALIDATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def validate_job_description(text):
    """
    Validate job description text before running JD matching.

    Checks:
      - Minimum 150 characters
      - Contains at least 3 technical keywords
      - Contains at least 1 structural keyword (required/responsibilities/etc.)

    Returns:
        (is_valid: bool, error_message: str)
    """
    if not text or not text.strip():
        return True, ""  # JD is optional in some flows

    clean = text.strip()

    # ── Length check ──
    if len(clean) < 200:
        return False, (
            "Job description is too short (minimum 200 characters required). "
            "Please provide a valid job description containing role requirements and skills."
        )

    text_lower = clean.lower()

    # ── Technical keyword check ──
    tech_found = sum(1 for kw in _TECHNICAL_KEYWORDS if kw in text_lower)
    if tech_found < 3:
        return False, (
            "Job description does not contain enough technical keywords. "
            "Please provide a valid job description containing role requirements and skills."
        )

    # ── Structural keyword check ──
    has_structural = any(kw in text_lower for kw in _JD_STRUCTURAL_KEYWORDS)
    if not has_structural:
        return False, (
            "Job description is missing structural elements (e.g., requirements, responsibilities, skills). "
            "Please provide a valid job description containing role requirements and skills."
        )

    return True, ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. AI-BASED CONTENT QUALITY CHECK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def validate_content_quality(text):
    """
    NLP-heuristic content quality check.

    Analyses:
      - Sentence count (expects ≥5 for a real resume)
      - Keyword density (resume-relevant words)
      - Stopword ratio (natural language indicator)
      - Bullet pattern presence (•, -, ▪, numbered lists)

    Returns:
        dict: {
            "is_valid": bool,
            "confidence_score": int (0-100),
            "reason": str,
            "sections_found": list[str]
        }
    """
    if not text or not text.strip():
        return {
            "is_valid": False,
            "confidence_score": 0,
            "reason": "No text content to analyze.",
            "sections_found": [],
        }

    clean = text.strip()
    text_lower = clean.lower()
    words = re.findall(r'[a-zA-Z]+', clean)
    word_count = len(words)

    score = 0

    # ── 1. Sentence count (max 25 points) ──
    sentences = re.split(r'[.!?\n]', clean)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    sentence_count = len(sentences)

    if sentence_count >= 15:
        score += 25
    elif sentence_count >= 10:
        score += 20
    elif sentence_count >= 5:
        score += 15
    elif sentence_count >= 3:
        score += 8
    # else 0

    # ── 2. Keyword density (max 30 points) ──
    resume_kw_hits = sum(1 for kw in _RESUME_SECTION_KEYWORDS if kw in text_lower)
    tech_kw_hits = sum(1 for kw in _TECHNICAL_KEYWORDS if kw in text_lower)
    total_kw = resume_kw_hits + tech_kw_hits

    if total_kw >= 12:
        score += 30
    elif total_kw >= 8:
        score += 25
    elif total_kw >= 5:
        score += 18
    elif total_kw >= 3:
        score += 10
    elif total_kw >= 1:
        score += 5

    # ── 3. Stopword ratio (max 20 points) ──
    # Normal English text has ~40-60% stopwords. Too low = gibberish, too high = filler
    if word_count > 0:
        stopword_count = sum(1 for w in words if w.lower() in _STOPWORDS)
        stopword_ratio = stopword_count / word_count

        if 0.20 <= stopword_ratio <= 0.65:
            score += 20  # Normal range
        elif 0.15 <= stopword_ratio <= 0.70:
            score += 12  # Slightly off
        else:
            score += 3   # Likely gibberish or filler
    else:
        score += 0

    # ── 4. Bullet pattern presence (max 15 points) ──
    bullet_patterns = re.findall(r'(?m)^[\s]*[•\-▪▸◦★✦✓✔➤►]\s', clean)
    numbered_patterns = re.findall(r'(?m)^[\s]*\d+[.)]\s', clean)
    bullet_count = len(bullet_patterns) + len(numbered_patterns)

    if bullet_count >= 5:
        score += 15
    elif bullet_count >= 3:
        score += 10
    elif bullet_count >= 1:
        score += 5

    # ── 5. Structure bonus (max 10 points) ──
    # Check for email, phone, URL patterns — strong resume indicators
    has_email = bool(re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', clean))
    has_phone = bool(re.search(r'\b\d{10}\b|\+\d{1,3}[\s-]?\d{3,}', clean))
    has_url = bool(re.search(r'https?://|linkedin|github', text_lower))

    structure_points = 0
    if has_email:
        structure_points += 4
    if has_phone:
        structure_points += 3
    if has_url:
        structure_points += 3
    score += min(structure_points, 10)

    # ── Clamp score ──
    confidence_score = max(0, min(100, score))

    # ── Sections found ──
    sections_found = [
        kw.title() for kw in _RESUME_SECTION_KEYWORDS if kw in text_lower
    ]

    # ── Determine validity ──
    is_valid = confidence_score >= 60

    # ── Build reason ──
    if is_valid:
        reason = "Content appears to be a valid, structured resume."
    elif confidence_score >= 40:
        reason = "Content has some resume-like elements but lacks sufficient structure."
    else:
        reason = "Content does not appear to be a valid resume. It may be random text or non-resume content."

    logger.info(f"Content quality score: {confidence_score} (sentences={sentence_count}, keywords={total_kw}, bullets={bullet_count})")

    return {
        "is_valid": is_valid,
        "confidence_score": confidence_score,
        "reason": reason,
        "sections_found": sections_found,
    }
