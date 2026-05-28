"""
Resume parser with semantic skill extraction + structural analysis.
Uses model_hub for embeddings (BGE-large with MiniLM fallback),
spaCy for NER, rapidfuzz for alias resolution.
"""

import re
import os
import json
import warnings

warnings.filterwarnings("ignore")

from utils.skill_normalizer import normalize as normalize_skills

import pdfplumber

# ── spaCy: try trf first, fallback to sm ──
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_trf")
    except OSError:
        nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

# ── Model Hub: centralized embeddings ──
try:
    import services.ml.model_hub as model_hub
    import numpy as np
except ImportError:
    model_hub = None
    np = None

# ── RapidFuzz: optional for fuzzy matching ──
try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Section heading patterns ──
SECTION_PATTERNS = {
    "Education": r"(?i)\b(education|academic|degree|university|college|school)\b",
    "Experience": r"(?i)\b(experience|work\s*experience|employment|professional\s*experience|internship)\b",
    "Skills": r"(?i)\b(skills|technical\s*skills|core\s*competencies|technologies)\b",
    "Projects": r"(?i)\b(projects|personal\s*projects|academic\s*projects)\b",
    "Certifications": r"(?i)\b(certifications?|licenses?|credentials?)\b",
    "Summary": r"(?i)\b(summary|objective|career\s*objective|profile|about)\b",
    "Awards": r"(?i)\b(awards?|achievements?|honors?)\b",
    "Publications": r"(?i)\b(publications?|papers?|research)\b",
}

# ── Date patterns for experience years ──
DATE_PATTERN = re.compile(
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|"
    r"april|may|june|july|august|september|october|november|december)"
    r"[\s.,\-]*(\d{4})",
    re.IGNORECASE
)
YEAR_RANGE = re.compile(r"(\d{4})\s*[-–—to]+\s*(\d{4}|present|current|now)", re.IGNORECASE)

# ── Load skill taxonomy ──
_all_skills = []
_skill_embeddings = None


def _load_skills():
    """Load all skills from taxonomy JSON, falling back to skills.txt."""
    global _all_skills, _skill_embeddings

    taxonomy_path = os.path.join(PROJECT_ROOT, "data", "skills_taxonomy.json")
    fallback_path = os.path.join(PROJECT_ROOT, "data", "skills.txt")

    if os.path.exists(taxonomy_path):
        try:
            with open(taxonomy_path, "r", encoding="utf-8") as f:
                taxonomy = json.load(f)
            for category_skills in taxonomy.values():
                _all_skills.extend(category_skills)
            _all_skills = list(set(_all_skills))
        except Exception:
            _all_skills = []

    if os.path.exists(fallback_path):
        with open(fallback_path, "r", encoding="utf-8") as f:
            txt_skills = f.read().splitlines()
        for s in txt_skills:
            s = s.strip()
            if s and s.lower() not in [x.lower() for x in _all_skills]:
                _all_skills.append(s)

    if not _all_skills:
        _all_skills = ["python", "java", "sql", "javascript", "html", "css"]

    # Pre-compute skill embeddings via model_hub
    if model_hub is not None and _all_skills:
        try:
            embedder = model_hub.get_embedder()
            if embedder:
                _skill_embeddings = embedder.encode(
                    _all_skills, normalize_embeddings=True
                )
        except Exception:
            pass


_load_skills()


def extract_text(path):
    import warnings
    import logging
    text = ""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
    except Exception as e:
        logging.getLogger(__name__).warning("PDF extraction error: %s", e)
    return text


def extract_email(text):
    match = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return match[0] if match else ""


def extract_phone(text):
    match = re.findall(r"\b\d{10}\b", text)
    return match[0] if match else ""


def extract_name(text):
    if nlp is None:
        return ""
    doc = nlp(text[:3000])
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return ""


def extract_sections(text):
    """Detect which resume sections are present."""
    found = []
    for section, pattern in SECTION_PATTERNS.items():
        if re.search(pattern, text):
            found.append(section)
    return found


def estimate_experience_years(text):
    """Estimate total years of experience from date ranges."""
    total_months = 0

    for match in YEAR_RANGE.finditer(text):
        start_year = int(match.group(1))
        end_str = match.group(2).lower()
        if end_str in ("present", "current", "now"):
            from datetime import datetime
            end_year = datetime.now().year
        else:
            try:
                end_year = int(end_str)
            except ValueError:
                continue

        if 1990 <= start_year <= 2030 and 1990 <= end_year <= 2030:
            total_months += max(0, (end_year - start_year) * 12)

    # Deduplicate overlapping ranges (rough estimate, cap at 30 years)
    years = min(total_months / 12.0, 30.0)
    return round(years, 1)


def count_bullets(text):
    """Count bullet-point lines."""
    bullet_patterns = [
        r"^\s*[•●○■◆▪→]\s",
        r"^\s*[-–—]\s",
        r"^\s*\d+[.)]\s",
        r"^\s*[a-z][.)]\s",
    ]
    count = 0
    for line in text.split("\n"):
        for pattern in bullet_patterns:
            if re.match(pattern, line.strip()):
                count += 1
                break
    return count


def extract_skills(text):
    """
    Strict dictionary-based skill extraction to prevent dirty text.
    Only allows standardized skills from taxonomy.
    """
    text_lower = text.lower()
    text_clean = re.sub(r'[^\w\s\-\+\#]', ' ', text_lower)
    tokens = text_clean.split()
    
    # Generate unigrams, bigrams, and trigrams
    phrases = set(tokens)
    for i in range(len(tokens)-1):
        phrases.add(f"{tokens[i]} {tokens[i+1]}")
    for i in range(len(tokens)-2):
        phrases.add(f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}")
        
    GENERIC_WORDS = {
        "api", "rest", "experience", "development", "software", "engineering",
        "knowledge", "skills", "tools", "systems", "platform", "engineer",
        "management", "responsibility", "requirements", "qualifications",
        "team", "work", "ability", "understanding", "proficiency",
        "familiarity", "working", "strong", "excellent", "good", "basic", "advanced"
    }
    
    found_skills = set()
    for skill in _all_skills:
        skill_clean = skill.strip().lower()
        if skill_clean in GENERIC_WORDS or len(skill_clean) < 2:
            continue
        if skill_clean in phrases:
            found_skills.add(skill_clean)
            
    # Clean skill extraction rule: max 2 words, < 25 chars
    skills = [
        s.lower().strip()
        for s in found_skills
        if isinstance(s, str)
        and len(s.split()) <= 2
        and len(s) < 25
    ]
    skills = list(set(skills))
            
    return normalize_skills(sorted(skills)[:20])


def parse_resume(path):
    """
    Parse a PDF resume and return structured data.
    Includes new fields: sections_found, experience_years,
    education_detected, bullet_count, word_count, section_count.
    """
    text = extract_text(path)
    skills = extract_skills(text)
    sections = extract_sections(text)

    data = {
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "skills": skills,
        "text": text,
        # ── NEW structured fields ──
        "sections_found": sections,
        "experience_years": estimate_experience_years(text),
        "education_detected": "Education" in sections,
        "bullet_count": count_bullets(text),
        "word_count": len(text.split()),
        "section_count": len(sections)
    }

    return data
