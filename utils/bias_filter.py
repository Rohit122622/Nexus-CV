"""
Bias Filter — Removes PII and demographic signals before agent processing.

Strips: name, email, phone, location, gender pronouns, age references.
Applied ONLY to text sent to agents — original data preserved for display.
"""

import re


def anonymize(text, candidate_id="Candidate"):
    """
    Remove demographic and personally-identifying information from resume text.
    Returns anonymized text safe for unbiased agent evaluation.

    Args:
        text: original resume text
        candidate_id: replacement label (e.g., "Candidate A")

    Returns:
        anonymized text string
    """
    if not text:
        return text

    anon = text

    # ── Strip email addresses ──
    anon = re.sub(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", "[EMAIL]", anon)

    # ── Strip phone numbers (various formats) ──
    anon = re.sub(r"(?:\+?\d{1,3}[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}", "[PHONE]", anon)
    anon = re.sub(r"\b\d{10}\b", "[PHONE]", anon)

    # ── Strip URLs / LinkedIn / GitHub profiles ──
    anon = re.sub(r"https?://[^\s]+", "[URL]", anon)
    anon = re.sub(r"(?:linkedin\.com|github\.com)/[\w\-/]+", "[URL]", anon, flags=re.IGNORECASE)

    # ── Strip location/address patterns ──
    # Common patterns: "City, State", "City, Country", etc.
    anon = re.sub(
        r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)?,\s*(?:[A-Z]{2}|[A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s*\d{5,6}?\b",
        "[LOCATION]", anon
    )

    # ── Strip gender pronouns ──
    anon = re.sub(r"\b(he|she|him|her|his|hers)\b", "they", anon, flags=re.IGNORECASE)
    anon = re.sub(r"\b(himself|herself)\b", "themselves", anon, flags=re.IGNORECASE)

    # ── Strip age/date of birth references ──
    anon = re.sub(r"(?i)\b(?:age|d\.?o\.?b\.?|date\s*of\s*birth)\s*[:\-]?\s*\d+[/\-\s]\d+[/\-\s]?\d*", "[DOB]", anon)
    anon = re.sub(r"(?i)\bage[d\s:]*\d{1,3}\b", "[AGE]", anon)

    # ── Strip the candidate name (first line is often the name) ──
    lines = anon.split("\n")
    if lines:
        first_line = lines[0].strip()
        # If first line is short and doesn't contain common section words, it's probably the name
        if (len(first_line) < 40 and first_line and
                not re.search(r"(?i)(education|experience|skills|summary|objective|resume)", first_line)):
            lines[0] = candidate_id
            anon = "\n".join(lines)

    return anon


def strip_name_from_text(text, name):
    """Remove a specific known name from text."""
    if not name or not text:
        return text
    # Remove case-insensitive name occurrences
    return re.sub(re.escape(name), "[CANDIDATE]", text, flags=re.IGNORECASE)
