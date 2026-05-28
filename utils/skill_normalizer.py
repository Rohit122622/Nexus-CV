"""
Skill Normalizer — Global utility for consistent skill handling.
Deduplicates, lowercases, strips whitespace, and removes non-string entries.
"""

import re

# Phrases to strip from skill names (e.g. "Java experience" → "java")
_NOISE_SUFFIXES = re.compile(
    r"\s+(experience|knowledge|proficiency|skills?|development|engineering"
    r"|certification|expertise|familiarity|understanding|background|ability)\s*$",
    re.IGNORECASE,
)


def normalize(skills):
    """
    Normalize a list of skills:
      - Filter non-strings
      - Lowercase + strip
      - Remove noise suffixes ("Java experience" → "java")
      - Deduplicate (preserves first occurrence order)
    """
    seen = set()
    result = []
    for s in skills:
        if not isinstance(s, str):
            continue
        cleaned = s.lower().strip()
        # Strip noise suffixes
        cleaned = _NOISE_SUFFIXES.sub("", cleaned).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result
