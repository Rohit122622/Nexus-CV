"""
Semantic Chunker — Section-aware resume chunking for RAG retrieval.

Detects resume sections (Skills, Experience, Projects, Education, Summary)
and chunks within section boundaries with configurable size and overlap.
"""

import re

# Section heading patterns (reused from resume_parser)
SECTION_PATTERNS = {
    "Skills": r"(?i)^[\s]*(?:skills|technical\s*skills|core\s*competencies|technologies)[\s:]*$",
    "Experience": r"(?i)^[\s]*(?:experience|work\s*experience|employment|professional\s*experience|internship)[\s:]*$",
    "Projects": r"(?i)^[\s]*(?:projects|personal\s*projects|academic\s*projects)[\s:]*$",
    "Education": r"(?i)^[\s]*(?:education|academic|degree|university|college|school)[\s:]*$",
    "Summary": r"(?i)^[\s]*(?:summary|objective|career\s*objective|profile|about\s*me?)[\s:]*$",
    "Certifications": r"(?i)^[\s]*(?:certifications?|licenses?|credentials?)[\s:]*$",
    "Awards": r"(?i)^[\s]*(?:awards?|achievements?|honors?)[\s:]*$",
}

# Broader inline patterns for section detection within lines
SECTION_INLINE = {
    "Skills": r"(?i)\b(skills|technical\s*skills|core\s*competencies|technologies)\b",
    "Experience": r"(?i)\b(experience|work\s*experience|employment|professional\s*experience)\b",
    "Projects": r"(?i)\b(projects|personal\s*projects|academic\s*projects)\b",
    "Education": r"(?i)\b(education|academic|degree|university|college)\b",
    "Summary": r"(?i)\b(summary|objective|career\s*objective|profile)\b",
    "Certifications": r"(?i)\b(certifications?|licenses?|credentials?)\b",
}


def detect_sections(text):
    """
    Detect section boundaries in resume text.
    Returns list of (section_name, start_idx, end_idx) tuples.
    """
    lines = text.split("\n")
    sections = []
    current_section = "Header"
    current_start = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        # Check if this line is a section heading
        for section_name, pattern in SECTION_PATTERNS.items():
            if re.match(pattern, stripped):
                # Close previous section
                end_char = sum(len(l) + 1 for l in lines[:i])
                if current_start < end_char:
                    sections.append((current_section, current_start, end_char))
                current_section = section_name
                current_start = end_char
                break
        else:
            # Check inline patterns for short lines (likely headings)
            if len(stripped) < 50:
                for section_name, pattern in SECTION_INLINE.items():
                    if re.search(pattern, stripped) and stripped.count(" ") < 6:
                        end_char = sum(len(l) + 1 for l in lines[:i])
                        if current_start < end_char:
                            sections.append((current_section, current_start, end_char))
                        current_section = section_name
                        current_start = end_char
                        break

    # Close final section
    if current_start < len(text):
        sections.append((current_section, current_start, len(text)))

    return sections


def chunk_text(text, min_size=300, max_size=800, overlap=100):
    """
    Section-aware semantic chunking.

    1. Detect resume sections
    2. Split each section into chunks at paragraph boundaries
    3. Tag each chunk with its section name
    4. Maintain overlap between chunks within same section

    Returns list of dicts: { text, section, char_count, index }
    """
    sections = detect_sections(text)
    chunks = []
    chunk_idx = 0

    for section_name, start, end in sections:
        section_text = text[start:end].strip()
        if not section_text or len(section_text) < 30:
            continue

        # Split by paragraph (double newline) or bullet points
        paragraphs = re.split(r"\n\s*\n|\n(?=[\s]*[•●○■◆▪→\-–—])", section_text)

        current_chunk = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) + 1 <= max_size:
                current_chunk += ("\n" + para if current_chunk else para)
            else:
                # Save current chunk if it meets minimum
                if len(current_chunk) >= min_size:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "section": section_name,
                        "char_count": len(current_chunk.strip()),
                        "index": chunk_idx
                    })
                    chunk_idx += 1
                    # Start new chunk with overlap
                    overlap_text = current_chunk[-overlap:] if overlap > 0 else ""
                    current_chunk = overlap_text + "\n" + para
                elif current_chunk:
                    # Current chunk too small — extend it
                    current_chunk += "\n" + para
                else:
                    current_chunk = para

        # Save remaining text
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "section": section_name,
                "char_count": len(current_chunk.strip()),
                "index": chunk_idx
            })
            chunk_idx += 1

    # Fallback: if no sections detected, chunk naively
    if not chunks:
        words = text.split()
        for i in range(0, len(words), 100):
            chunk_words = words[max(0, i - 20):i + 100]
            chunk_text_str = " ".join(chunk_words)
            if len(chunk_text_str) >= 50:
                chunks.append({
                    "text": chunk_text_str,
                    "section": "Unknown",
                    "char_count": len(chunk_text_str),
                    "index": chunk_idx
                })
                chunk_idx += 1

    return chunks


def get_priority_chunks(chunks, jd_text, top_k=5, threshold=0.15):
    """
    TRUE VECTOR RETRIEVAL: Use FAISS index for semantic chunk retrieval.

    JD-Aware Strategy:
      - If JD is skill-heavy -> boost Skills/Projects chunks
      - If JD is experience-heavy -> boost Experience chunks
      - Only top 3-5 chunks sent to LLM (reduces token usage)

    Falls back to embedding dot-product if FAISS unavailable.
    Falls back to section priority if embeddings unavailable.
    """
    if not chunks:
        return []

    # ── Detect JD focus for retrieval strategy ──
    jd_lower = jd_text.lower() if jd_text else ""
    skill_keywords = sum(1 for w in ["python", "java", "react", "sql", "aws", "docker",
                                      "kubernetes", "node", "typescript", "golang",
                                      "tensorflow", "pytorch", "skills", "proficient"]
                         if w in jd_lower)
    exp_keywords = sum(1 for w in ["years", "experience", "senior", "lead", "managed",
                                    "leadership", "team", "architect", "principal"]
                       if w in jd_lower)

    jd_focus = "skills" if skill_keywords > exp_keywords else "experience"

    # Section boost map based on JD focus
    if jd_focus == "skills":
        section_boost = {"Skills": 0.10, "Projects": 0.07, "Experience": 0.03}
    else:
        section_boost = {"Experience": 0.10, "Projects": 0.05, "Skills": 0.03}

    # ── Try FAISS vector search first ──
    try:
        import faiss
        import numpy as np
        import services.ml.model_hub as model_hub

        jd_emb = model_hub.embed_text(jd_text[:2000])
        if jd_emb is None:
            raise ValueError("JD embedding failed")

        # Embed all chunks
        chunk_texts = [c["text"] for c in chunks]
        chunk_embs = model_hub.embed_batch(chunk_texts)

        # Filter valid embeddings
        valid = [(i, emb) for i, emb in enumerate(chunk_embs) if emb is not None]
        if not valid:
            raise ValueError("No valid chunk embeddings")

        # Build FAISS index
        indices, embs = zip(*valid)
        matrix = np.vstack(embs).astype("float32")
        faiss.normalize_L2(matrix)

        dim = matrix.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(matrix)

        # Search
        query_vec = np.array([jd_emb], dtype="float32")
        faiss.normalize_L2(query_vec)
        k = min(top_k + 2, len(valid))  # fetch extras for re-ranking
        scores, faiss_indices = index.search(query_vec, k)

        # Re-rank with section boost
        scored = []
        for score, fi in zip(scores[0], faiss_indices[0]):
            if fi < 0:
                continue
            orig_idx = indices[fi]
            chunk = chunks[orig_idx]
            boosted_score = float(score) + section_boost.get(chunk.get("section", ""), 0)
            scored.append((boosted_score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        result = [c for s, c in scored if s >= threshold]

        # Ensure minimum 3 chunks
        if len(result) < 3 and scored:
            result = [c for _, c in scored[:3]]

        return result[:top_k]

    except (ImportError, ValueError, Exception) as e:
        pass  # Fall through to embedding fallback

    # ── Fallback: dot-product without FAISS ──
    try:
        import services.ml.model_hub as model_hub
        import numpy as np

        jd_emb = model_hub.embed_text(jd_text[:2000])
        if jd_emb is None:
            raise ValueError("JD embedding failed")

        chunk_texts = [c["text"] for c in chunks]
        chunk_embs = model_hub.embed_batch(chunk_texts)

        scored = []
        for chunk, emb in zip(chunks, chunk_embs):
            if emb is not None:
                sim = float(np.dot(jd_emb, emb))
                sim += section_boost.get(chunk.get("section", ""), 0)
                scored.append((sim, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        result = [c for sim, c in scored if sim >= threshold]
        if len(result) < 3 and scored:
            result = [c for _, c in scored[:3]]
        return result[:top_k]

    except (ImportError, Exception):
        pass

    # ── Final fallback: section priority ordering ──
    priority_order = ["Experience", "Skills", "Projects", "Education", "Summary", "Header"]
    sorted_chunks = sorted(chunks, key=lambda c: (
        priority_order.index(c["section"]) if c["section"] in priority_order else 99
    ))
    return sorted_chunks[:top_k]

