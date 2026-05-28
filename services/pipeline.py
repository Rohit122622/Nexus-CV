"""
2-Stage Pipeline Architecture for Nexus CV.

PIPELINE 1 — DATA PROCESSING (No LLM calls):
  1. Extract ZIP/PDF files
  2. Extract text from PDFs
  3. Semantic chunking (section-aware)
  4. Generate embeddings (BGE model)
  Output: structured_chunks, embeddings, parsed_data

PIPELINE 2 — INTELLIGENCE (LLM-powered):
  1. SkillAgent (rule-based, NO LLM)
  2. ExperienceAgent (1 LLM call max)
  3. ATSAgent (XGBoost + heuristic, NO LLM)
  4. DecisionAgent (ReAct loop, 2 LLM iterations)
  5. Final Hybrid Score
  6. Ranking + Top-N selection
  Output: scores, reasoning, evidence, rankings
"""

import os
import sys
import time
import logging
import zipfile
import tempfile

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from services.processing.resume_parser import parse_resume


# ─────────────── PIPELINE 1: DATA PROCESSING ───────────────

class DataPipeline:
    """Pipeline 1: Extract, parse, chunk, embed. NO LLM calls."""

    def __init__(self):
        self._log = []

    def _add_log(self, step, status, duration_ms, detail=""):
        self._log.append({
            "pipeline": "data",
            "step": step,
            "status": status,
            "duration_ms": duration_ms,
            "detail": detail
        })
        logger.info("[P1:%s] %s in %dms -- %s", step, status, duration_ms, detail)

    def get_log(self):
        return list(self._log)

    # ── Step 1: Extract files from ZIP or PDF list ──
    def extract_files(self, file_paths):
        """
        Accept a list of file paths (PDFs or ZIPs).
        Returns list of PDF paths extracted.
        """
        t0 = time.time()
        pdf_paths = []
        for path in file_paths:
            if path.lower().endswith('.zip'):
                extracted = self._extract_zip(path)
                pdf_paths.extend(extracted)
            elif path.lower().endswith('.pdf'):
                pdf_paths.append(path)
            else:
                logger.warning("[P1] Skipping non-PDF/ZIP file: %s", path)

        self._add_log("extract", "success", int((time.time()-t0)*1000),
                      f"{len(pdf_paths)} PDFs extracted from {len(file_paths)} inputs")
        return pdf_paths

    def _extract_zip(self, zip_path):
        """Extract PDFs from a ZIP file."""
        pdfs = []
        try:
            extract_dir = tempfile.mkdtemp(prefix="nexus_")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for name in zf.namelist():
                    if name.lower().endswith('.pdf') and not name.startswith('__MACOSX'):
                        zf.extract(name, extract_dir)
                        pdfs.append(os.path.join(extract_dir, name))
        except Exception as e:
            logger.error("[P1] ZIP extraction failed: %s", e)
        return pdfs

    # ── Step 2: Parse PDFs to structured data ──
    def parse_resumes(self, pdf_paths):
        """Parse each PDF into structured data. Returns list of parsed dicts."""
        t0 = time.time()
        parsed_list = []
        for path in pdf_paths:
            try:
                parsed = parse_resume(path)
                parsed["_source_path"] = path
                parsed["_filename"] = os.path.basename(path)
                parsed_list.append(parsed)
            except Exception as e:
                logger.error("[P1] Parse failed for %s: %s", path, e)
                parsed_list.append({
                    "text": "", "skills": [], "word_count": 0,
                    "_source_path": path, "_filename": os.path.basename(path),
                    "_error": str(e)
                })

        self._add_log("parse", "success", int((time.time()-t0)*1000),
                      f"{len(parsed_list)} resumes parsed")
        return parsed_list

    # ── Step 3: Semantic chunking ──
    def chunk_resumes(self, parsed_list):
        """Create semantic chunks for each resume. Returns list of chunk lists."""
        t0 = time.time()
        try:
            from services.processing.semantic_chunker import chunk_text, get_priority_chunks
            chunker = "semantic"
        except ImportError:
            chunker = "legacy"

        all_chunks = []
        for parsed in parsed_list:
            text = parsed.get("text", "")
            if not text or len(text) < 50:
                all_chunks.append([])
                continue

            if chunker == "semantic":
                chunks = chunk_text(text)
            else:
                # Legacy word-based fallback
                words = text.split()
                chunks = []
                for i in range(0, len(words), 400):
                    chunk_words = words[max(0, i-50):i+400]
                    if len(chunk_words) >= 20:
                        chunks.append({
                            "text": " ".join(chunk_words),
                            "section": "Unknown",
                            "char_count": len(" ".join(chunk_words)),
                            "index": len(chunks)
                        })

            all_chunks.append(chunks)

        self._add_log("chunk", "success", int((time.time()-t0)*1000),
                      f"{sum(len(c) for c in all_chunks)} chunks from {len(parsed_list)} resumes ({chunker})")
        return all_chunks

    # ── Step 4: Generate embeddings ──
    def embed_resumes(self, parsed_list):
        """Generate embeddings for each resume text. Returns list of embeddings."""
        t0 = time.time()
        embeddings = []
        try:
            import services.ml.model_hub as model_hub
            for parsed in parsed_list:
                text = parsed.get("text", "")[:2000]
                if text:
                    emb = model_hub.embed_text(text)
                    embeddings.append(emb)
                else:
                    embeddings.append(None)
            self._add_log("embed", "success", int((time.time()-t0)*1000),
                          f"{sum(1 for e in embeddings if e is not None)} embeddings generated")
        except ImportError:
            embeddings = [None] * len(parsed_list)
            self._add_log("embed", "skipped", int((time.time()-t0)*1000),
                          "model_hub not available")
        except Exception as e:
            embeddings = [None] * len(parsed_list)
            self._add_log("embed", "error", int((time.time()-t0)*1000), str(e))

        return embeddings

    # ── Full Pipeline 1 execution ──
    def run(self, file_paths, jd_text=""):
        """
        Execute full data pipeline.
        Returns: {
            pdf_paths, parsed_list, all_chunks, embeddings,
            pipeline_log
        }
        """
        self._log = []
        pdf_paths = self.extract_files(file_paths)
        parsed_list = self.parse_resumes(pdf_paths)
        all_chunks = self.chunk_resumes(parsed_list)
        embeddings = self.embed_resumes(parsed_list)

        return {
            "pdf_paths": pdf_paths,
            "parsed_list": parsed_list,
            "all_chunks": all_chunks,
            "embeddings": embeddings,
            "pipeline_log": self.get_log()
        }


# ─────────────── PIPELINE 2: INTELLIGENCE ───────────────

class IntelligencePipeline:
    """Pipeline 2: Agent evaluation, scoring, ranking. Uses LLM."""

    def __init__(self):
        self._log = []

    def _add_log(self, step, status, duration_ms, detail=""):
        self._log.append({
            "pipeline": "intelligence",
            "step": step,
            "status": status,
            "duration_ms": duration_ms,
            "detail": detail
        })
        logger.info("[P2:%s] %s in %dms -- %s", step, status, duration_ms, detail)

    def get_log(self):
        return list(self._log)

    def score_single(self, parsed_data, chunks, jd_text, role):
        """
        Score a single resume through the 4-agent pipeline.
        Returns agent result dict.
        """
        from services.ai.agent_reasoner import run_agent

        t0 = time.time()
        result = run_agent(
            resume_text=parsed_data.get("text", ""),
            jd_text=jd_text,
            role=role,
            matched_skills=parsed_data.get("skills", []),
            ats_score=0,
            parsed_data=parsed_data
        )
        self._add_log("score", "success", int((time.time()-t0)*1000),
                      f"agent_score={result.get('agent_score', 0)}, "
                      f"multi_agent={result.get('multi_agent', False)}")
        return result

    def rank_candidates(self, scored_candidates, top_n=None):
        """
        Rank scored candidates by hybrid score.
        Returns sorted list with rank numbers.
        """
        t0 = time.time()
        # Sort by agent_score descending
        sorted_candidates = sorted(
            scored_candidates,
            key=lambda c: c.get("agent_score", 0),
            reverse=True
        )

        # Add rank
        for i, c in enumerate(sorted_candidates):
            c["rank"] = i + 1

        if top_n:
            sorted_candidates = sorted_candidates[:top_n]

        self._add_log("rank", "success", int((time.time()-t0)*1000),
                      f"Ranked {len(sorted_candidates)} candidates" +
                      (f" (top {top_n})" if top_n else ""))
        return sorted_candidates

    def run(self, data_output, jd_text, role="Software Engineer", top_n=None):
        """
        Execute full intelligence pipeline on Pipeline 1 output.
        Returns: {
            scored_candidates, rankings, pipeline_log
        }
        """
        self._log = []
        parsed_list = data_output["parsed_list"]
        all_chunks = data_output["all_chunks"]

        scored = []
        for i, (parsed, chunks) in enumerate(zip(parsed_list, all_chunks)):
            if parsed.get("_error"):
                scored.append({
                    "filename": parsed.get("_filename", f"resume_{i}"),
                    "agent_score": 0,
                    "confidence": "low",
                    "reason": f"Parse error: {parsed['_error']}",
                    "evidence": [],
                    "missing_skills": [],
                    "thought_trace": ["Parse error -- skipped"]
                })
                continue

            try:
                result = self.score_single(parsed, chunks, jd_text, role)
                result["filename"] = parsed.get("_filename", f"resume_{i}")
                scored.append(result)
            except Exception as e:
                logger.error("[P2] Scoring failed for %s: %s", parsed.get("_filename"), e)
                scored.append({
                    "filename": parsed.get("_filename", f"resume_{i}"),
                    "agent_score": 0,
                    "confidence": "low",
                    "reason": f"Scoring error: {str(e)[:100]}",
                    "evidence": [],
                    "missing_skills": [],
                    "thought_trace": [f"Error: {str(e)[:100]}"]
                })

        rankings = self.rank_candidates(scored, top_n)

        return {
            "scored_candidates": scored,
            "rankings": rankings,
            "total_processed": len(parsed_list),
            "pipeline_log": self.get_log()
        }


# ─────────────── FULL PIPELINE ORCHESTRATOR ───────────────

def run_full_pipeline(file_paths, jd_text, role="Software Engineer", top_n=None):
    """
    Run both pipelines end-to-end.

    Pipeline 1 (Data): Extract -> Parse -> Chunk -> Embed
    Pipeline 2 (Intelligence): Score -> Rank -> Top-N

    Returns combined result with both pipeline logs.
    """
    # Pipeline 1: Data Processing
    data_pipeline = DataPipeline()
    data_output = data_pipeline.run(file_paths, jd_text)

    # Pipeline 2: Intelligence
    intel_pipeline = IntelligencePipeline()
    intel_output = intel_pipeline.run(data_output, jd_text, role, top_n)

    return {
        "rankings": intel_output["rankings"],
        "total_processed": intel_output["total_processed"],
        "pipeline_1_log": data_output["pipeline_log"],
        "pipeline_2_log": intel_output["pipeline_log"],
        "data_summary": {
            "pdfs_extracted": len(data_output["pdf_paths"]),
            "chunks_total": sum(len(c) for c in data_output["all_chunks"]),
            "embeddings_generated": sum(1 for e in data_output["embeddings"] if e is not None),
        }
    }
