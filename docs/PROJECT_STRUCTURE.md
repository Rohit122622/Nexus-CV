# Project Structure

Detailed explanation of the Nexus CV directory layout and key files.

---

## Root Files

| File | Purpose |
|------|---------|
| `run.py` | Application entry point — starts Flask on `localhost:5000` |
| `requirements.txt` | Python package dependencies |
| `.env.example` | Environment configuration template with placeholder values |
| `.gitignore` | Git exclusion rules for secrets, caches, and runtime files |
| `LICENSE` | MIT License |
| `README.md` | Project overview and documentation |
| `CONTRIBUTING.md` | Contribution guidelines |
| `CHANGELOG.md` | Version history |

---

## `backend/` — Flask Application Layer

The Flask web application, route handlers, and supporting infrastructure.

| File | Purpose |
|------|---------|
| `app.py` | Main Flask app (1400+ lines) — all routes, middleware, security headers, error handlers |
| `database.py` | SQLite database operations — user registration, analysis history |
| `agent_controller.py` | Orchestrates the multi-agent pipeline for API endpoints |
| `input_validator.py` | Request validation — resume text quality, JD validation, content sanitization |

---

## `services/` — Core Service Modules

### `services/ai/` — AI & LLM Services

| File | Purpose |
|------|---------|
| `multi_llm.py` | Central LLM caller with 6-provider fallback chain and thread-safe throttling |
| `gemini_agent.py` | Gemini-specific functions: resume rewriting, comparison, insights |
| `agent_reasoner.py` | Executes the 4-agent pipeline (Skill → Experience → ATS → Decision) |

### `services/ai/agents/` — Specialist Agents

| File | Purpose |
|------|---------|
| `skill_agent.py` | Rule-based skill matching against role registry (no LLM) |
| `experience_agent.py` | Experience quality assessment (1 LLM call max) |
| `ats_agent.py` | Wrapper around XGBoost + heuristic ATS scorer (no LLM) |
| `decision_agent.py` | Final synthesizer with ReAct loop (2 LLM iterations) |

### `services/ml/` — Machine Learning Components

| File | Purpose |
|------|---------|
| `ats_scorer.py` | ATS score calculation — XGBoost + heuristic hybrid with 7-feature vector |
| `model_hub.py` | Embedding model management (BGE-large, with MiniLM fallback) |
| `skill_registry.py` | Role-skill mappings, critical skills, non-skill word filters |
| `embedding_cache.py` | LRU cache for embedding results (max 500 entries) |

### `services/processing/` — Data Processing

| File | Purpose |
|------|---------|
| `resume_parser.py` | PDF text extraction via pdfplumber with section detection |
| `resume_builder.py` | AI resume builder — form validation, ATS refinement, PDF generation |
| `bulk_screener.py` | Bulk screening engine — ZIP extraction, parallel processing, ranking |
| `semantic_chunker.py` | Section-aware text chunking with FAISS-based priority retrieval |
| `career_recommender.py` | Career roadmap generation based on skill gaps |
| `jd_matcher.py` | Job description keyword matching |
| `multi_role_predictor.py` | Zero-shot multi-role classification |
| `resume_insights.py` | Resume quality analysis with AI enhancement |
| `resume_suggestions.py` | Personalized improvement recommendations |
| `pdf_generator.py` | Analysis report PDF generation |
| `compare_pdf_generator.py` | Comparison report PDF generation |
| `email_sender.py` | SendGrid API + SMTP fallback for email delivery |

---

## `utils/` — Shared Utilities

| File | Purpose |
|------|---------|
| `bias_filter.py` | Resume anonymization before agent evaluation |
| `cleanup.py` | Scheduled cleanup of old uploads and reports |
| `json_utils.py` | Safe JSON parsing for LLM responses |
| `pdf_utils.py` | Shared PDF styling constants and helpers |
| `rag_store.py` | FAISS vector store for RAG retrieval |
| `skill_normalizer.py` | Skill name normalization and deduplication |

---

## `model/` — Trained ML Models

| File | Purpose |
|------|---------|
| `ats_xgb.pkl` | Trained XGBoost regressor for ATS score prediction |

---

## `data/` — Reference Data

| File | Purpose |
|------|---------|
| `career_paths.json` | Career progression mappings for roadmap generation |
| `job_roles.json` | Role-to-skill definitions |
| `skills.txt` | Skill vocabulary for extraction |
| `skills_taxonomy.json` | Hierarchical skill categories |
| `rag/resume_samples.json` | Sample resumes for RAG knowledge base |
| `rag/jd_samples.json` | Sample job descriptions for RAG retrieval |

---

## `n8n/` — Workflow Definitions

| File | Purpose |
|------|---------|
| `bulk_resume_workflow.json` | 11-node n8n production workflow for bulk pipeline orchestration |

---

## `frontend/` — Presentation Layer

### `frontend/static/` — Assets

CSS, JavaScript, and SVG assets for the UI.

### `frontend/templates/` — Jinja2 Templates

17 HTML templates covering all application pages (home, dashboard, upload, results, bulk screening, resume builder, comparison, auth, error pages).

---

## Runtime Directories (Gitignored)

| Directory | Purpose |
|-----------|---------|
| `uploads/` | Temporary storage for uploaded resume PDFs |
| `reports/` | Generated analysis and comparison PDF reports |
| `logs/` | Application logs (`app.log`, `error.log`) |
