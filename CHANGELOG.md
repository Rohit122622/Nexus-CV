# Changelog

All notable changes to Nexus CV will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [2.0.0] — 2026-05-28

### Added
- **Multi-Agent AI Pipeline** — 4-agent system (SkillAgent, ExperienceAgent, ATSAgent, DecisionAgent) with ReAct reasoning loop
- **RAG Retrieval** — FAISS-based vector store with JD-aware semantic chunk prioritization
- **XGBoost ML Scoring** — Trained gradient boosting model with 7-feature vector for ATS prediction
- **Multi-Model LLM Fallback** — Gemini 2.5 Flash → Groq → DeepSeek → Qwen → Claude → Local rule-based
- **Thread-safe LLM Throttling** — 1.5s minimum interval between Gemini API calls
- **n8n Workflow Orchestration** — 11-node production pipeline with SplitInBatches and per-resume error handling
- **Bulk Resume Screening** — ZIP upload supporting up to 50 resumes with 5-signal hybrid ranking
- **Semantic Chunking** — Section-aware text splitting with BGE-large embeddings
- **Resume Builder AI Rewriting** — Gemini-powered bullet point enhancement and objective rewriting
- **Bias Filter** — Resume anonymization before agent evaluation
- **Agent Controller** — Backend orchestration layer for multi-agent pipeline
- **API Endpoints** — `/api/v1/analyze`, `/api/v1/compare`, `/api/v1/bulk-rank`, `/api/v1/score`, `/api/v1/health`
- **Embedding Cache** — Max 500 entries to prevent redundant computation

### Changed
- Upgraded scoring from simple keyword matching to hybrid XGBoost + heuristic
- Converted all debug `print()` statements to proper `logging` framework calls
- Restructured project into clean package architecture (`backend/`, `services/`, `utils/`)
- Moved n8n workflow to dedicated `n8n/` directory
- Comprehensive `.gitignore` with all runtime exclusions
- Professional README with complete documentation

### Security
- Input validation on all file uploads (type, size, content quality)
- Path traversal protection on PDF downloads
- CSRF protection on all form submissions
- Rate limiting on all analysis endpoints
- Security headers (X-Content-Type-Options, X-Frame-Options, CSP in production)

---

## [1.0.0] — 2026-02-10

### Added
- **Resume Analysis** — PDF upload with ATS scoring, skill detection, role prediction
- **Resume Builder** — Form-based resume generator with ATS optimization
- **Resume Comparison** — Side-by-side ATS score diff with skills analysis
- **Career Roadmap** — Personalized 6-month learning plan based on skill gaps
- **JD Matching** — Paste a job description for keyword match analysis
- **Email Reports** — Auto-email PDF reports via SendGrid
- **Auth System** — Local register/login + Google & Microsoft OAuth 2.0
- **Analysis History** — Track past analyses with downloadable reports
- **Dashboard** — Analytics overview with score tracking
- **Dark Mode** — Full dark/light theme toggle with glassmorphism design
- **PDF Generation** — ReportLab-powered analysis and comparison reports
- **Background Cleanup** — APScheduler job for periodic file cleanup
