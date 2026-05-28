
NEXUS CV — AI-POWERED RESUME ANALYSIS & RECRUITMENT INTELLIGENCE SYSTEM

Final Year Project Report
Department of Computer Science and Engineering


========================================
1. ABSTRACT
========================================

Nexus CV is an AI-powered resume analysis and recruitment intelligence platform that automates the hiring pipeline from resume upload to candidate ranking. The system combines Machine Learning models (XGBoost, Sentence Transformers), a multi-agent AI reasoning pipeline (SkillAgent, ExperienceAgent, ATSAgent, DecisionAgent), and Retrieval-Augmented Generation (RAG) to deliver accurate, explainable, and bias-filtered candidate evaluations.

The platform supports single resume analysis, resume-vs-resume comparison, AI-assisted resume generation, and bulk screening of up to 50 candidates via ZIP upload. A multi-model LLM fallback chain (Gemini 2.5 Flash → Gemini 2.5 Flash Lite → Groq Llama-3.3-70b) ensures 100% uptime. Workflow orchestration is handled by n8n, which coordinates a two-stage pipeline: data processing (no LLM) followed by intelligence scoring (LLM-powered). Final candidate ranking uses a 5-signal hybrid formula combining semantic similarity, ATS score, agent reasoning, skill match, and XGBoost prediction.

Keywords: Resume Analysis, ATS Scoring, XGBoost, Multi-Agent AI, RAG, FAISS, n8n, LLM Fallback


========================================
2. INTRODUCTION
========================================

2.1 Problem Statement

Traditional hiring relies heavily on manual resume screening, where recruiters spend an average of 6-7 seconds per resume. This leads to inconsistent evaluations, unconscious bias, and missed qualified candidates. With companies receiving hundreds of applications per opening, manual screening becomes a bottleneck that delays hiring and increases costs.

2.2 Need for ATS Systems

Applicant Tracking Systems (ATS) automate initial resume filtering by matching keywords against job descriptions. However, conventional ATS tools use simple keyword matching, which misses semantic relationships (e.g., "ML" vs "Machine Learning") and fails to evaluate candidate quality holistically.

2.3 Role of AI and ML in Recruitment

Modern AI enables semantic understanding of resumes through transformer-based embeddings, structured scoring through gradient-boosted models like XGBoost, and explainable decision-making through multi-agent reasoning. These technologies transform recruitment from keyword-counting to genuine candidate understanding.

2.4 Objective of Nexus CV

Nexus CV aims to build a production-grade, AI-powered recruitment platform that:
- Provides accurate ATS scoring using ML models (not just keyword matching)
- Evaluates candidates through a 4-agent reasoning pipeline with evidence-based decisions
- Supports bulk screening with automated ranking using a 5-signal hybrid formula
- Ensures reliability through multi-model LLM fallback and n8n workflow orchestration
- Eliminates bias through anonymized resume processing


========================================
3. SYSTEM OVERVIEW
========================================

3.1 End-to-End Flow

Step 1: User uploads a PDF resume (or ZIP of multiple resumes) through the web interface.
Step 2: The Flask backend parses the PDF, extracting text, skills, sections, and metadata.
Step 3: For single analysis — the ATS scorer computes a score using XGBoost + heuristics, the semantic chunker creates section-aware chunks, and AI agents evaluate the candidate.
Step 4: For bulk screening — the file is sent to n8n via webhook, which orchestrates a two-stage pipeline (data processing → LLM scoring), ranks candidates, and returns results.
Step 5: Results are displayed on the dashboard with scores, matched/missing skills, evidence quotes, career recommendations, and downloadable PDF reports.

3.2 High-Level Architecture

The system consists of five layers:
- Presentation Layer: HTML/CSS/JS frontend with responsive design and dark mode
- Application Layer: Flask backend with route handlers, input validation, and CSRF protection
- Intelligence Layer: Multi-agent pipeline (Skill → Experience → ATS → Decision agents)
- ML Layer: XGBoost scoring, BGE-large embeddings, FAISS vector store, semantic chunking
- Orchestration Layer: n8n workflow for bulk pipeline automation


========================================
4. TECHNOLOGY STACK
========================================

4.1 Frontend
- HTML5 for semantic page structure
- CSS3 (61KB stylesheet) with glassmorphism effects, gradient backgrounds, smooth animations, and full responsive design
- JavaScript for dynamic interactions, theme switching, file upload handling, and progress indicators

4.2 Backend
- Python 3.x with Flask web framework
- Flask extensions: Flask-Limiter (rate limiting), Flask-WTF (CSRF protection), Authlib (OAuth with Google and Microsoft)
- SQLite database for user accounts and analysis history
- Rotating file-based logging (app.log and error.log)

4.3 Machine Learning
- XGBoost (XGBRegressor) for ATS score prediction using a 7-feature vector
- Sentence Transformers (BAAI/bge-large-en-v1.5) for semantic embeddings with fallback to all-MiniLM-L6-v2
- BART-large-MNLI for zero-shot role classification

4.4 AI Models (LLM)
- Primary: Gemini 2.5 Flash — high-quality reasoning with thread-safe throttling
- Secondary: Gemini 2.5 Flash Lite — lighter model for rate-limit situations
- Fallback: Groq (llama-3.3-70b-versatile) — fast inference when Google quota is exhausted
- Additional fallbacks: DeepSeek, Qwen, Claude (safety nets)
- Final safety net: Local rule-based response (never crashes)

4.5 Vector Storage
- FAISS (Facebook AI Similarity Search) for inner-product-based vector retrieval
- Used in RAG store and semantic chunk retrieval

4.6 Automation
- n8n (self-hosted workflow engine) for bulk resume pipeline orchestration
- 11-node production workflow with error handling and webhook-based communication


========================================
5. SYSTEM ARCHITECTURE
========================================

5.1 Component Interaction

The system follows a layered architecture where each component has a specific responsibility:

Frontend UI → Sends HTTP requests to Flask backend
Flask Backend (app.py) → Routes requests to appropriate service modules
Resume Parser → Extracts text, skills, sections from PDFs using pdfplumber
ATS Scorer → Computes scores using XGBoost model + heuristic rules
Agent Controller → Orchestrates the multi-agent evaluation pipeline
Multi-LLM Service → Provides LLM calls with automatic fallback chain
RAG Store → Supplies contextual knowledge for improved LLM prompts
n8n Workflow → Orchestrates bulk processing via webhook integration

5.2 Two-Stage Pipeline Architecture

Pipeline 1 — Data Processing (No LLM calls):
  1. Extract ZIP/PDF files
  2. Parse PDFs to structured data (text, skills, sections)
  3. Semantic chunking (section-aware boundaries)
  4. Generate BGE embeddings

Pipeline 2 — Intelligence (LLM-powered):
  1. SkillAgent evaluates skill match (rule-based, no LLM)
  2. ExperienceAgent assesses experience quality (1 LLM call max)
  3. ATSAgent computes ATS compatibility (XGBoost + heuristic, no LLM)
  4. DecisionAgent synthesizes all signals (ReAct loop, 2 LLM iterations)
  5. Hybrid score computation and ranking


========================================
6. MACHINE LEARNING COMPONENTS
========================================

6.1 ATS Scoring Model

The ATS scorer (ats_scorer.py) computes a 0-100 score based on three components:

Skill Score (0-50): Measures overlap between resume skills and role-required skills using both exact string matching and semantic embedding similarity (cosine > 0.70 threshold).

Keyword Score (0-30): Measures density of role-specific keywords found in the full resume text, covering both ROLE_SKILL_MAP and CRITICAL_SKILLS registries.

Completeness Score (0-20): Checks for five standard resume sections — Education, Experience, Projects, Skills, and Summary — awarding 4 points each.

Logical consistency rules ensure that resumes with matched skills score at least 40, and those with 3+ matches score at least 50.

6.2 XGBoost Model

XGBoost (Extreme Gradient Boosting) is used as an ML-based scoring adjustment layer. It operates on a 7-feature vector:
  1. skill_match_ratio (0.0-1.0)
  2. keyword_density (role keywords / total words)
  3. section_completeness (sections found / 5)
  4. experience_years (normalized, capped at 15)
  5. has_education (binary)
  6. bullet_count (normalized, capped at 30)
  7. word_count (normalized, capped at 600)

The model is trained on synthetic data with realistic feature distributions and provides a confidence-weighted adjustment to the heuristic score. Feature importance is computed per-prediction for explainability.

6.3 Semantic Matching

The system uses BAAI/bge-large-en-v1.5 sentence transformer to encode resume text and job descriptions into dense vector embeddings. Similarity is computed via normalized dot product (cosine similarity).

Applications:
- Skill matching: Detects semantic equivalences (e.g., "JS" matches "JavaScript")
- Resume-JD similarity: Measures overall alignment for bulk ranking
- Chunk retrieval: Selects most relevant resume sections for LLM prompts
- Resume comparison: Computes semantic similarity between two resume versions

An embedding cache (max 500 entries) prevents redundant computation.


========================================
7. AI AGENT SYSTEM
========================================

7.1 Multi-Agent Pipeline

Nexus CV uses a 4-agent pipeline where each agent is a specialist:

SkillAgent (No LLM): Evaluates skill match by comparing parsed resume skills against the role skill registry. Produces a 0-10 skill score with lists of matched and missing skills.

ExperienceAgent (1 LLM call max): Assesses work experience quality by analyzing years of experience, project count, and achievement depth. Uses LLM to evaluate qualitative aspects like leadership and impact.

ATSAgent (No LLM): Wraps the XGBoost + heuristic ATS scorer as an agent interface. Provides ATS score, breakdown, and confidence without any LLM calls.

DecisionAgent (2 LLM iterations): The final synthesizer that uses a ReAct (Reasoning + Acting) loop. It receives outputs from all three preceding agents and performs:
  - Iteration 1: Initial analysis comparing sub-scores, identifying conflicts
  - Iteration 2: Final assessment with mandatory evidence quotes from the resume

7.2 DecisionAgent Evidence Requirements

The DecisionAgent enforces strict evidence standards:
- Minimum 2 direct quotes from resume chunks must be included
- Evidence is validated against actual chunk text (not hallucinated)
- If LLM provides insufficient evidence, the system extracts quotes using regex patterns matching achievement statements and technical skills

7.3 Confidence Computation

Confidence is computed deterministically from signal strength:
- HIGH: Skill score >= 7 with 4+ matched skills AND experience score >= 6 with 3+ years
- MEDIUM: Any one strong signal present, or moderate scores across the board
- LOW: Weak signals across all dimensions

7.4 Bias Filtering

Before agent evaluation, resumes are anonymized using the bias filter module. Candidate names and identifying information are stripped, and text is processed as "Candidate" to prevent demographic bias in LLM evaluations.


========================================
8. RAG (RETRIEVAL-AUGMENTED GENERATION)
========================================

8.1 What is RAG

Retrieval-Augmented Generation combines information retrieval with LLM generation. Instead of relying solely on the LLM's training data, RAG retrieves relevant context from a knowledge base and injects it into the prompt, improving accuracy and reducing hallucination.

8.2 Implementation in Nexus CV

The RAG store (rag_store.py) uses FAISS to build a vector index from:
- Role-specific resume samples and JD samples (from data/rag/ directory)
- Skills taxonomy organized by category

At query time, the input text is embedded using BGE-large, and FAISS performs inner-product search to retrieve the top-k most similar documents (threshold > 0.2).

8.3 Semantic Chunking

The semantic chunker (semantic_chunker.py) performs section-aware chunking:
1. Detects resume sections using regex patterns (Skills, Experience, Projects, Education, Summary, Certifications, Awards)
2. Splits each section into chunks at paragraph boundaries (300-800 chars with 100-char overlap)
3. Tags each chunk with its section name

The get_priority_chunks function uses FAISS-based retrieval to select the most JD-relevant chunks:
- Embeds all chunks and the JD using BGE-large
- Builds a temporary FAISS index and performs vector search
- Applies section boost weights based on JD focus (skill-heavy vs experience-heavy)
- Returns top 5-8 chunks for LLM context


========================================
9. MULTI-MODEL LLM SYSTEM
========================================

9.1 Fallback Chain

The multi_llm.py service implements a strict priority-based fallback:

Priority 1: Gemini 2.5 Flash — Primary model, best quality
Priority 2: Gemini 2.5 Flash Lite — Lighter variant for quota situations
Priority 3: Groq (llama-3.3-70b-versatile) — Fast external fallback
Priority 4-6: DeepSeek, Qwen, Claude — Additional safety nets
Priority 7: Local rule-based response — Guaranteed to never crash

9.2 Rate Limit Handling

- Thread-safe throttling enforces a minimum 1.5-second interval between Gemini API calls using a threading lock
- Each model gets 2 retry attempts before moving to the next
- HTTP 429 (rate limit) errors immediately skip to the next model without retrying
- "Model not found" errors also skip immediately

9.3 Reliability Guarantees

The system is designed to NEVER crash from LLM failures. If all cloud providers are unavailable, the local rule-based fallback returns a structured response with generic suggestions. All LLM responses are parsed through safe_json_parse() which handles malformed JSON gracefully.


========================================
10. BULK RESUME PROCESSING
========================================

10.1 ZIP Upload and Processing

Users upload a ZIP file containing multiple PDF resumes (up to 50). The system:
1. Extracts all PDFs from the ZIP (excluding __MACOSX artifacts)
2. Deduplicates identical resumes by text hash
3. Parses each resume for structured data
4. Processes through the ML + Agent pipeline

10.2 5-Signal Hybrid Ranking Formula

When agents are enabled:

  Final Score = 0.25 x Semantic + 0.20 x ATS + 0.25 x Agent + 0.15 x Skill + 0.15 x XGBoost

When agents are unavailable:

  Final Score = 0.35 x Semantic + 0.25 x ATS + 0.20 x Skill + 0.20 x XGBoost

Where:
- Semantic = Cosine similarity between resume and JD embeddings (0-100)
- ATS = XGBoost + heuristic ATS score (0-100)
- Agent = Multi-agent DecisionAgent score (0-10, normalized to 0-1)
- Skill = Skill match percentage (0-100)
- XGBoost = XGBoost regression prediction (0-100)

10.3 Top-N Selection

Candidates are sorted by hybrid score in descending order. The top-N candidates (configurable, default 3) receive detailed AI explanations including evidence quotes, confidence levels, and missing skill analysis.


========================================
11. n8n WORKFLOW
========================================

11.1 Overview

n8n serves as the orchestration engine for bulk resume processing. It receives files from the Flask backend via webhook, processes them through a two-stage pipeline, and returns ranked results. The workflow is defined in n8n_bulk_resume_workflow.json.

11.2 Node-by-Node Explanation

Node 1 — Webhook Trigger:
Listens for POST requests at /webhook/bulk-resume. Configured with rawBody=true to handle multipart file uploads. Uses responseNode mode so the final response is sent by a dedicated response node.

Node 2 — Auth Guard:
A pass-through code node for request validation. Since only the trusted Flask backend calls this webhook, no API key validation is performed. This node exists as a placeholder for future authentication logic.

Node 3 — Extract Files:
Validates that binary file data is present in the incoming request. Handles different binary key names (n8n may use "file", "data", or other keys) by normalizing to a standard "file" key for downstream compatibility.

Node 4 — PDF Text Extract:
Uses n8n's built-in extractFromFile node to convert PDF binary data into plain text. This is the raw text extraction step before any cleaning or structuring.

Node 5 — Pipeline 1: Data Processing (No LLM):
A JavaScript code node that performs:
- Text cleaning (normalizing line breaks, removing non-printable characters)
- Section detection using regex patterns for Education, Experience, Skills, Projects, Summary
- Word count computation
- Forwarding job description, role, and top_n parameters from the webhook body

Node 6 — Split In Batches:
Loops through each resume one at a time (batchSize=1). This ensures sequential processing and prevents API rate limits. It has two outputs: one feeds into Pipeline 2 for scoring, and the other feeds into Rank and Shortlist when all batches are complete.

Node 7 — Pipeline 2: Score (LLM):
An HTTP Request node that calls the Flask backend's /api/v1/score endpoint with:
- resume_text: The cleaned resume text
- role: Target job role
- job_description: The JD text
- run_agents: true (enables multi-agent evaluation)
This is where all ML scoring and LLM-based agent reasoning happens on the Flask side.

Node 8 — Collect Result:
Extracts scoring results from the API response and structures them into a standardized format with ats_score, agent_score, confidence, reasoning, evidence, matched_skills, and missing_skills.

Node 9 — Rank and Shortlist:
After all resumes are scored, this node:
- Computes a hybrid score: ATS x 0.4 + Agent x 10 x 0.6
- Sorts candidates by hybrid score descending
- Applies top_n cutoff
- Assigns rank numbers and shortlisted flags

Node 10 — Build Response:
Constructs the final JSON response with status, total processed count, shortlisted count, candidate details, and pipeline metadata (models used, timestamp, orchestrator info).

Node 11 — Webhook Response:
Sends the final ranked results back to the Flask backend as a JSON response with HTTP 200 status.

Error Handling:
- Per-Resume Error Handler: Catches individual resume failures without crashing the entire pipeline. Failed resumes get score 0 and continue to the next batch.
- Global Error Handler: Catches pipeline-level failures and returns HTTP 500 with error details.


========================================
12. UI/UX DESIGN
========================================

12.1 Navigation Structure
The navbar provides links to: Home, Upload Resume, Resume Builder, Compare Resumes, Bulk Screen, History, and Dashboard. It includes theme toggle and user session controls.

12.2 Responsive Design
The CSS stylesheet implements full responsive design with media queries, glassmorphism card effects, gradient backgrounds, smooth hover animations, and a dark/light theme toggle.

12.3 Key Pages
- Home: Landing page with feature overview and call-to-action
- Dashboard: Shows total analyses, best score, last activity, and analysis history
- Upload: PDF upload with optional job description input
- Result: Detailed analysis with ATS score, skill breakdown, suggestions, and career roadmap
- Bulk Screen: ZIP upload with role selection, progress tracking, and ranked results table
- Resume Builder: Form-based resume generator with ATS optimization
- Compare: Side-by-side comparison of two resume versions


========================================
13. FEATURES
========================================

1. Resume Analysis: Upload a PDF and get detailed ATS scoring with skill matching, keyword density analysis, and section completeness evaluation.

2. Resume Generation: Build an ATS-optimized resume using a structured form with auto-refinement for the target role and downloadable PDF output.

3. Resume Comparison: Compare two versions of a resume side-by-side with ATS score diff, skills added/removed analysis, and AI-generated improvement summary.

4. Bulk Screening: Upload a ZIP of up to 50 resumes, specify a job description, and receive ranked candidates with evidence-based reasoning and confidence scores.

5. AI Suggestions: Personalized improvement recommendations based on missing skills, keyword gaps, and section completeness analysis.

6. Career Recommendations: Role-specific career roadmaps with skill gap analysis and learning paths.

7. Multi-Role Prediction: Automatic detection of suitable job roles based on the candidate's skill profile.

8. PDF Report Generation: Downloadable analysis reports with scoring breakdown, sent automatically via email.

9. OAuth Authentication: Login via Google or Microsoft accounts alongside traditional username/password registration.


========================================
14. ADVANTAGES
========================================

1. Faster Hiring: Automates resume screening that would take hours manually. Bulk screening of 50 resumes completes in minutes.

2. AI-Based Decisions: Multi-agent reasoning provides evidence-backed evaluations, not just keyword counts.

3. Scalable System: The n8n orchestration layer handles batch processing with per-resume error isolation.

4. Explainable Scoring: XGBoost feature importance and agent thought traces provide full transparency.

5. Reliability: The multi-model LLM fallback chain guarantees the system never crashes.

6. Bias Reduction: Resume anonymization before agent evaluation reduces unconscious bias.

7. Semantic Understanding: BGE-large embeddings capture meaning beyond exact keywords.


========================================
15. LIMITATIONS
========================================

1. External API Dependency: The system relies on Gemini and Groq APIs. Network issues degrade quality to rule-based fallback.

2. Model Rate Limits: Gemini enforces per-minute quotas. Heavy bulk processing may exhaust limits.

3. Structured Resume Requirement: Non-standard formatting, images, or tables may not parse correctly.

4. Synthetic Training Data: The XGBoost model is trained on synthetic data rather than real hiring outcomes.

5. Single-Language Support: Currently optimized for English-language resumes only.

6. Local Storage: SQLite may not scale for enterprise-level concurrent usage.


========================================
16. FUTURE ENHANCEMENTS
========================================

1. Cloud Deployment: Deploy on AWS/GCP with Docker + Kubernetes for horizontal scaling.

2. Database Migration: Move from SQLite to PostgreSQL or MongoDB for production-grade persistence.

3. Recruiter Dashboard: Admin panel for managing job postings, candidate pipelines, and screening criteria.

4. Real-Time Analytics: Live dashboards with hiring funnel metrics and time-to-hire analytics.

5. Multi-Language Support: Extend analysis to Hindi, Spanish, French, and other languages.

6. Interview Scheduling: Integrate calendar APIs for auto-scheduling with shortlisted candidates.

7. Fine-Tuned Models: Replace synthetic training with real hiring outcome data.

8. Mobile Application: Companion mobile app for on-the-go resume analysis.


========================================
17. CONCLUSION
========================================

Nexus CV demonstrates that modern AI and ML technologies can transform resume screening from a manual, biased, and time-consuming process into an automated, explainable, and scalable system. By combining XGBoost-based scoring, transformer embeddings, multi-agent reasoning, RAG-enhanced context, and n8n workflow orchestration, the platform delivers a comprehensive recruitment intelligence solution.

The 4-agent pipeline (Skill → Experience → ATS → Decision) ensures that candidate evaluation considers multiple dimensions with evidence-based reasoning, while the 5-signal hybrid scoring formula produces reliable rankings. The multi-model LLM fallback chain guarantees system reliability, and the n8n orchestration layer enables scalable bulk processing.

This project showcases the practical application of Machine Learning, Natural Language Processing, Large Language Models, and workflow automation in solving a real-world problem that affects millions of job seekers and recruiters worldwide.


========================================
REFERENCES
========================================

1. Chen, T. and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. ACM SIGKDD.
2. Reimers, N. and Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP.
3. Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.
4. Yao, S. et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. ICLR.
5. Google (2025). Gemini API Documentation. https://ai.google.dev/
6. Johnson, J. et al. (2019). Billion-scale similarity search with GPUs. IEEE Transactions on Big Data.
7. n8n GmbH (2024). n8n Documentation. https://docs.n8n.io/
8. Flask Documentation. https://flask.palletsprojects.com/
