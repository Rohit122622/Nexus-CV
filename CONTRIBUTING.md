# Contributing to Nexus CV

Thank you for your interest in contributing to Nexus CV! This document provides guidelines for contributing to the project.

## Getting Started

### Development Setup

1. **Fork** the repository
2. **Clone** your fork:
   ```bash
   git clone https://github.com/your-username/Nexus-CV.git
   cd Nexus-CV
   ```
3. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```
4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
5. **Set up environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```
6. **Run the application:**
   ```bash
   python run.py
   ```

## How to Contribute

### Reporting Bugs

- Open an issue with a clear title and description
- Include steps to reproduce the bug
- Include expected vs. actual behavior
- Attach screenshots if applicable

### Suggesting Features

- Open an issue with the **enhancement** label
- Describe the feature and its use case
- Explain how it fits into the existing architecture

### Submitting Pull Requests

1. Create a **feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes following the code style guidelines below
3. Test your changes locally
4. Commit with clear, descriptive messages:
   ```bash
   git commit -m "feat: add resume language detection"
   ```
5. Push to your fork and open a Pull Request

## Code Style

- **Python**: Follow PEP 8 conventions
- **Imports**: Group by standard library → third-party → local, separated by blank lines
- **Docstrings**: Use triple-quote docstrings for all public functions
- **Logging**: Use `logging.getLogger(__name__)` — avoid `print()` statements
- **Error Handling**: Always handle exceptions gracefully; never let the app crash
- **Type Safety**: Validate all external inputs before processing

## Project Architecture

- **`backend/`** — Flask application, routes, database
- **`services/ai/`** — LLM integrations and agent system
- **`services/ml/`** — Machine learning models and scoring
- **`services/processing/`** — Data processing and PDF generation
- **`utils/`** — Shared utilities
- **`frontend/`** — Templates and static assets

## Important Guidelines

- **DO NOT** commit API keys or secrets — use `.env` for all credentials
- **DO NOT** modify the multi-model LLM fallback chain order without discussion
- **DO NOT** remove error fallbacks — the system must never crash from LLM failures
- Keep the n8n workflow JSON in sync with API endpoint changes
- All new routes must include rate limiting and input validation

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
