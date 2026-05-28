"""
Skill Registry — Single Source of Truth for all role-skill mappings.

All modules import from here. NO inline skill definitions elsewhere.
Used by: ats_scorer, multi_role_predictor, gemini_agent, bulk_screener, resume_builder.
"""

# ── Full role-skill mapping (15 roles) ──
ROLE_SKILL_MAP = {
    "Full Stack Developer": ["html", "css", "javascript", "react", "node.js", "express", "mongodb", "sql", "git", "docker", "aws", "typescript", "python"],
    "Data Scientist": ["python", "pandas", "numpy", "machine learning", "tensorflow", "pytorch", "sql", "data analysis", "statistics", "r"],
    "Web Developer": ["html", "css", "javascript", "react", "angular", "bootstrap", "sass", "webpack", "typescript", "vue"],
    "Software Engineer": ["java", "python", "c++", "data structures", "algorithms", "git", "sql", "c", "docker"],
    "Backend Developer": ["python", "java", "sql", "flask", "django", "api", "docker", "node.js", "postgresql", "redis"],
    "Frontend Developer": ["html", "css", "javascript", "react", "typescript", "vue", "angular", "sass", "webpack", "figma"],
    "Data Analyst": ["python", "sql", "pandas", "excel", "statistics", "tableau", "power bi", "data analysis"],
    "ML Engineer": ["python", "machine learning", "tensorflow", "pytorch", "numpy", "pandas", "docker", "mlops"],
    "DevOps Engineer": ["docker", "kubernetes", "aws", "ci/cd", "linux", "terraform", "ansible", "jenkins"],
    "Mobile Developer": ["android", "ios", "react native", "flutter", "kotlin", "swift", "java", "dart"],
    "Cloud Engineer": ["aws", "azure", "gcp", "docker", "terraform", "kubernetes", "linux", "networking"],
    "Cybersecurity Analyst": ["cybersecurity", "networking", "linux", "python", "firewalls", "siem", "penetration testing"],
    "Database Administrator": ["sql", "postgresql", "mysql", "oracle", "mongodb", "performance tuning", "backup"],
    "Product Manager": ["agile", "scrum", "roadmap", "data analysis", "stakeholder management", "user research", "jira"],
    "QA Engineer": ["testing", "selenium", "automation", "ci/cd", "python", "jmeter", "api testing"],
}

# ── Critical must-have skills per role (4 per role) ──
CRITICAL_SKILLS = {
    "Web Developer": ["html", "css", "javascript", "react"],
    "Backend Developer": ["python", "sql", "api", "docker"],
    "Data Analyst": ["python", "sql", "pandas", "statistics"],
    "ML Engineer": ["python", "machine learning", "tensorflow", "numpy"],
    "Software Engineer": ["data structures", "algorithms", "java", "python"],
    "DevOps Engineer": ["docker", "kubernetes", "aws", "ci/cd"],
    "Frontend Developer": ["html", "css", "javascript", "react"],
    "Full Stack Developer": ["javascript", "react", "node.js", "sql"],
    "Data Scientist": ["python", "machine learning", "statistics", "pandas"],
    "Mobile Developer": ["android", "ios", "react native", "kotlin"],
    "Cloud Engineer": ["aws", "azure", "docker", "terraform"],
    "Cybersecurity Analyst": ["cybersecurity", "networking", "linux", "python"],
    "Database Administrator": ["sql", "postgresql", "mysql", "performance tuning"],
    "Product Manager": ["agile", "roadmap", "data analysis", "stakeholder management"],
    "QA Engineer": ["testing", "selenium", "automation", "ci/cd"],
}

# ── Role skill scope descriptions for LLM prompts ──
ROLE_SKILL_SCOPE = {
    "Software Engineer": "data structures, algorithms, Python, Java, C++, Go, system design, Git, Docker, REST APIs, SQL, testing, CI/CD",
    "Web Developer": "HTML, CSS, JavaScript, TypeScript, React, Vue.js, Angular, Node.js, REST APIs, responsive design, Git, Webpack, accessibility",
    "Backend Developer": "Python, Java, Node.js, Go, SQL, PostgreSQL, MongoDB, Redis, Docker, REST APIs, microservices, message queues",
    "Frontend Developer": "HTML, CSS, JavaScript, TypeScript, React, Vue.js, Angular, Redux, Tailwind CSS, Webpack, Vite, testing, accessibility",
    "Full Stack Developer": "JavaScript, TypeScript, React, Node.js, Python, SQL, MongoDB, Docker, REST APIs, Git, authentication",
    "Data Scientist": "Python, R, pandas, NumPy, scikit-learn, TensorFlow, PyTorch, statistics, machine learning, deep learning, SQL, data visualization, Jupyter",
    "Data Analyst": "SQL, Python, R, Excel, Tableau, Power BI, pandas, statistics, data visualization, data cleaning, A/B testing",
    "ML Engineer": "Python, TensorFlow, PyTorch, scikit-learn, MLOps, Docker, Kubernetes, model serving, ONNX, MLflow, data pipelines, feature engineering",
    "DevOps Engineer": "Docker, Kubernetes, AWS, Azure, GCP, Terraform, CI/CD, Jenkins, GitHub Actions, Linux, Ansible, monitoring, Prometheus",
    "Cloud Engineer": "AWS, Azure, GCP, Terraform, CloudFormation, Docker, Kubernetes, serverless, IAM, networking, security",
    "Mobile Developer": "Android, iOS, Kotlin, Swift, React Native, Flutter, Dart, mobile UI/UX, REST APIs, Firebase, app deployment",
    "Cybersecurity Analyst": "cybersecurity, networking, Linux, Python, SIEM, penetration testing, vulnerability assessment, firewalls, IDS/IPS, compliance",
    "Database Administrator": "SQL, PostgreSQL, MySQL, Oracle, MongoDB, query optimization, indexing, replication, backup, performance tuning",
    "Product Manager": "Agile, Scrum, roadmap, stakeholder management, data analysis, user research, A/B testing, OKRs, Jira, communication",
    "QA Engineer": "testing, Selenium, Cypress, Playwright, API testing, CI/CD, test automation, performance testing, ISTQB, quality assurance",
}

# ── Non-skill words to always filter out ──
NON_SKILL_WORDS = {
    "engineer", "experience", "knowledge", "development", "software",
    "engineering", "skills", "tools", "systems", "platform", "management",
    "responsibility", "requirements", "qualifications", "team", "work",
    "ability", "understanding", "proficiency", "familiarity", "working",
    "strong", "excellent", "good", "basic", "advanced",
}

# Standard resume sections for completeness scoring
STANDARD_SECTIONS = ["Education", "Experience", "Skills", "Projects", "Summary"]
