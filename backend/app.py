import os
import sys
import logging
import json
from logging.handlers import RotatingFileHandler

# ── Project root path setup ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from authlib.integrations.flask_client import OAuth
from flask import Flask, request, jsonify, render_template, flash
from flask import send_file, session, redirect, url_for
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_wtf.csrf import CSRFProtect
from datetime import datetime
from uuid import uuid4

# ── Local imports (new package structure) ──
from services.processing.pdf_generator import generate_pdf
from services.processing.resume_builder import validate_form_data, format_resume_data, build_parsed_data, refine_for_ats, generate_resume_pdf
from services.processing.resume_parser import parse_resume
from services.ml.ats_scorer import calculate_ats_score
from services.processing.career_recommender import recommend_career
from backend.database import connect_db, create_users_table, register_user, verify_user
from backend.database import create_analysis_table
from services.processing.jd_matcher import match_jd
from services.processing.email_sender import send_email
from services.processing.resume_insights import analyze_resume_insights
from services.processing.resume_suggestions import generate_suggestions
from services.processing.multi_role_predictor import predict_multiple_roles
from services.processing.compare_pdf_generator import generate_comparison_pdf
from backend.input_validator import validate_resume_text, validate_job_description, validate_content_quality

try:
    from services.ai import gemini_agent
except ImportError:
    gemini_agent = None

try:
    from services.processing import bulk_screener
except ImportError:
    bulk_screener = None

try:
    from backend.agent_controller import AgentController
    _agent = AgentController()
except ImportError:
    _agent = None

# ── Garbage Text Sanitizer ──
_GARBAGE_PATTERNS = ["system error", "exception", "traceback"]

def _sanitize_strings(data):
    """Remove any string containing garbage patterns from dicts/lists before rendering."""
    if isinstance(data, dict):
        return {k: _sanitize_strings(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_sanitize_strings(item) for item in data
                if not (isinstance(item, str) and any(g in item.lower() for g in _GARBAGE_PATTERNS))]
    elif isinstance(data, str):
        if any(g in data.lower() for g in _GARBAGE_PATTERNS):
            return "Try improving role-specific skills"
    return data


# ── Flask App ──
app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, "frontend", "templates"),
    static_folder=os.path.join(PROJECT_ROOT, "frontend", "static")
) 

# ProxyFix: trust X-Forwarded-* headers from reverse proxy
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# SECRET_KEY: require from env in production, fallback only for local dev
_env = os.getenv("FLASK_ENV", "development")
if _env == "production" and not os.getenv("SECRET_KEY"):
    raise RuntimeError("SECRET_KEY must be set in production!")
app.secret_key = os.getenv("SECRET_KEY", "dev-fallback-key-change-me")

# ── Session Security ──
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
if _env == "production":
    app.config["SESSION_COOKIE_SECURE"] = True

# ── CSRF Protection ──
csrf = CSRFProtect(app)

# ── Logging ──
os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)

file_handler = RotatingFileHandler(
    os.path.join(PROJECT_ROOT, "logs", "app.log"), maxBytes=5_000_000, backupCount=5
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
))

error_handler = RotatingFileHandler(
    os.path.join(PROJECT_ROOT, "logs", "error.log"), maxBytes=5_000_000, backupCount=3
)
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s\n%(pathname)s:%(lineno)d"
))

app.logger.addHandler(file_handler)
app.logger.addHandler(error_handler)
app.logger.setLevel(logging.INFO)
app.logger.info("Nexus CV starting up...")

# ── Rate Limiting ──
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://"
)

@app.errorhandler(500)
def handle_500_error(e):
    app.logger.error(f"Internal Server Error: {e}", exc_info=True)
    return render_template("500.html"), 500

@app.errorhandler(Exception)
def handle_exception(e):
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return e
    app.logger.error(f"Unhandled Exception: {e}", exc_info=True)
    return render_template("500.html"), 500

oauth = OAuth(app)

google = oauth.register(
    name='google',
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile'
    }
)

microsoft = oauth.register(
    name='microsoft',
    client_id=os.getenv("MICROSOFT_CLIENT_ID"),
    client_secret=os.getenv("MICROSOFT_CLIENT_SECRET"),
    authorize_url='https://login.microsoftonline.com/common/oauth2/v2.0/authorize',
    access_token_url='https://login.microsoftonline.com/common/oauth2/v2.0/token',
    api_base_url='https://graph.microsoft.com/v1.0/',
    client_kwargs={'scope': 'User.Read'},
)

UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "uploads")
REPORT_FOLDER = os.path.join(PROJECT_ROOT, "reports")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["REPORT_FOLDER"] = REPORT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# ── Initialize database tables (once) ──
create_users_table()
create_analysis_table()

# ------------------------
# WEBSITE ROUTES
# ------------------------

@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/login")

    conn = connect_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT role, ats_score, date, pdf_path FROM analysis WHERE username=? ORDER BY id DESC",
        (session["user"],)
    )
    records = cur.fetchall()

    total = len(records)
    best = max((r[1] for r in records), default=0)
    last = records[0][2] if records else None

    conn.close()

    stats = {
        "total_analyzed": total,
        "best_score": best,
        "last_activity": last
    }

    return render_template("dashboard.html", stats=stats, records=records)

@app.route("/upload-page", methods=["GET"])
def upload_page():
    if "user" not in session:
        return redirect("/login")
    return render_template("upload.html")

# ── Safe Template Data Helper ──
def _safe_template_data(**kwargs):
    """Replace None values with safe defaults before passing to templates."""
    safe = {}
    for key, value in kwargs.items():
        if value is None:
            if key in ("matched_skills", "missing_skills", "suggestions", "roles"):
                safe[key] = []
            elif key in ("ats_score", "match_percentage"):
                safe[key] = 0
            elif key in ("role",):
                safe[key] = "Software Engineer"
            elif key in ("roadmap", "jd_result", "insights", "ats_breakdown",
                         "missing_classified", "validation_summary"):
                safe[key] = {}
            else:
                safe[key] = ""
        else:
            safe[key] = value
    return safe


@app.route("/analyze", methods=["POST"])
@limiter.limit("10/minute")
def analyze():
    if "user" not in session:
        return redirect("/login")

    try:
        import traceback

        file = request.files.get("resume")
        if not file or not file.filename:
            flash("Please upload a resume file.", "error")
            return redirect("/upload-page")

        # ── File type validation ──
        if not file.filename.lower().endswith(".pdf"):
            flash("Only PDF resume files are allowed.", "error")
            return redirect("/upload-page")

        # ── File size validation (5 MB limit) ──
        file.seek(0, 2)
        file_size = file.tell()
        file.seek(0)
        if file_size > 5 * 1024 * 1024:
            flash("File size must be under 5 MB.", "error")
            return redirect("/upload-page")

        path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(path)

        parsed_data = parse_resume(path)
        resume_text = parsed_data.get("text", "")

        # ── Resume text validation ──
        is_valid_resume, resume_err = validate_resume_text(resume_text)
        if not is_valid_resume:
            flash(resume_err, "error")
            return redirect("/upload-page")

        # ── Content quality validation ──
        quality = validate_content_quality(resume_text)
        if not quality["is_valid"]:
            flash(quality["reason"], "error")
            return redirect("/upload-page")

        # ── JD validation (if provided) ──
        job_description = request.form.get("job_description", None)
        if job_description:
            jd_valid, jd_err = validate_job_description(job_description)
            if not jd_valid:
                flash(jd_err, "error")
                return redirect("/upload-page")

        # ── Build validation summary ──
        validation_summary = {
            "resume_detected": True,
            "sections_found": quality.get("sections_found", []),
            "jd_detected": bool(job_description),
            "quality_score": quality.get("confidence_score", 0)
        }

        # MULTI-ROLE PREDICTION
        multi_roles = predict_multiple_roles(parsed_data.get("skills", []))
        if not multi_roles:
            multi_roles = [{"role": "Software Engineer", "score": 50, "reason": "Default prediction"}]

        primary_role = multi_roles[0]["role"]
        role = primary_role
        ats_result = calculate_ats_score(parsed_data, role)
        # Sanitize all ATS output before rendering
        ats_result = _sanitize_strings(ats_result)

        # Now compute insights with ATS data for Gemini AI enhancement
        insights = analyze_resume_insights(
            resume_text, parsed_data.get("skills", []),
            ats_data=ats_result, role=role,
            jd_text=job_description
        )

        jd_result = {
            "match_percentage": 0,
            "matched_skills": [],
            "missing_keywords": []
        }
        if job_description:
            jd_result = match_jd(parsed_data.get("skills", []), job_description)
            jd_result = _sanitize_strings(jd_result)

        career = recommend_career(ats_result, role, parsed_data.get("skills", []), insights=insights, jd_result=jd_result)

        suggestions = generate_suggestions(insights, ats_result.get("missing_skills", []), jd_result)
        suggestions = _sanitize_strings(suggestions)
        if not suggestions:
            suggestions = ["Your resume is well optimized. Minor improvements can make it even stronger."]

        pdf_filename = f"analysis_{session['user']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex}.pdf"
        pdf_path = os.path.join(app.config["REPORT_FOLDER"], pdf_filename)

        try:
            generate_pdf(
                {
                    "role": role,
                    "ats_score": ats_result.get("ats_score", 10),
                    "matched_skills": ats_result.get("matched_skills", []),
                    "missing_skills": ats_result.get("missing_skills", []),
                    "missing_classified": ats_result.get("missing_classified", {}),
                    "roadmap": career.get("roadmap", {}),
                    "ats_breakdown": {
                        "skill_score": ats_result.get("skill_score", 0),
                        "keyword_score": ats_result.get("keyword_score", 0),
                        "completeness_score": ats_result.get("completeness_score", 0)
                    },
                    "jd_result": jd_result,
                    "insights": insights,
                    "suggestions": suggestions
                },
                pdf_path
            )
        except Exception as pdf_err:
            app.logger.error(f"PDF generation failed: {pdf_err}")
            pdf_path = None

        try:
            if pdf_path:
                send_email(session.get("email", ""), pdf_path)
        except Exception as email_err:
            app.logger.error(f"Email send failed: {email_err}")

        if pdf_path:
            session["last_pdf"] = pdf_path

        try:
            conn = connect_db()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO analysis (username, role, ats_score, date, pdf_path) VALUES (?, ?, ?, ?, ?)",
                (
                    session["user"],
                    role,
                    ats_result.get("ats_score", 10),
                    datetime.now().strftime("%d-%m-%Y %H:%M"),
                    pdf_path or ""
                )
            )
            conn.commit()
            conn.close()
        except Exception as db_err:
            app.logger.error(f"Database insert failed: {db_err}")

        # ── Final Sanity Filter Before Render ──
        matched_skills = ats_result.get("matched_skills", [])
        matched_skills = [
            s for s in matched_skills
            if isinstance(s, str) and len(s.split()) <= 2 and len(s) < 25
        ]
        ats_result["matched_skills"] = matched_skills

        # Fix template key mismatch
        missing_classified = ats_result.get("missing_classified", {})
        if "nice_to_have" in missing_classified:
            missing_classified["optional"] = missing_classified.pop("nice_to_have")

        session["last_result"] = {
            "role": role,
            "roles": multi_roles,
            "ats_score": ats_result.get("ats_score", 10),
            "matched_skills": matched_skills,
            "missing_skills": ats_result.get("missing_skills", []),
            "missing_classified": missing_classified,
            "roadmap": career.get("roadmap", {}),
            "ats_breakdown": {
                "skill_score": ats_result.get("skill_score", 0),
                "keyword_score": ats_result.get("keyword_score", 0),
                "completeness_score": ats_result.get("completeness_score", 0)
            }
        }

        template_data = _safe_template_data(
            role=role,
            roles=multi_roles,
            ats_score=ats_result.get("ats_score", 10),
            matched_skills=matched_skills,
            missing_skills=ats_result.get("missing_skills", []),
            missing_classified=missing_classified,
            roadmap=career.get("roadmap", {}),
            jd_result=jd_result,
            insights=insights,
            suggestions=suggestions,
            ats_breakdown={
                "skill": ats_result.get("skill_score", 0),
                "keyword": ats_result.get("keyword_score", 0),
                "completeness": ats_result.get("completeness_score", 0)
            },
            validation_summary=validation_summary
        )

        return render_template("result.html", **template_data)

    except Exception as e:
        import traceback
        app.logger.error(f"ANALYZE ROUTE ERROR: {e}\n{traceback.format_exc()}")
        flash("An error occurred during analysis. Please try again.", "error")
        return redirect("/upload-page")

@app.route("/compare-analyze", methods=["POST"])
@limiter.limit("5/minute")
def compare_analyze():
    if "user" not in session:
        return redirect("/login")

    resume1 = request.files.get("resume_v1")
    resume2 = request.files.get("resume_v2")

    if not resume1 or not resume2:
        return redirect("/compare-resume")

    try:
        import traceback

        # ── File type validation ──
        for f, label in [(resume1, "Resume V1"), (resume2, "Resume V2")]:
            if not f.filename.lower().endswith(".pdf"):
                flash(f"{label}: Only PDF resume files are allowed.", "error")
                return redirect("/compare-resume")
            f.seek(0, 2)
            fsize = f.tell()
            f.seek(0)
            if fsize > 5 * 1024 * 1024:
                flash(f"{label}: File size must be under 5 MB.", "error")
                return redirect("/compare-resume")

        from uuid import uuid4

        path1 = os.path.join(
            app.config["UPLOAD_FOLDER"],
            f"v1_{uuid4().hex}_{resume1.filename}"
        )
        path2 = os.path.join(
            app.config["UPLOAD_FOLDER"],
            f"v2_{uuid4().hex}_{resume2.filename}"
        )

        resume1.save(path1)
        resume2.save(path2)

        parsed1 = parse_resume(path1)
        parsed2 = parse_resume(path2)

        # ── Resume text validation (both files) ──
        for parsed, label in [(parsed1, "Resume V1"), (parsed2, "Resume V2")]:
            is_valid, err_msg = validate_resume_text(parsed.get("text", ""))
            if not is_valid:
                flash(f"{label}: {err_msg}", "error")
                return redirect("/compare-resume")
            q_result = validate_content_quality(parsed.get("text", ""))
            if not q_result["is_valid"]:
                flash(f"{label}: {q_result['reason']}", "error")
                return redirect("/compare-resume")

        multi_roles = predict_multiple_roles(parsed2.get("skills", []))
        role = multi_roles[0]["role"] if multi_roles else "Software Engineer"

        ats1 = calculate_ats_score(parsed1, role)
        ats2 = calculate_ats_score(parsed2, role)

        job_description = request.form.get("job_description")
        jd1 = jd2 = None

        # ── JD validation (if provided) ──
        if job_description:
            jd_valid, jd_err = validate_job_description(job_description)
            if not jd_valid:
                flash(jd_err, "error")
                return redirect("/compare-resume")
            jd1 = _sanitize_strings(match_jd(parsed1.get("skills", []), job_description))
            jd2 = _sanitize_strings(match_jd(parsed2.get("skills", []), job_description))

        from services.ml.ats_scorer import ROLE_SKILL_MAP
        required_skills = set(s.lower() for s in ROLE_SKILL_MAP.get(role, []))
        
        # safely extract JD words for skill comparison constraint
        import re
        jd_skills = set(re.findall(r'\b\w+\b', job_description.lower())) if job_description else set()
        
        skills1 = set(s.lower() for s in parsed1.get("skills", []))
        skills2 = set(s.lower() for s in parsed2.get("skills", []))

        # Build AI summary based on actual diff data
        change = ats2.get("ats_score", 0) - ats1.get("ats_score", 0)
        added = list(skills2 - skills1)
        
        core_skills = {"python", "sql", "java", "html", "css", "javascript"}
        removed_raw = [
            s for s in skills1 
            if s not in skills2 
            and s not in required_skills
            and s not in jd_skills
            and s not in core_skills
        ]
        
        removed = [s for s in removed_raw if len(s.split()) <= 2]

        app.logger.debug("COMPARE skills1=%s", sorted(skills1))
        app.logger.debug("COMPARE skills2=%s", sorted(skills2))
        app.logger.debug("COMPARE removed_raw=%s removed_final=%s", removed_raw, removed)

        if change > 0:
            ai_summary = (
                f"Your updated resume shows a +{change} point ATS improvement for the {role} role. "
            )
            if added:
                ai_summary += f"Skills gained ({', '.join(added[:4])}) strengthened your profile. "
            if removed:
                ai_summary += f"Note: you removed {', '.join(removed[:3])} — verify these aren't critical for your target role. "
            ai_summary += "Continue optimizing keyword coverage and quantifying achievements."
        elif change < 0:
            ai_summary = (
                f"Your ATS score decreased by {abs(change)} points for the {role} role. "
            )
            if removed:
                ai_summary += f"Removing {', '.join(removed[:3])} likely contributed to the decline. "
            if added:
                ai_summary += f"Despite adding {', '.join(added[:3])}, the net impact was negative. "
            ai_summary += "Consider restoring key role-specific skills and improving keyword density."
        else:
            ai_summary = (
                f"Both resume versions score equally for the {role} role. "
            )
            if added and removed:
                ai_summary += f"You swapped {', '.join(removed[:2])} for {', '.join(added[:2])}. "
            ai_summary += "Try adding quantified achievements and role-specific keywords to improve."

        comparison = {
            "role": role,
            "ats_before": ats1.get("ats_score", 0),
            "ats_after": ats2.get("ats_score", 0),
            "ats_change": change,
            "skills_added": added,
            "skills_removed": removed,
            "jd_before": jd1,
            "jd_after": jd2,
            "ai_summary": ai_summary
        }

        # ── Gemini AI Comparison (optional) ──
        ai_comparison = None
        if gemini_agent:
            try:
                ai_comparison = gemini_agent.compare_resumes(
                    resume_a_text=parsed1.get("text", ""),
                    resume_b_text=parsed2.get("text", ""),
                    ats_a=ats1,
                    ats_b=ats2,
                    role=role,
                    jd_text=job_description
                )
            except Exception:
                ai_comparison = None

        # Generate comparison PDF (always — needed for download)
        cmp_pdf_name = f"compare_{session['user']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex}.pdf"
        cmp_pdf_path = os.path.join(app.config["REPORT_FOLDER"], cmp_pdf_name)
        try:
            generate_comparison_pdf(comparison, cmp_pdf_path)
            session["compare_pdf_path"] = cmp_pdf_path
        except Exception as pdf_err:
            app.logger.error(f"Compare PDF generation failed: {pdf_err}")

        # Auto-email comparison report
        try:
            change_val = comparison["ats_change"]
            if change_val > 0:
                change_msg = f"Your ATS score improved by {change_val} points."
            elif change_val < 0:
                change_msg = f"Your ATS score decreased by {abs(change_val)} points."
            else:
                change_msg = "Your ATS score remained unchanged."

            send_email(
                session.get("email", ""),
                cmp_pdf_path,
                subject="Your Resume Comparison Report \u2014 Nexus CV",
                body=(
                    f"Hello,\n\n"
                    f"Your resume comparison report for the {role} role is attached.\n\n"
                    f"ATS Score: {comparison['ats_before']} \u2192 {comparison['ats_after']}\n"
                    f"{change_msg}\n\n"
                    f"Review the attached PDF for full details on skills added, "
                    f"removed, and overall improvement.\n\n"
                    f"Best regards,\nNexus CV Team"
                ),
                attachment_name="Resume_Comparison_Report.pdf"
            )
        except Exception as e:
            app.logger.error(f"Failed to send comparison email: {e}")

        return render_template("compare_result.html", comparison=comparison, ai_comparison=ai_comparison)

    except Exception as e:
        import traceback
        app.logger.error(f"COMPARE ROUTE ERROR: {e}\n{traceback.format_exc()}")
        flash("An error occurred during comparison. Please try again.", "error")
        return redirect("/compare-resume")


@app.route("/download-compare-report")
def download_compare_report():
    if "compare_pdf_path" not in session:
        return redirect("/")
    return send_file(session["compare_pdf_path"], as_attachment=True)

@app.route("/download-pdf")
def download_pdf():
    if "last_pdf" not in session:
        return redirect("/")

    return send_file(session["last_pdf"], as_attachment=True)


@app.route("/history")
def history():
    if "user" not in session:
        return redirect("/login")

    conn = connect_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT role, ats_score, date, pdf_path FROM analysis WHERE username=? ORDER BY id DESC",
        (session["user"],)
    )
    records = cur.fetchall()
    conn.close()

    return render_template("history.html", records=records)

# ------------------------
# API ROUTES (Postman)
# ------------------------
@app.route("/compare-resume", methods=["GET"])
def compare_resume():
    if "user" not in session:
        return redirect("/login")

    return render_template("compare.html")

@app.route("/match", methods=["POST"])
def match():
    data = request.json
    return jsonify(calculate_ats_score({"skills": data["skills"], "text": " ".join(data["skills"])}, data["job_role"]))


@app.route("/predict-role", methods=["POST"])
def predict():
    data = request.json
    roles = predict_multiple_roles(data["skills"])
    return jsonify({"predicted_role": roles[0]["role"] if roles else "Unknown"})


@app.route("/ats-score", methods=["POST"])
def ats_score():
    data = request.json
    return jsonify(calculate_ats_score(data["parsed_data"], data["job_role"]))


@app.route("/career-recommendation", methods=["POST"])
def career():
    data = request.json
    return jsonify(recommend_career(data["ats_result"], data["job_role"]))


@app.route("/download-history-pdf")
def download_history_pdf():
    if "user" not in session:
        return redirect("/login")

    pdf_path = request.args.get("path")

    if not pdf_path or not os.path.exists(pdf_path):
        return "File not found", 404

    # Security: prevent path traversal — only serve files from REPORT_FOLDER
    real_path = os.path.realpath(pdf_path)
    real_report_dir = os.path.realpath(app.config["REPORT_FOLDER"])
    if not real_path.startswith(real_report_dir):
        return "Access denied", 403

    return send_file(pdf_path, as_attachment=True)

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        email = request.form["email"].strip()
        password = request.form["password"].strip()

        success = register_user(username, email, password)
        if not success:
            flash("Username already exists", "error")
            return redirect("/register")

        flash("Registration successful. Please log in.", "success")
        return redirect("/login")

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"].strip()

        user = verify_user(username, password)

        if user:
            session["user"] = user[0]
            session["email"] = user[1]
            return redirect("/")
        else:
            flash("Invalid credentials", "error")
            return redirect("/login")

    return render_template("login.html")

@app.route("/login/google")
def login_google():
    redirect_uri = url_for('google_callback', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route("/login/google/callback")
def google_callback():
    token = google.authorize_access_token()
    user_info = google.get(
        "https://openidconnect.googleapis.com/v1/userinfo"
    ).json()


    email = user_info["email"]
    name = user_info.get("name", email)

    # Auto-register user if not exists
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("SELECT username FROM users WHERE email=?", (email,))
    user = cur.fetchone()

    if not user:
        cur.execute(
            "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
            (name, email, "GOOGLE_OAUTH")
        )
        conn.commit()

    conn.close()

    session["user"] = name
    session["email"] = email

    return redirect("/")

@app.route("/login/microsoft")
def login_microsoft():
    redirect_uri = url_for('microsoft_callback', _external=True)
    return microsoft.authorize_redirect(redirect_uri)

@app.route("/login/microsoft/callback")
def microsoft_callback():
    token = microsoft.authorize_access_token()
    user_info = microsoft.get('me').json()

    email = user_info.get("mail") or user_info.get("userPrincipalName")
    name = user_info.get("displayName", email)

    session["user"] = name
    session["email"] = email

    return redirect("/")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/login")

# ------------------------
# RESUME BUILDER ROUTES
# ------------------------

@app.route("/resume-builder", methods=["GET"])
def resume_builder_page():
    if "user" not in session:
        return redirect("/login")

    # Pre-fill form if editing
    builder_data = session.get("resume_builder_data", {})
    prefill = builder_data.get("resume_data", {})
    return render_template("resume_builder.html", prefill=prefill)


@app.route("/generate-resume", methods=["POST"])
@limiter.limit("5/minute")
def generate_resume():
    if "user" not in session:
        return redirect("/login")

    try:
        import traceback

        # Validate form
        is_valid, error_msg = validate_form_data(request.form)
        if not is_valid:
            flash(error_msg, "error")
            return redirect("/resume-builder")

        # Format resume data
        resume_data = format_resume_data(request.form)

        # Determine target role
        target_role = resume_data.get("target_role", "")
        if not target_role:
            # Predict role from skills
            roles = predict_multiple_roles(resume_data.get("skills_list", []))
            target_role = roles[0]["role"] if roles else "Software Engineer"

        # Run ATS scoring + refinement (max 2 attempts)
        from services.processing.multi_role_predictor import ROLE_SKILLS
        resume_data, ats_result = refine_for_ats(
            resume_data, target_role, calculate_ats_score, ROLE_SKILLS
        )

        # Generate initial PDF to parse real structure
        resume_data["ats_score"] = ats_result.get("ats_score", 10)
        pdf_filename = f"generated_{session['user']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex}.pdf"
        pdf_path = os.path.join(app.config["REPORT_FOLDER"], pdf_filename)
        generate_resume_pdf(resume_data, pdf_path)

        # ── ACTUALLY SCORE THE FINAL GENERATED PDF ──
        parsed_pdf = parse_resume(pdf_path)
        
        # Force completeness sections since we built them
        sf = set(s.lower() for s in parsed_pdf.get("sections_found", []))
        if resume_data.get("education"): sf.add("education")
        if resume_data.get("experience"): sf.add("experience")
        if resume_data.get("projects"): sf.add("projects")
        if resume_data.get("skills_list"): sf.add("skills")
        if resume_data.get("certifications"): sf.add("certifications")
        # Builder always generates a summary/objective section
        if resume_data.get("summary") or resume_data.get("objective"):
            sf.add("summary")
        else:
            sf.add("summary")  # builder template always includes summary
        parsed_pdf["sections_found"] = list(sf)

        app.logger.debug("GENERATE parsed_pdf sections=%s skills=%d", parsed_pdf['sections_found'], len(parsed_pdf.get('skills', [])))

        ats_result = calculate_ats_score(parsed_pdf, target_role)
        resume_data["ats_score"] = ats_result.get("ats_score", 10)
        
        app.logger.debug("GENERATE FINAL ats=%s skill=%s keyword=%s completeness=%s",
                         ats_result.get('ats_score'), ats_result.get('skill_score'),
                         ats_result.get('keyword_score'), ats_result.get('completeness_score'))

        # Re-generate PDF with exact ATS score embedded
        generate_resume_pdf(resume_data, pdf_path)

        # Auto-email generated resume to user
        try:
            send_email(
                session.get("email", ""),
                pdf_path,
                subject="Your Generated Resume \u2014 Nexus CV",
                body=(
                    f"Hello {resume_data.get('full_name', '')},\n\n"
                    f"Your resume has been generated and optimized for the "
                    f"{target_role} role.\n\n"
                    f"ATS Score: {ats_result.get('ats_score', 10)}\n\n"
                    f"Your resume PDF is attached. You can also download it "
                    f"from your Nexus CV dashboard.\n\n"
                    f"Best regards,\nNexus CV Team"
                ),
                attachment_name="Generated_Resume.pdf"
            )
        except Exception as e:
            app.logger.error(f"Failed to send builder email: {e}")

        # ── Final Sanity Filter Before Render ──
        clean_builder_skills = [
            s for s in ats_result.get("matched_skills", [])
            if isinstance(s, str)
            and len(s.split()) <= 2
        ]

        # Store in session under single key
        session["resume_builder_data"] = {
            "resume_data": resume_data,
            "ats_score": ats_result.get("ats_score", 10),
            "ats_result": {
                "skill_score": ats_result.get("skill_score", 0),
                "keyword_score": ats_result.get("keyword_score", 0),
                "completeness_score": ats_result.get("completeness_score", 0),
                "matched_skills": clean_builder_skills,
                "missing_skills": ats_result.get("missing_skills", [])
            },
            "target_role": target_role,
            "pdf_path": pdf_path
        }

        return redirect("/resume-preview")

    except Exception as e:
        import traceback
        app.logger.error(f"GENERATE ROUTE ERROR: {e}\n{traceback.format_exc()}")
        flash("An error occurred during resume generation. Please try again.", "error")
        return redirect("/resume-builder")


@app.route("/resume-preview", methods=["GET"])
def resume_preview():
    if "user" not in session:
        return redirect("/login")

    builder_data = session.get("resume_builder_data")
    if not builder_data:
        return redirect("/resume-builder")

    return render_template(
        "resume_preview.html",
        resume=builder_data["resume_data"],
        ats_score=builder_data["ats_score"],
        ats_result=builder_data["ats_result"],
        target_role=builder_data["target_role"]
    )


@app.route("/download-resume")
def download_resume():
    if "user" not in session:
        return redirect("/login")

    builder_data = session.get("resume_builder_data")
    if not builder_data or not builder_data.get("pdf_path"):
        return redirect("/resume-builder")

    pdf_path = builder_data["pdf_path"]
    if not os.path.exists(pdf_path):
        flash("Generated resume file not found. Please regenerate.", "error")
        return redirect("/resume-builder")

    # Clear builder session data after download
    session.pop("resume_builder_data", None)

    return send_file(pdf_path, as_attachment=True)


@app.route("/edit-resume")
def edit_resume():
    if "user" not in session:
        return redirect("/login")
    return redirect("/resume-builder")


# ------------------------
# SECURITY HEADERS
# ------------------------

@app.after_request
def set_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    if os.getenv("FLASK_ENV") == "production":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data:; "
            "connect-src 'self'"
        )
    # Cache static assets (CSS, JS, images, fonts)
    if response.content_type:
        ct = response.content_type
        if any(t in ct for t in ["text/css", "javascript", "image/", "font/"]):
            response.headers["Cache-Control"] = "public, max-age=86400"
    return response


# ------------------------
# BULK RESUME SCREENING
# ------------------------

# SSE progress state (per-session)
_bulk_progress = {}


@app.route("/bulk-screen/progress")
def bulk_progress_sse():
    """Server-Sent Events endpoint for bulk processing progress."""
    import queue
    import time

    user = session.get("user", "anonymous")

    def generate():
        q = _bulk_progress.get(user)
        if not q:
            yield f"data: {json.dumps({'step': 'parsing', 'detail': 'Starting...'})}\n\n"
            return

        start = time.time()
        while time.time() - start < 120:  # 2 min timeout
            try:
                msg = q.get(timeout=2)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("step") == "complete":
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'step': 'heartbeat'})}\n\n"

    return app.response_class(generate(), mimetype="text/event-stream")


@app.route("/bulk-screen", methods=["GET", "POST"])
@app.route("/bulk-upload", methods=["GET", "POST"])
def bulk_screen():
    if "user" not in session:
        return redirect("/login")

    if request.method == "GET":
        return render_template("bulk_screen.html")

    if bulk_screener is None:
        flash("Bulk screening module not available.", "error")
        return redirect("/bulk-screen")

    from werkzeug.utils import secure_filename
    import queue

    files = request.files.getlist("resumes")
    jd_text = request.form.get("job_description", "")
    role = request.form.get("role", "Software Engineer")

    if not files or not jd_text:
        flash("Please upload resumes and provide a job description", "error")
        return redirect("/bulk-screen")

    # Set up progress tracking
    user = session.get("user", "anonymous")
    progress_queue = queue.Queue()
    _bulk_progress[user] = progress_queue

    def progress_callback(step, detail=""):
        try:
            progress_queue.put({"step": step, "detail": detail})
        except Exception:
            pass

    bulk_screener.set_progress_callback(progress_callback)
    progress_callback("parsing", f"Processing {len(files)} files")

    saved_paths = []
    for file in files:
        if not file or not file.filename:
            continue
        fname_lower = file.filename.lower()

        if fname_lower.endswith(".pdf"):
            fname = secure_filename(file.filename)
            path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
            file.save(path)
            saved_paths.append(path)

        elif fname_lower.endswith(".zip"):
            # Extract PDFs from ZIP
            zip_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
            file.save(zip_path)
            try:
                from services.processing.bulk_screener import extract_pdfs_from_zip
                extracted = extract_pdfs_from_zip(zip_path, app.config["UPLOAD_FOLDER"])
                saved_paths.extend(extracted)
            except Exception as e:
                app.logger.error(f"ZIP extraction failed: {e}")
            finally:
                try:
                    os.remove(zip_path)
                except OSError:
                    pass

    # Enforce 50-resume limit
    saved_paths = saved_paths[:50]

    if not saved_paths:
        flash("No valid PDF files found. Upload PDFs or a ZIP containing PDFs.", "error")
        return redirect("/bulk-screen")

    top_n_val = int(request.form.get("top_n", 3))
    result = bulk_screener.screen_resumes(saved_paths, jd_text, role, top_n=top_n_val)

    # Clean up progress
    bulk_screener.set_progress_callback(None)
    _bulk_progress.pop(user, None)

    # Trigger n8n webhook automatically — send actual file as multipart
    try:
        import requests as http_requests

        # Re-send the first uploaded file (ZIP or PDF) to n8n as multipart
        n8n_file = None
        for file in request.files.getlist("resumes"):
            if file and file.filename:
                file.stream.seek(0)
                n8n_file = file
                break

        if n8n_file:
            n8n_response = http_requests.post(
                "http://localhost:5678/webhook/bulk-resume",
                files={
                    "file": (n8n_file.filename, n8n_file.stream, n8n_file.mimetype)
                },
                data={
                    "job_description": jd_text,
                    "role": role,
                    "top_n": str(top_n_val)
                },
                timeout=30
            )
            app.logger.info(f"n8n webhook response: {n8n_response.status_code}")
        else:
            app.logger.warning("No file available to send to n8n webhook")
    except Exception as e:
        app.logger.error(f"Failed to trigger n8n webhook: {e}")

    return render_template("bulk_result.html", result=result, role=role, top_n=result.get("top_n", 3))


# ------------------------
# n8n WEBHOOK API ENDPOINTS
# ------------------------

@app.route("/api/v1/analyze", methods=["POST"])
@csrf.exempt
@limiter.limit("10/minute")
def api_analyze():
    """n8n webhook: analyze a single resume. Expects multipart form with 'resume' PDF + optional 'role' and 'jd'."""
    api_key = request.headers.get("X-API-Key", "")
    expected_key = os.getenv("NEXUS_API_KEY", "")
    if not expected_key or api_key != expected_key:
        return jsonify({"error": "Unauthorized"}), 401

    f = request.files.get("resume")
    if not f or not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "PDF file required"}), 400

    from werkzeug.utils import secure_filename
    fname = secure_filename(f.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    f.save(path)

    role = request.form.get("role", None)
    jd_text = request.form.get("jd", None)

    if _agent:
        result = _agent.analyze_resume(path, role=role, jd_text=jd_text)
    else:
        parsed = parse_resume(path)
        ats = calculate_ats_score(parsed, role or "Software Engineer")
        result = {"success": True, "ats_result": ats, "parsed_data": {"skills": parsed["skills"]}}

    # Remove non-serializable fields
    if "parsed_data" in result:
        result["parsed_data"].pop("text", None)

    return jsonify(result)


@app.route("/api/v1/compare", methods=["POST"])
@csrf.exempt
@limiter.limit("5/minute")
def api_compare():
    """n8n webhook: compare two resumes."""
    api_key = request.headers.get("X-API-Key", "")
    expected_key = os.getenv("NEXUS_API_KEY", "")
    if not expected_key or api_key != expected_key:
        return jsonify({"error": "Unauthorized"}), 401

    f1 = request.files.get("resume_a")
    f2 = request.files.get("resume_b")
    if not f1 or not f2:
        return jsonify({"error": "Two PDF files required (resume_a, resume_b)"}), 400

    from werkzeug.utils import secure_filename
    path1 = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(f1.filename))
    path2 = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(f2.filename))
    f1.save(path1)
    f2.save(path2)

    role = request.form.get("role", "Software Engineer")
    jd_text = request.form.get("jd", None)

    if _agent:
        result = _agent.compare_resumes(path1, path2, role, jd_text)
    else:
        result = {"error": "Agent controller not available"}

    for key in ("parsed_a", "parsed_b"):
        if key in result and "text" in result[key]:
            del result[key]["text"]

    return jsonify(result)


@app.route("/api/v1/bulk-rank", methods=["POST"])
@csrf.exempt
@limiter.limit("3/minute")
def api_bulk_rank():
    """n8n webhook: rank multiple resumes against a JD."""
    api_key = request.headers.get("X-API-Key", "")
    expected_key = os.getenv("NEXUS_API_KEY", "")
    if not expected_key or api_key != expected_key:
        return jsonify({"error": "Unauthorized"}), 401

    files = request.files.getlist("resumes")
    jd_text = request.form.get("jd", "")
    role = request.form.get("role", "Software Engineer")

    if not files or not jd_text:
        return jsonify({"error": "PDF files and JD required"}), 400

    from werkzeug.utils import secure_filename
    paths = []
    for f in files[:50]:
        if f and f.filename and f.filename.lower().endswith(".pdf"):
            path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(f.filename))
            f.save(path)
            paths.append(path)

    if not paths:
        return jsonify({"error": "No valid PDFs"}), 400

    if _agent:
        result = _agent.rank_resumes(paths, jd_text, role)
    elif bulk_screener:
        result = bulk_screener.screen_resumes(paths, jd_text, role)
    else:
        result = {"error": "Ranking not available"}

    return jsonify(result)


@app.route("/api/v1/score", methods=["POST"])
@csrf.exempt
@limiter.limit("20/minute")
def api_score():
    """n8n webhook: score a single resume from text. Called by n8n Pipeline 2."""
    api_key = request.headers.get("X-API-Key", "")
    expected_key = os.getenv("NEXUS_API_KEY", "")
    # Allow unauthenticated requests from localhost (n8n internal calls)
    is_local = request.remote_addr in ("127.0.0.1", "::1", "localhost")
    if expected_key and api_key != expected_key and not is_local:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    resume_text = data.get("resume_text", "")
    role = data.get("role", "Software Engineer")
    jd_text = data.get("job_description", "")
    run_agents = data.get("run_agents", False)

    if not resume_text:
        return jsonify({"error": "resume_text is required"}), 400

    try:
        # Build a parsed_data-like dict from raw text
        parsed_data = {
            "text": resume_text,
            "skills": [],
            "sections_found": []
        }
        # Try to extract skills from text via parser's skill extraction
        try:
            from services.processing.resume_parser import extract_skills_from_text
            parsed_data["skills"] = extract_skills_from_text(resume_text)
        except (ImportError, AttributeError):
            # Fallback: use simple keyword matching
            import re
            words = set(re.findall(r'\b[A-Za-z+#]{2,}\b', resume_text))
            parsed_data["skills"] = list(words)

        ats_result = calculate_ats_score(parsed_data, role)

        # Optionally run agent scoring
        agent_score = 0
        confidence = "medium"
        reasoning = ""
        evidence = []
        thought_trace = []

        if run_agents and _agent:
            try:
                agent_result = _agent.score_resume_text(resume_text, role, jd_text)
                agent_score = agent_result.get("agent_score", 0)
                confidence = agent_result.get("confidence", "medium")
                reasoning = agent_result.get("reasoning", "")
                evidence = agent_result.get("evidence", [])
                thought_trace = agent_result.get("thought_trace", [])
            except Exception as agent_err:
                app.logger.error(f"Agent scoring failed: {agent_err}")

        return jsonify({
            "file_name": data.get("file_name", "unknown"),
            "ats_score": ats_result.get("ats_score", 0),
            "agent_score": agent_score,
            "confidence": confidence,
            "reasoning": reasoning,
            "evidence": evidence,
            "matched_skills": ats_result.get("matched_skills", []),
            "missing_skills": ats_result.get("missing_skills", []),
            "thought_trace": thought_trace,
            "role": role
        })

    except Exception as e:
        app.logger.error(f"Score API error: {e}")
        return jsonify({
            "file_name": data.get("file_name", "unknown"),
            "ats_score": 0,
            "agent_score": 0,
            "confidence": "low",
            "reasoning": f"Scoring failed: {str(e)}",
            "error": str(e)
        }), 500


@app.route("/api/v1/health", methods=["GET"])
@csrf.exempt
def api_health():
    """Health check for n8n and monitoring."""
    from services.ml import model_hub as _mh
    return jsonify({
        "status": "ok",
        "agent_controller": _agent is not None,
        "gemini_available": gemini_agent is not None,
        "embedding_cache": _mh.get_cache_stats() if _mh else {},
    })


# ------------------------
# ERROR HANDLERS
# ------------------------

@app.errorhandler(404)
def page_not_found(e):
    app.logger.warning("404 Not Found: %s", request.url)
    return render_template("404.html"), 404

@app.errorhandler(500)
def internal_server_error(e):
    app.logger.error("500 Internal Server Error: %s", e)
    return render_template("500.html"), 500

@app.errorhandler(429)
def ratelimit_handler(e):
    app.logger.warning("Rate limit exceeded: %s from %s", request.url, request.remote_addr)
    return jsonify(error="Rate limit exceeded. Please try again later."), 429


# ------------------------
# BACKGROUND CLEANUP JOB
# ------------------------

def _start_cleanup_scheduler():
    """Start APScheduler background job for file cleanup.
    Uses job ID + replace_existing to prevent duplicate jobs on reload."""
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from utils.cleanup import cleanup_old_files
        scheduler = BackgroundScheduler(daemon=True)
        scheduler.add_job(
            cleanup_old_files,
            "interval",
            hours=6,
            id="nexuscv_cleanup",
            replace_existing=True
        )
        scheduler.start()
        app.logger.info("Cleanup scheduler started (every 6 hours)")
    except ImportError:
        app.logger.warning("APScheduler not installed, skipping cleanup background job")


# ------------------------
# RUN APP
# ------------------------

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)