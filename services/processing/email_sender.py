"""
Email sender for Nexus CV.
Supports SendGrid API (preferred) with SMTP fallback.
"""

import os
import logging
from email.message import EmailMessage

logger = logging.getLogger(__name__)

# ── SendGrid (preferred) ──
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
SENDGRID_FROM_EMAIL = os.getenv("SENDGRID_FROM_EMAIL", "noreply@nexuscv.com")

# ── SMTP fallback ──
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")


def _send_via_sendgrid(to_email, subject, body, pdf_path, attachment_name):
    """Send email using SendGrid API."""
    import base64
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import (
        Mail, Attachment, FileContent, FileName, FileType, Disposition
    )

    message = Mail(
        from_email=SENDGRID_FROM_EMAIL,
        to_emails=to_email,
        subject=subject,
        plain_text_content=body
    )

    with open(pdf_path, "rb") as f:
        encoded_file = base64.b64encode(f.read()).decode()

    attachment = Attachment(
        FileContent(encoded_file),
        FileName(attachment_name),
        FileType("application/pdf"),
        Disposition("attachment")
    )
    message.attachment = attachment

    client = SendGridAPIClient(SENDGRID_API_KEY)
    response = client.send(message)
    logger.info("SendGrid response: status=%s", response.status_code)
    return response.status_code


def _send_via_smtp(to_email, subject, body, pdf_path, attachment_name):
    """Send email using Gmail SMTP as fallback."""
    import smtplib

    if not EMAIL_USER or not EMAIL_PASS:
        raise ValueError("SMTP credentials not set (EMAIL_USER / EMAIL_PASS)")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = EMAIL_USER
    msg["To"] = to_email
    msg.set_content(body)

    with open(pdf_path, "rb") as f:
        msg.add_attachment(
            f.read(),
            maintype="application",
            subtype="pdf",
            filename=attachment_name
        )

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)

    logger.info("SMTP email sent to %s", to_email)


def send_email(to_email, pdf_path, subject=None, body=None, attachment_name=None):
    """
    Send an email with an attached PDF.

    Uses SendGrid API if SENDGRID_API_KEY is set, otherwise falls back to SMTP.

    Parameters are optional for backward compatibility:
      - subject: defaults to "Your Resume Analysis Report"
      - body: defaults to the standard analyze-flow message
      - attachment_name: defaults to "Resume_Analysis_Report.pdf"
    """
    if subject is None:
        subject = "Your Resume Analysis Report"
    if body is None:
        body = (
            "Hello,\n\n"
            "Your resume analysis report is attached.\n\n"
            "Thank you for using Nexus CV.\n"
        )
    if attachment_name is None:
        attachment_name = "Resume_Analysis_Report.pdf"

    # Try SendGrid first
    if SENDGRID_API_KEY:
        try:
            _send_via_sendgrid(to_email, subject, body, pdf_path, attachment_name)
            return
        except Exception as e:
            logger.warning("SendGrid failed, falling back to SMTP: %s", e)

    # SMTP fallback
    try:
        _send_via_smtp(to_email, subject, body, pdf_path, attachment_name)
    except Exception as e:
        logger.error("Email send failed (all providers): %s", e)
