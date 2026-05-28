"""
Shared PDF utility functions for Nexus CV.
Used by both pdf_generator.py (analysis reports) and resume_builder.py (generated resumes).
"""

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ────────────── CONSTANTS ──────────────

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN_LEFT = 50
MARGIN_RIGHT = 50
MARGIN_TOP = 40
MARGIN_BOTTOM = 50
CONTENT_WIDTH = PAGE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT

FONT_REGULAR = "Helvetica"
FONT_BOLD = "Helvetica-Bold"

LINE_HEIGHT_NORMAL = 16
LINE_HEIGHT_HEADING = 22
SECTION_GAP = 14


# ────────────── HELPERS ──────────────

def draw_section_heading(c, y, title):
    """Draw a bold uppercase section heading with a separator line below it."""
    if y < MARGIN_BOTTOM + 40:
        c.showPage()
        y = PAGE_HEIGHT - MARGIN_TOP

    c.setFont(FONT_BOLD, 12)
    c.drawString(MARGIN_LEFT, y, title.upper())
    y -= 4
    c.setStrokeColorRGB(0.08, 0.08, 0.08)
    c.setLineWidth(0.5)
    c.line(MARGIN_LEFT, y, PAGE_WIDTH - MARGIN_RIGHT, y)
    y -= LINE_HEIGHT_NORMAL
    return y


def draw_text_line(c, y, text, font=FONT_REGULAR, size=10, indent=0):
    """Draw a single line of text. Returns new y position."""
    if y < MARGIN_BOTTOM + 20:
        c.showPage()
        y = PAGE_HEIGHT - MARGIN_TOP

    c.setFont(font, size)
    c.drawString(MARGIN_LEFT + indent, y, text)
    y -= LINE_HEIGHT_NORMAL
    return y


def draw_bullet_list(c, y, items, font=FONT_REGULAR, size=10, indent=10, max_items=None):
    """Draw a bulleted list of items. Returns new y position."""
    count = 0
    for item in items:
        if max_items and count >= max_items:
            break
        if y < MARGIN_BOTTOM + 20:
            c.showPage()
            y = PAGE_HEIGHT - MARGIN_TOP
        c.setFont(font, size)
        # Truncate long lines to fit page width
        display_text = item[:95] + "..." if len(item) > 98 else item
        c.drawString(MARGIN_LEFT + indent, y, f"• {display_text}")
        y -= LINE_HEIGHT_NORMAL
        count += 1
    return y


def safe_new_page(c, y, threshold=None):
    """Start a new page if y is below threshold. Returns new y."""
    if threshold is None:
        threshold = MARGIN_BOTTOM + 40
    if y < threshold:
        c.showPage()
        y = PAGE_HEIGHT - MARGIN_TOP
    return y
