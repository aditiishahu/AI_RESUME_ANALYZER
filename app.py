import os
import sqlite3
from contextlib import closing
from datetime import datetime
from io import BytesIO

import pdfplumber
from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename

import analyzer_core

app = Flask(__name__)
app.config.update(
    UPLOAD_FOLDER='uploads',
    DATABASE='database.db',
    MAX_CONTENT_LENGTH=5 * 1024 * 1024,  # 5MB
)

ALLOWED_EXTENSIONS = {'.pdf'}

def get_db_connection():
    return sqlite3.connect(app.config['DATABASE'])


# ---------------- DATABASE SETUP ---------------- #

def init_db():
    with closing(get_db_connection()) as conn:
        c = conn.cursor()
        c.execute(
            '''
            CREATE TABLE IF NOT EXISTS resumes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                job_description TEXT NOT NULL,
                score REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            '''
        )
        conn.commit()


init_db()


# ---------------- PDF TEXT EXTRACTION ---------------- #

def allowed_file(filename):
    _, ext = os.path.splitext((filename or '').lower())
    return ext in ALLOWED_EXTENSIONS


def extract_text_from_pdf(path):
    text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text() or "")
    return "\n".join(text).strip()


# ---------------- AI SIMILARITY ---------------- #

def calculate_similarity(resume_text, job_desc):
    if not resume_text.strip() or not job_desc.strip():
        return 0.0

    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        vectors = vectorizer.fit_transform([resume_text, job_desc])
    except ValueError:
        return 0.0

    similarity = cosine_similarity(vectors[0], vectors[1])
    return round(float(similarity[0][0]) * 100, 2)


def calculate_section_scores(resume_text, job_desc):
    """Calculate score components for different resume sections."""
    resume_lower = (resume_text or '').lower()

    skills_keywords = [
        'python', 'java', 'javascript', 'sql', 'aws', 'azure', 'react', 'angular', 'node',
        'django', 'flask', 'tableau', 'power bi', 'excel', 'git', 'docker', 'kubernetes'
    ]
    skills_found = sum(1 for skill in skills_keywords if skill in resume_lower)
    skills_score = min(100, (skills_found / 10) * 100)

    action_verbs = [
        'led', 'developed', 'managed', 'increased', 'improved', 'reduced', 'implemented',
        'created', 'designed', 'built', 'achieved'
    ]
    action_count = sum(1 for verb in action_verbs if verb in resume_lower)
    experience_score = min(100, (action_count / 5) * 100)

    education_keywords = ['bachelor', 'master', 'phd', 'degree', 'certification', 'university', 'college', 'institute']
    education_found = sum(1 for keyword in education_keywords if keyword in resume_lower)
    education_score = min(100, (education_found / 2) * 100)

    word_count = len(analyzer_core.normalize_text_to_tokens(resume_text))
    formatting_score = min(100, (word_count / 500) * 100) if word_count > 100 else 50

    return {
        'skills': round(skills_score, 1),
        'experience': round(experience_score, 1),
        'education': round(education_score, 1),
        'formatting': round(formatting_score, 1),
    }



def generate_pdf_report(filename, score, section_scores, matched, missing, feedback, keyword_density):
    """Generate a PDF report of the analysis."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=30,
        alignment=1,
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=12,
        spaceBefore=12,
    )

    story.append(Paragraph("📊 Resume Analysis Report", title_style))
    story.append(Spacer(1, 0.3 * inch))

    info_data = [
        ['Resume File:', filename],
        ['Analysis Date:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['Overall ATS Score:', f'{score}%'],
    ]
    info_table = Table(info_data, colWidths=[2 * inch, 4 * inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f4ff')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph("Section Scores", heading_style))
    score_data = [
        ['Section', 'Score'],
        ['Skills', f"{section_scores['skills']}%"],
        ['Experience', f"{section_scores['experience']}%"],
        ['Education', f"{section_scores['education']}%"],
        ['Formatting', f"{section_scores['formatting']}%"],
    ]
    score_table = Table(score_data, colWidths=[3 * inch, 2 * inch])
    score_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')]),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph("Matched Keywords", heading_style))
    story.append(Paragraph(', '.join(matched) if matched else 'None', styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Missing Keywords", heading_style))
    story.append(Paragraph(', '.join(missing) if missing else 'None', styles['Normal']))
    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph("Top Keywords in Resume", heading_style))
    keyword_data = [['Keyword', 'Frequency']]
    for keyword, freq in keyword_density[:10]:
        keyword_data.append([keyword.capitalize(), str(freq)])

    keyword_table = Table(keyword_data, colWidths=[3 * inch, 2 * inch])
    keyword_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')]),
    ]))
    story.append(keyword_table)
    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph("AI Feedback & Suggestions", heading_style))
    for item in feedback:
        story.append(Paragraph(f"<b>{item['title']}:</b> {item['message']}", styles['Normal']))
        story.append(Spacer(1, 0.1 * inch))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ---------------- ROUTES ---------------- #

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'timestamp': datetime.utcnow().isoformat()})


@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files.get('resume')
    job_desc = (request.form.get('job_description') or '').strip()

    if not file or not file.filename:
        return redirect(url_for('index'))

    if not job_desc:
        return redirect(url_for('index'))

    if not allowed_file(file.filename):
        return redirect(url_for('index'))

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    safe_filename = secure_filename(file.filename)
    if not safe_filename:
        safe_filename = f'resume_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.pdf'

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
    file.save(filepath)

    try:
        resume_text = extract_text_from_pdf(filepath)
    except Exception:
        return redirect(url_for('index'))

    score = calculate_similarity(resume_text, job_desc)
    section_scores = calculate_section_scores(resume_text, job_desc)

    if score >= 80:
        rating = "Excellent Match"
        strength = "Professional"
        strength_icon = "🏆"
    elif score >= 60:
        rating = "Good Match"
        strength = "Intermediate"
        strength_icon = "⭐"
    else:
        rating = "Needs Improvement"
        strength = "Beginner"
        strength_icon = "📈"

    readiness = round((score + sum(section_scores.values()) / 4) / 2, 1)

    keyword_analysis = analyzer_core.analyze_keywords(resume_text, job_desc, max_items=10)
    matched_keywords = keyword_analysis['matched']
    missing_keywords = keyword_analysis['missing']

    feedback = analyzer_core.generate_smart_feedback(
        resume_text,
        matched_keywords,
        missing_keywords,
        keyword_analysis['matched_count_total'],
        keyword_analysis['missing_count_total'],
    )
    keyword_density = analyzer_core.calculate_keyword_density(resume_text)

    with closing(get_db_connection()) as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO resumes (filename, job_description, score, created_at) VALUES (?, ?, ?, ?)",
            (safe_filename, job_desc, score, datetime.utcnow().isoformat()),
        )
        conn.commit()

    return render_template(
        'result.html',
        score=score,
        matched=matched_keywords,
        missing=missing_keywords,
        rating=rating,
        section_scores=section_scores,
        strength=strength,
        strength_icon=strength_icon,
        readiness=readiness,
        feedback=feedback,
        keyword_density=keyword_density,
        resume_filename=safe_filename,
    )


@app.route('/history')
def history():
    with closing(get_db_connection()) as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM resumes ORDER BY id DESC LIMIT 200")
        data = c.fetchall()
    return render_template('history.html', data=data)


@app.route('/download_report', methods=['POST'])
def download_report():
    """Generate and download PDF report."""
    data = request.get_json(silent=True) or {}

    filename = secure_filename(data.get('filename', 'resume.pdf')) or 'resume.pdf'
    score = round(float(data.get('score', 0) or 0), 2)
    section_scores = {
        'skills': float(data.get('skills_score', 0) or 0),
        'experience': float(data.get('experience_score', 0) or 0),
        'education': float(data.get('education_score', 0) or 0),
        'formatting': float(data.get('formatting_score', 0) or 0),
    }
    matched = data.get('matched', [])[:25]
    missing = data.get('missing', [])[:25]
    feedback = data.get('feedback', [])[:20]
    keyword_density = data.get('keyword_density', [])[:25]

    pdf_buffer = generate_pdf_report(filename, score, section_scores, matched, missing, feedback, keyword_density)

    return send_file(
        pdf_buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'resume_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf',
    )


if __name__ == '__main__':
    app.run(debug=os.environ.get('FLASK_DEBUG', '0') == '1')
