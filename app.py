import os
import sqlite3
import pdfplumber
import json
from collections import Counter
from flask import Flask, render_template, request, redirect, url_for, send_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib import colors
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'


# ---------------- DATABASE SETUP ---------------- #

def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            job_description TEXT,
            score REAL,
            created_at TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()


# ---------------- PDF TEXT EXTRACTION ---------------- #

def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


# ---------------- AI SIMILARITY ---------------- #

def calculate_similarity(resume_text, job_desc):
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([resume_text, job_desc])
    similarity = cosine_similarity(vectors[0], vectors[1])
    return round(float(similarity[0][0]) * 100, 2)


def calculate_section_scores(resume_text, job_desc):
    """Calculate scores for different resume sections"""
    resume_lower = resume_text.lower()
    job_lower = job_desc.lower()
    
    # Skills section score
    skills_keywords = ['python', 'java', 'javascript', 'sql', 'aws', 'azure', 'react', 'angular', 'node', 'django', 'flask', 'tableau', 'power bi', 'excel', 'git', 'docker', 'kubernetes']
    skills_found = sum(1 for skill in skills_keywords if skill in resume_lower)
    skills_score = min(100, (skills_found / 10) * 100)
    
    # Experience section score (looks for action verbs and metrics)
    action_verbs = ['led', 'developed', 'managed', 'increased', 'improved', 'reduced', 'implemented', 'created', 'designed', 'built', 'achieved']
    action_count = sum(1 for verb in action_verbs if verb in resume_lower)
    experience_score = min(100, (action_count / 5) * 100)
    
    # Education section score
    education_keywords = ['bachelor', 'master', 'phd', 'degree', 'certification', 'university', 'college', 'institute']
    education_found = sum(1 for keyword in education_keywords if keyword in resume_lower)
    education_score = min(100, (education_found / 2) * 100)
    
    # Formatting score (based on word count and structure)
    word_count = len(resume_text.split())
    formatting_score = min(100, (word_count / 500) * 100) if word_count > 100 else 50
    
    return {
        'skills': round(skills_score, 1),
        'experience': round(experience_score, 1),
        'education': round(education_score, 1),
        'formatting': round(formatting_score, 1)
    }


def generate_smart_feedback(resume_text, job_desc, matched_keywords, missing_keywords):
    """Generate detailed, actionable feedback"""
    feedback = []
    
    # Skills feedback
    if len(matched_keywords) < 5:
        feedback.append({
            'type': 'critical',
            'title': 'Low keyword match',
            'message': f'Your resume contains only {len(matched_keywords)} matched keywords. Target roles typically need 10-20.'
        })
    
    # Missing keywords feedback
    if len(missing_keywords) > 15:
        feedback.append({
            'type': 'warning',
            'title': 'Missing key skills',
            'message': f'You\'re missing {len(missing_keywords)} important keywords. Consider adding: {", ".join(missing_keywords[:5])}'
        })
    
    # Metrics feedback
    if 'increased' not in resume_text.lower() and 'improved' not in resume_text.lower():
        feedback.append({
            'type': 'warning',
            'title': 'No measurable achievements',
            'message': 'Add quantifiable metrics (e.g., "increased revenue by 20%", "reduced processing time by 30%")'
        })
    
    # Action verb feedback
    action_verbs = ['led', 'developed', 'managed', 'implemented', 'designed']
    if not any(verb in resume_text.lower() for verb in action_verbs):
        feedback.append({
            'type': 'warning',
            'title': 'Weak action verbs',
            'message': 'Use strong action verbs like: Led, Developed, Managed, Implemented, Designed'
        })
    
    if len(matched_keywords) > 10:
        feedback.append({
            'type': 'success',
            'title': 'Great keyword alignment',
            'message': 'Your resume matches well with the job requirements!'
        })
    
    return feedback if feedback else [{'type': 'success', 'title': 'Good resume', 'message': 'Your resume looks solid!'}]


def calculate_keyword_density(resume_text):
    """Calculate keyword frequency and density"""
    words = resume_text.lower().split()
    # Filter out common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'is', 'was', 'be', 'are', 'were', 'been', 'with', 'from', 'by', 'as', 'it', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how'}
    filtered_words = [w for w in words if w.isalpha() and len(w) > 3 and w not in stop_words]
    
    word_freq = Counter(filtered_words)
    top_keywords = word_freq.most_common(15)
    
    return top_keywords


def generate_pdf_report(filename, job_desc, score, section_scores, matched, missing, feedback, keyword_density):
    """Generate a PDF report of the analysis"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=30,
        alignment=1
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    story.append(Paragraph("📊 Resume Analysis Report", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Resume Info
    info_data = [
        ['Resume File:', filename],
        ['Analysis Date:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['Overall ATS Score:', f'{score}%']
    ]
    info_table = Table(info_data, colWidths=[2*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f4ff')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Section Scores
    story.append(Paragraph("Section Scores", heading_style))
    score_data = [
        ['Section', 'Score'],
        ['Skills', f"{section_scores['skills']}%"],
        ['Experience', f"{section_scores['experience']}%"],
        ['Education', f"{section_scores['education']}%"],
        ['Formatting', f"{section_scores['formatting']}%"]
    ]
    score_table = Table(score_data, colWidths=[3*inch, 2*inch])
    score_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')])
    ]))
    story.append(score_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Keywords
    story.append(Paragraph("Matched Keywords", heading_style))
    matched_text = ', '.join(matched) if matched else 'None'
    story.append(Paragraph(matched_text, styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("Missing Keywords", heading_style))
    missing_text = ', '.join(missing) if missing else 'None'
    story.append(Paragraph(missing_text, styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Top Keywords
    story.append(Paragraph("Top Keywords in Resume", heading_style))
    keyword_data = [['Keyword', 'Frequency']]
    for keyword, freq in keyword_density[:10]:
        keyword_data.append([keyword.capitalize(), str(freq)])
    
    keyword_table = Table(keyword_data, colWidths=[3*inch, 2*inch])
    keyword_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')])
    ]))
    story.append(keyword_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Feedback
    story.append(Paragraph("AI Feedback & Suggestions", heading_style))
    for item in feedback:
        feedback_text = f"<b>{item['title']}:</b> {item['message']}"
        story.append(Paragraph(feedback_text, styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer


# ---------------- ROUTES ---------------- #

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['resume']
    job_desc = request.form['job_description']

    if file and job_desc:

        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        resume_text = extract_text_from_pdf(filepath)

        # Calculate ATS score
        score = calculate_similarity(resume_text, job_desc)
        
        # Calculate section scores
        section_scores = calculate_section_scores(resume_text, job_desc)

        # Determine resume strength
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

        # Calculate readiness percentage
        readiness = round((score + sum(section_scores.values()) / 4) / 2, 1)

        # ---------------- KEYWORD ANALYSIS ---------------- #

        resume_words = set(resume_text.lower().split())
        job_words = set(job_desc.lower().split())

        matched_keywords = list(resume_words.intersection(job_words))
        missing_keywords = list(job_words.difference(resume_words))

        # Limit results to avoid UI overload
        matched_keywords = matched_keywords[:10]
        missing_keywords = missing_keywords[:10]
        
        # Generate smart feedback
        feedback = generate_smart_feedback(resume_text, job_desc, matched_keywords, missing_keywords)
        
        # Calculate keyword density
        keyword_density = calculate_keyword_density(resume_text)

        # ---------------- SAVE TO DATABASE ---------------- #

        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute(
            "INSERT INTO resumes (filename, job_description, score, created_at) VALUES (?, ?, ?, ?)",
            (file.filename, job_desc, score, datetime.now())
        )
        conn.commit()
        conn.close()

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
            resume_filename=file.filename
        )

    return redirect(url_for('index'))


@app.route('/history')
def history():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT * FROM resumes ORDER BY id DESC")
    data = c.fetchall()
    conn.close()
    return render_template('history.html', data=data)


@app.route('/download_report', methods=['POST'])
def download_report():
    """Generate and download PDF report"""
    data = request.get_json()
    
    filename = data.get('filename', 'resume.pdf')
    score = float(data.get('score', 0))
    section_scores = {
        'skills': float(data.get('skills_score', 0)),
        'experience': float(data.get('experience_score', 0)),
        'education': float(data.get('education_score', 0)),
        'formatting': float(data.get('formatting_score', 0))
    }
    matched = data.get('matched', [])
    missing = data.get('missing', [])
    feedback = data.get('feedback', [])
    keyword_density = data.get('keyword_density', [])
    
    pdf_buffer = generate_pdf_report(filename, '', score, section_scores, matched, missing, feedback, keyword_density)
    
    return send_file(
        pdf_buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'resume_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
    )


# ---------------- RUN APP ---------------- #

if __name__ == '__main__':
    app.run(debug=True)
