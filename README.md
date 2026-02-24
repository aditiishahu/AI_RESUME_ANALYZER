# AI Resume Analyzer

AI Resume Analyzer is a Flask-based web application that evaluates a resume against a target job description and returns an ATS-style score with actionable recommendations.

## Highlights

- Upload PDF resumes and analyze against custom job descriptions.
- ATS-style similarity scoring using TF-IDF + cosine similarity.
- Section-level scoring for skills, experience, education, and formatting.
- Keyword gap analysis (matched vs missing terms).
- AI-style feedback suggestions for improvement.
- Analysis history persisted in SQLite.
- Downloadable PDF analysis report.
- Health endpoint for runtime checks (`/health`).

## Tech Stack

- **Backend:** Flask, SQLite, scikit-learn, pdfplumber, reportlab
- **Frontend:** Bootstrap 5, Chart.js, Jinja2 templates

## Security and Reliability Improvements

- Enforced file extension checks for uploads (`.pdf` only).
- Sanitized uploaded filenames using `secure_filename`.
- Limited upload size via `MAX_CONTENT_LENGTH` (5MB).
- Safer request parsing and fallback behavior for invalid payloads.
- Added deterministic tokenization for better keyword matching.
- Added basic test coverage for core analysis functions.
- Debug mode now controlled by `FLASK_DEBUG` environment variable.

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open: `http://127.0.0.1:5000`

## Tests

```bash
python -m unittest discover -s tests
```

## Project Structure

```text
AI_RESUME_ANALYZER/
├── app.py
├── database.db
├── requirements.txt
├── templates/
│   ├── index.html
│   ├── result.html
│   └── history.html
├── uploads/
└── tests/
    └── test_app_logic.py
```

## Suggested Next Upgrades

- Add user authentication and per-user analysis history.
- Add asynchronous/background processing for large files.
- Containerize with Docker and add CI workflow.
- Add richer NLP (stemming/lemmatization, phrase extraction).
- Export CSV/JSON analytics in addition to PDF reports.
