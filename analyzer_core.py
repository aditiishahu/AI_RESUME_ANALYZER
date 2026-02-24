import re
from collections import Counter

COMMON_STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'is', 'was', 'be',
    'are', 'were', 'been', 'with', 'from', 'by', 'as', 'it', 'this', 'that', 'these', 'those', 'i',
    'you', 'he', 'she', 'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how'
}


def normalize_text_to_tokens(text):
    return re.findall(r"[a-zA-Z]{2,}", (text or '').lower())


def filter_meaningful_tokens(tokens):
    return [t for t in tokens if t not in COMMON_STOP_WORDS and len(t) > 2]


def analyze_keywords(resume_text, job_desc, max_items=15):
    resume_tokens = set(filter_meaningful_tokens(normalize_text_to_tokens(resume_text)))
    job_tokens = set(filter_meaningful_tokens(normalize_text_to_tokens(job_desc)))

    matched = sorted(resume_tokens.intersection(job_tokens))
    missing = sorted(job_tokens.difference(resume_tokens))

    return {
        'matched': matched[:max_items],
        'missing': missing[:max_items],
        'matched_count_total': len(matched),
        'missing_count_total': len(missing),
    }


def generate_smart_feedback(resume_text, matched_keywords, missing_keywords, matched_total, missing_total):
    feedback = []
    resume_lower = (resume_text or '').lower()

    if matched_total < 8:
        feedback.append({
            'type': 'critical',
            'title': 'Low keyword match',
            'message': f'Only {matched_total} relevant keywords were found. Aim for at least 10-20 job-specific terms.'
        })

    if missing_total > 15:
        feedback.append({
            'type': 'warning',
            'title': 'Missing key skills',
            'message': f'{missing_total} important keywords are missing. Consider adding: {", ".join(missing_keywords[:5]) or "job-specific terms"}.'
        })

    if 'increased' not in resume_lower and 'improved' not in resume_lower and '%' not in resume_text:
        feedback.append({
            'type': 'warning',
            'title': 'No measurable achievements',
            'message': 'Add quantifiable metrics (e.g., "reduced processing time by 30%", "improved conversion by 12%").'
        })

    action_verbs = ['led', 'developed', 'managed', 'implemented', 'designed']
    if not any(verb in resume_lower for verb in action_verbs):
        feedback.append({
            'type': 'warning',
            'title': 'Weak action verbs',
            'message': 'Use stronger action verbs: Led, Developed, Managed, Implemented, Designed.'
        })

    if matched_total >= 12 and missing_total <= 8:
        feedback.append({
            'type': 'success',
            'title': 'Great keyword alignment',
            'message': 'Your resume aligns well with this role. Keep tailoring for each application.'
        })

    return feedback or [{'type': 'success', 'title': 'Good resume foundation', 'message': 'Your resume looks solid. Continue tailoring it for each role.'}]


def calculate_keyword_density(resume_text):
    words = filter_meaningful_tokens(normalize_text_to_tokens(resume_text))
    return Counter(words).most_common(15)
