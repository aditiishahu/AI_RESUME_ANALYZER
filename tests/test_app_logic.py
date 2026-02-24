import unittest

from analyzer_core import (
    analyze_keywords,
    calculate_keyword_density,
    filter_meaningful_tokens,
    generate_smart_feedback,
    normalize_text_to_tokens,
)


class AppLogicTests(unittest.TestCase):
    def test_normalize_text_to_tokens(self):
        tokens = normalize_text_to_tokens('Python, SQL & AI-driven! 2025')
        self.assertIn('python', tokens)
        self.assertIn('sql', tokens)
        self.assertIn('driven', tokens)

    def test_filter_meaningful_tokens(self):
        tokens = filter_meaningful_tokens(['python', 'and', 'team', 'the'])
        self.assertEqual(tokens, ['python', 'team'])

    def test_keyword_analysis_counts(self):
        result = analyze_keywords('Python SQL Flask', 'Python Java SQL Docker', max_items=10)
        self.assertEqual(result['matched_count_total'], 2)
        self.assertEqual(result['missing_count_total'], 2)
        self.assertIn('python', result['matched'])

    def test_feedback_contains_critical_for_low_match(self):
        feedback = generate_smart_feedback(
            resume_text='Worked on tasks',
            matched_keywords=['python'],
            missing_keywords=['sql', 'docker', 'kubernetes'],
            matched_total=1,
            missing_total=20,
        )
        titles = [item['title'] for item in feedback]
        self.assertIn('Low keyword match', titles)

    def test_keyword_density(self):
        density = calculate_keyword_density('Python python SQL data data data and the')
        self.assertEqual(density[0][0], 'data')
        self.assertEqual(density[0][1], 3)


if __name__ == '__main__':
    unittest.main()
