"""Synthetic regressions for leakage and selection hazards in upload preparation."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.uploaded_pilot import load_authored, prepare_cohort
from src.pilot_sensitivity import matched_positions


def row(author, community, date, suffix, body=None):
    return {'author': author, 'subreddit': community, 'created_utc': pd.Timestamp(date, tz='UTC').timestamp(),
            'score': 1, 'selftext': body if body is not None else 'This synthetic document contains enough different words to satisfy the fixed eligibility rule.',
            'title': 'Fixture title ' + suffix}


def fixture():
    return pd.DataFrame([
        row(f'{tag}_{label}_{i}', community, f'2022-{month}-10', f'{tag} {label} {i}')
        for month, tag in [('03', 'train'), ('05', 'val'), ('06', 'test')]
        for label, community in [(0, 'mentalhealth'), (1, 'Anxiety')]
        for i in range(3)
    ])


class UploadedPilotTests(unittest.TestCase):
    def test_earlier_removed_posts_still_exclude_later_authors(self):
        raw = fixture()
        extra = row('val_1_0', 'Anxiety', '2022-03-01', 'removed', body='[removed]')
        result, report = prepare_cohort(pd.concat([raw, pd.DataFrame([extra])]), 'mentalhealth')
        self.assertNotIn('val_1_0', set(result['author']))
        self.assertEqual(report['windows']['val']['previously_observed_authors_excluded'], 1)

    def test_future_community_switch_does_not_relabel_training(self):
        raw = fixture()
        original, _ = prepare_cohort(raw, 'mentalhealth')
        extra = row('train_1_0', 'mentalhealth', '2022-06-20', 'future switch')
        result, _ = prepare_cohort(pd.concat([raw, pd.DataFrame([extra])]), 'mentalhealth')
        pd.testing.assert_frame_equal(original[original['split'].eq('train')].reset_index(drop=True),
                                      result[result['split'].eq('train')].reset_index(drop=True))
        self.assertEqual(result.loc[result['author'].eq('train_1_0'), 'label'].tolist(), [1])

    def test_ambiguous_affiliation_uses_metadata_before_body_filter(self):
        raw = fixture()
        extra = row('test_1_0', 'mentalhealth', '2022-06-15', 'cross post', body='[removed]')
        result, report = prepare_cohort(pd.concat([raw, pd.DataFrame([extra])]), 'mentalhealth')
        self.assertNotIn('test_1_0', set(result['author']))
        self.assertEqual(report['windows']['test']['ambiguous_community_authors'], 1)

    def test_utc_boundary_and_forward_duplicate_policy(self):
        raw = fixture()
        # The apparent May date in an undocumented local string is irrelevant.
        extra = row('boundary_author', 'Anxiety', '2022-04-30 20:00', 'boundary')
        extra['timestamp'] = '2022-05-01 07:00:00'
        copied = row('test_1_0', 'Anxiety', '2022-06-20', '')
        copied['title'], copied['selftext'] = raw.iloc[3]['title'], raw.iloc[3]['selftext']
        result, report = prepare_cohort(pd.concat([raw, pd.DataFrame([extra, copied])]), 'mentalhealth')
        self.assertNotIn('boundary_author', set(result['author']))
        self.assertEqual(report['windows']['test']['earlier_window_duplicate_text_rows_excluded'], 1)
        self.assertIn('train_1_0', set(result['author']))
        self.assertEqual(report['cohort_integrity']['cross_split_duplicate_text'], 0)

    def test_file_duplicates_skipped_and_modified_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'input.csv'
            fixture().to_csv(source, index=False)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            manifest = [{'name': name, 'path': str(source), 'sha256': digest} for name in ['a.csv', 'b.csv']]
            path = root/'manifest.json'
            path.write_text(json.dumps(manifest))
            raw, files = load_authored(path)
            self.assertEqual(len(raw), len(fixture()))
            self.assertEqual(files[1]['status'], 'duplicate_file_skipped')
            source.write_bytes(source.read_bytes() + b'altered')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_authored(path)

    def test_matching_is_order_invariant_balanced_and_uses_no_predictions(self):
        users = pd.DataFrame({'label':[0,0,0,1,1,1], 'mean_words':[80,85,180,82,88,800],
                              'posts':[1,1,2,1,1,3], 'probability':[0.,0.,0.,1.,1.,1.]},
                             index=['account_a','account_b','account_c','account_d','account_e','account_f'])
        positions, strata = matched_positions(users)
        selected = users.iloc[positions]
        self.assertEqual(set(selected.index), {'account_a','account_b','account_d','account_e'})
        self.assertEqual(selected['label'].value_counts().to_dict(), {0:2,1:2})
        altered = users.iloc[::-1].copy()
        altered['probability'] = 1-altered['probability']
        reordered, _ = matched_positions(altered)
        self.assertEqual(set(selected.index), set(altered.iloc[reordered].index))
        self.assertTrue(any(x['kept_per_class']==0 for x in strata))
        with self.assertRaisesRegex(ValueError, 'one row'):
            matched_positions(pd.concat([users,users.iloc[:1]]))


if __name__ == '__main__':
    unittest.main()
