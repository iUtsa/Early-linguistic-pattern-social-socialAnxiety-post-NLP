"""Prepare private, time-forward community cohorts from the supplied CSV manifest.

Only aggregate counts are printed. No diagnosis or healthy-control label is inferred.
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .preprocess import clean_text
from .research_benchmark import audit_posts


SCHEMA = {'author', 'created_utc', 'score', 'selftext', 'subreddit', 'title'}
COMMUNITIES = {'anxiety', 'mentalhealth', 'depression', 'lonely', 'suicidewatch'}
PLACEHOLDERS = {'', 'none', 'nan', 'null', 'deleted', 'removed', 'unknown', 'automoderator'}
WINDOWS = [('2022-03', 'train'), ('2022-05', 'val'), ('2022-06', 'test')]


def load_authored(manifest_path):
    """Verify inputs and return schema-compatible CSV rows without echoing cells."""
    manifest = json.loads(Path(manifest_path).read_text())
    frames, files, seen = [], [], set()
    for item in manifest:
        if not item['name'].lower().endswith('.csv'):
            continue
        digest = hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()
        if digest != item['sha256']:
            raise ValueError('An uploaded file does not match its recorded checksum')
        file_report = {'name': item['name'], 'sha256': digest}
        if digest in seen:
            files.append({**file_report, 'status': 'duplicate_file_skipped'})
            continue
        seen.add(digest)
        frame = pd.read_csv(item['path'], low_memory=False)
        if not SCHEMA <= set(frame.columns):
            files.append({**file_report, 'status': 'schema_not_used', 'rows': len(frame)})
            continue
        frames.append(frame[list(SCHEMA)].assign(_file=item['name']))
        files.append({**file_report, 'status': 'authored_schema_loaded', 'rows': len(frame)})
    if not frames:
        raise ValueError('No observed-author CSVs found in manifest')
    return pd.concat(frames, ignore_index=True), files


def prepare_cohort(raw, comparison):
    """Build selected monthly cohorts; future labels/text never alter training rows."""
    if comparison not in {'mentalhealth', 'depression'}:
        raise ValueError('Comparison must be mentalhealth or depression')
    report = {'comparison': comparison, 'loaded_authored_rows': len(raw), 'exclusions': {}}
    d = raw.copy()
    d['_community'] = d['subreddit'].fillna('').astype(str).str.strip().str.casefold()
    canonical = d['_community'].isin(COMMUNITIES)
    report['exclusions']['noncanonical_community_rows_all_authored_files'] = int((~canonical).sum())
    d = d[d['_community'].isin({'anxiety', comparison})].copy()
    report['selected_community_rows'] = len(d)
    d['author'] = d['author'].fillna('').astype(str).str.strip().str.casefold()
    valid_author = d['author'].str.fullmatch(r'[a-z0-9_-]{3,20}', na=False) & ~d['author'].isin(PLACEHOLDERS)
    report['exclusions']['invalid_or_placeholder_author_rows'] = int((~valid_author).sum())
    d = d[valid_author].copy()
    d['created_utc'] = pd.to_numeric(d['created_utc'], errors='coerce')
    d['_time'] = pd.to_datetime(d['created_utc'], unit='s', utc=True, errors='coerce')
    valid_time = d['_time'].ge(pd.Timestamp('2022-01-01', tz='UTC')) & d['_time'].lt(pd.Timestamp('2023-01-01', tz='UTC'))
    valid_score = pd.to_numeric(d['score'], errors='coerce').notna()
    valid_meta = valid_time & valid_score
    report['exclusions']['invalid_numeric_time_or_score_rows'] = int((~valid_meta).sum())
    d = d[valid_meta].copy()
    report['metadata_valid_rows'] = len(d)
    d['_month'] = d['_time'].dt.strftime('%Y-%m')
    d['_body'] = d['selftext'].fillna('').map(clean_text)
    d['_title'] = d['title'].fillna('').map(clean_text)
    good_body = ~d['_body'].isin(['', '[deleted]', '[removed]']) & d['_body'].str.split().map(len).ge(10)
    d['_good_body'] = good_body
    d['text'] = (d['_title'] + ' ' + d['_body']).str.strip()
    d['_text_hash'] = d['text'].map(lambda text: hashlib.sha256(text.encode()).hexdigest())
    report['exclusions']['short_empty_or_removed_body_rows_all_dates'] = int((~good_body).sum())
    # A content hash is not an original platform post ID.
    identical_record = d.duplicated(['author', 'created_utc', '_community', '_text_hash'])
    report['exclusions']['identical_record_rows'] = int(identical_record.sum())
    d = d[~identical_record].copy()
    kept, seen_text, window_reports = [], set(), {}
    for month, split in WINDOWS:
        start = pd.Timestamp(month + '-01', tz='UTC')
        current = d[d['_month'].eq(month)].copy()
        wr = {'metadata_valid_posts': len(current), 'metadata_valid_authors': int(current['author'].nunique())}
        mixed = current.groupby('author')['_community'].nunique().loc[lambda x: x > 1].index
        wr['ambiguous_community_authors'] = len(mixed)
        wr['ambiguous_community_posts'] = int(current['author'].isin(mixed).sum())
        current = current[~current['author'].isin(mixed)].copy()
        earlier_authors = set(d.loc[d['_time'].lt(start), 'author']) if split != 'train' else set()
        wr['previously_observed_authors_excluded'] = int(current.loc[current['author'].isin(earlier_authors), 'author'].nunique())
        wr['previously_observed_posts_excluded'] = int(current['author'].isin(earlier_authors).sum())
        current = current[~current['author'].isin(earlier_authors)].copy()
        wr['text_quality_posts_excluded'] = int((~current['_good_body']).sum())
        current = current[current['_good_body']].copy()
        shared = current.groupby('_text_hash')['author'].nunique().loc[lambda x: x > 1].index
        wr['shared_text_posts_quarantined_within_window'] = int(current['_text_hash'].isin(shared).sum())
        current = current[~current['_text_hash'].isin(shared)].copy()
        current = current.sort_values(['created_utc', 'author'], kind='stable')
        duplicate = current['_text_hash'].duplicated()
        wr['repeated_same_author_text_rows'] = int(duplicate.sum())
        current = current[~duplicate].copy()
        cross_window = current['_text_hash'].isin(seen_text)
        wr['earlier_window_duplicate_text_rows_excluded'] = int(cross_window.sum())
        current = current[~cross_window].copy()
        seen_text.update(current['_text_hash'])
        current['label'] = current['_community'].eq('anxiety').astype(int)
        current['split'] = split
        wr['retained_posts'] = len(current)
        wr['retained_authors'] = int(current['author'].nunique())
        wr['positive_authors'] = int(current.loc[current['label'].eq(1), 'author'].nunique())
        wr['negative_authors'] = wr['retained_authors'] - wr['positive_authors']
        wr['retained_posts_by_label'] = {str(k): int(v) for k, v in current['label'].value_counts().sort_index().items()}
        wr['authors_with_at_least_3_posts_by_label'] = {}
        for label in [0, 1]:
            counts = current.loc[current['label'].eq(label)].groupby('author').size()
            wr['authors_with_at_least_3_posts_by_label'][str(label)] = int(counts.ge(3).sum())
        wr['author_timestamp_tied_rows'] = int(current.duplicated(['author', 'created_utc'], keep=False).sum())
        wr['utc_start'] = current['_time'].min().isoformat() if len(current) else None
        wr['utc_end'] = current['_time'].max().isoformat() if len(current) else None
        window_reports[split] = wr
        kept.append(current[['author', 'text', 'label', 'split', 'created_utc']])
    result = pd.concat(kept, ignore_index=True)
    result, audit = audit_posts(result)
    report['windows'] = window_reports
    report['cohort_integrity'] = audit
    report['observed_nonmodel_month_rows'] = int((~d['_month'].isin([m for m, _ in WINDOWS])).sum())
    return result, report


def prepare(manifest, comparison, output):
    output = Path(output).resolve()
    if output.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError('Keep prepared private data outside the Git checkout')
    if output.exists():
        raise ValueError('Use a new output directory')
    raw, files = load_authored(manifest)
    posts, report = prepare_cohort(raw, comparison)
    output.mkdir(parents=True, mode=0o700)
    report['files'] = files
    report['protocol_sha256'] = hashlib.sha256((Path(__file__).resolve().parents[1]/'research/UPLOADED_PILOT_PROTOCOL.md').read_bytes()).hexdigest()
    report['preparation_code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    posts.to_csv(output/'posts.csv', index=False)
    report['prepared_csv_sha256'] = hashlib.sha256((output/'posts.csv').read_bytes()).hexdigest()
    provenance = {
        'dataset_source': 'Author-supplied local CSV examples; upstream dataset URL not provided or verified',
        'dataset_version': 'Uploaded file SHA-256 manifest; upstream version and completeness unknown',
        'permission_basis': 'Author requested local exploratory analysis of their uploads; upstream terms and redistribution rights unverified',
        'label_definition': f'Exclusive observed r/Anxiety (1) versus r/{comparison} (0) affiliation in the assigned month; community proxy, not diagnosis',
        'ethics_status': 'Institutional approval/exemption, collection consent basis and data-use determination not supplied; local private exploratory pilot only',
        'author_id_origin': 'observed_stable', 'synthetic_data': False,
        'split_policy': 'UTC March train, May validation, June test; later authors absent from all supplied earlier selected-community records',
        'protocol_sha256': report['protocol_sha256'],
    }
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2))
    (output/'cohort_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    for path in output.iterdir():
        path.chmod(0o600)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--comparison', choices=['mentalhealth', 'depression'], required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = prepare(args.manifest, args.comparison, args.output)
    print(json.dumps({'comparison': report['comparison'], 'counts': report['cohort_integrity']['split_counts']}, indent=2))


if __name__ == '__main__':
    main()
