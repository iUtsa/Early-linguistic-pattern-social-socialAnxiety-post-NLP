"""Fail-fast checks for author-level experiments; never resolve labels by majority vote."""

import numpy as np


def validate_author_metadata(posts):
    required = {'author', 'label', 'split'}
    missing = required - set(posts.columns)
    if missing:
        raise ValueError(f'Missing author metadata: {sorted(missing)}')
    if posts.empty or posts[list(required)].isna().any().any():
        raise ValueError('Author metadata must be nonempty and non-null')
    authors = posts['author'].astype(str).str.strip()
    if authors.str.lower().isin(['', '[deleted]', '[removed]', 'none', 'nan']).any():
        raise ValueError('Missing or placeholder author identifiers cannot define independent users')
    if not set(posts['label'].unique()) <= {0, 1}:
        raise ValueError('Labels must be binary integers 0 and 1')
    grouped = posts.groupby('author', sort=True)
    if (grouped['label'].nunique() != 1).any():
        raise ValueError('An author has conflicting labels; define the label construct first')
    if (grouped['split'].nunique() != 1).any():
        raise ValueError('Author leakage: an author appears in multiple splits')
    if not set(posts['split'].unique()) <= {'train', 'val', 'test'}:
        raise ValueError('Use canonical split names train, val, test')
    return posts


def validate_finite_matrix(matrix, n_rows=None):
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError('Features must be a finite two-dimensional matrix')
    if n_rows is not None and len(matrix) != n_rows:
        raise ValueError('Feature rows and post rows must have the same length')
    return matrix
