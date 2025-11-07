#!/usr/bin/env python3
"""
Fix dataset split to properly stratify by both author AND label
This version handles the mixed dtypes warning and ensures both labels appear in all splits
"""
import pandas as pd
import numpy as np

# Load with proper dtype handling
print("Loading dataset...")
df = pd.read_csv('data/raw/final_dataset.csv', low_memory=False)

print(f"\nOriginal: {len(df):,} samples")
print(f"Authors: {df['author'].nunique():,}")

# Show current problematic split
print("\nCurrent split (BROKEN):")
current_split = df.groupby(['split', 'label']).size().unstack(fill_value=0)
print(current_split)
print("\n⚠ Problem: Control (label=0) only in train, not in val/test!")

# Check label distribution
print(f"\nLabel distribution:")
print(df['label'].value_counts())

# Get each author's primary label
print("\nGrouping authors by label...")
author_labels = df.groupby('author')['label'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])

# Separate authors by label
anxiety_authors = author_labels[author_labels == 1].index.tolist()
control_authors = author_labels[author_labels == 0].index.tolist()

print(f"\nAuthors by label:")
print(f"  Anxiety (label=1): {len(anxiety_authors):,} authors")
print(f"  Control (label=0): {len(control_authors):,} authors")

# Check if we have control authors
if len(control_authors) == 0:
    print("\n⚠ ERROR: No control authors found!")
    print("Checking label values in dataset:")
    print(df['label'].unique())
    
    # Check if control posts exist but weren't grouped properly
    control_posts = df[df['label'] == 0]
    print(f"\nControl posts: {len(control_posts):,}")
    print(f"Control authors in posts: {control_posts['author'].nunique():,}")
    
    # Re-extract control authors directly from posts
    control_authors = control_posts['author'].unique().tolist()
    print(f"Re-extracted control authors: {len(control_authors):,}")

# Shuffle with seed for reproducibility
np.random.seed(42)
np.random.shuffle(anxiety_authors)
np.random.shuffle(control_authors)

print(f"\nAfter shuffle:")
print(f"  Anxiety: {len(anxiety_authors):,} authors")
print(f"  Control: {len(control_authors):,} authors")

# Split 70/15/15 within each label group
def split_authors(authors, label_name):
    n = len(authors)
    if n == 0:
        print(f"  ⚠ WARNING: No authors for {label_name}")
        return {'train': set(), 'val': set(), 'test': set()}
    
    train_idx = int(0.70 * n)
    val_idx = int(0.85 * n)
    
    splits = {
        'train': set(authors[:train_idx]),
        'val': set(authors[train_idx:val_idx]),
        'test': set(authors[val_idx:])
    }
    
    print(f"\n  {label_name} split:")
    print(f"    train: {len(splits['train']):,} authors")
    print(f"    val:   {len(splits['val']):,} authors")
    print(f"    test:  {len(splits['test']):,} authors")
    
    return splits

print("\nCreating splits:")
anxiety_splits = split_authors(anxiety_authors, "Anxiety")
control_splits = split_authors(control_authors, "Control")

# Assign splits to each row
def assign_split(row):
    author = row['author']
    label = row['label']
    
    # Choose the appropriate split dictionary based on label
    if label == 1:  # Anxiety
        splits = anxiety_splits
    elif label == 0:  # Control
        splits = control_splits
    else:
        print(f"  ⚠ Unknown label: {label} for author {author}")
        return 'unknown'
    
    # Find which split this author belongs to
    for split_name, author_set in splits.items():
        if author in author_set:
            return split_name
    
    return 'unknown'

print("\nAssigning splits to all posts...")
df['split'] = df.apply(assign_split, axis=1)

# Check for unknown splits
unknown_count = (df['split'] == 'unknown').sum()
if unknown_count > 0:
    print(f"⚠ WARNING: {unknown_count} posts assigned to 'unknown' split")

# Verify the new split
print("\n" + "="*60)
print("NEW SPLIT DISTRIBUTION:")
print("="*60)
new_split = df.groupby(['split', 'label']).size().unstack(fill_value=0)
print(new_split)

# Show percentages
print("\nPercentages by split:")
for split in ['train', 'val', 'test']:
    split_df = df[df['split'] == split]
    if len(split_df) > 0:
        total = len(split_df)
        anxiety = (split_df['label'] == 1).sum()
        control = (split_df['label'] == 0).sum()
        print(f"  {split}: {total:,} total")
        print(f"    - Anxiety: {anxiety:,} ({anxiety/total*100:.1f}%)")
        print(f"    - Control: {control:,} ({control/total*100:.1f}%)")

# Recalculate posts_seen to ensure it's correct
print("\nRecalculating posts_seen...")
df = df.sort_values(['author', 'created_utc'])
df['posts_seen'] = df.groupby('author').cumcount() + 1

# Check for author leakage
print("\n" + "="*60)
print("LEAKAGE CHECK:")
print("="*60)
train_authors = set(df[df['split'] == 'train']['author'])
val_authors = set(df[df['split'] == 'val']['author'])
test_authors = set(df[df['split'] == 'test']['author'])

train_val_overlap = len(train_authors & val_authors)
train_test_overlap = len(train_authors & test_authors)
val_test_overlap = len(val_authors & test_authors)

print(f"Train/Val overlap:  {train_val_overlap} authors")
print(f"Train/Test overlap: {train_test_overlap} authors")
print(f"Val/Test overlap:   {val_test_overlap} authors")

if train_test_overlap == 0 and val_test_overlap == 0:
    print("\n✓ No leakage detected!")
else:
    print("\n⚠ WARNING: Author leakage detected!")

# Verify both labels in each split
print("\n" + "="*60)
print("SPLIT VALIDATION:")
print("="*60)
all_good = True
for split in ['train', 'val', 'test']:
    split_labels = df[df['split'] == split]['label'].unique()
    if len(split_labels) < 2:
        print(f"⚠ {split}: MISSING LABEL {0 if 0 not in split_labels else 1}!")
        all_good = False
    else:
        print(f"✓ {split}: Both labels present")

if all_good:
    print("\n✓ All splits have both labels!")
else:
    print("\n⚠ ERROR: Some splits are missing labels!")
    print("This will cause the model to fail on those splits.")

# Save the fixed dataset
output_path = 'data/raw/final_dataset_fixed.csv'
df.to_csv(output_path, index=False)
print(f"\n{'='*60}")
print(f"✓ Saved to: {output_path}")
print(f"{'='*60}")

print("\nNext steps:")
print("  1. Update config.yaml to use 'data/raw/final_dataset_fixed.csv'")
print("  2. Run: make validate")
print("  3. Run: make full")
print("  4. Run: make early")