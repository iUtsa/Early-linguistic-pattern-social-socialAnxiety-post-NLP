#!/usr/bin/env python3
"""
Fix data leakage by ensuring author-disjoint splits
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

print("Loading data...")
df = pd.read_csv('data/processed/posts_with_features.csv')

print(f"Original shape: {df.shape}")
print(f"Original splits: {df['split'].value_counts()}")

# Remove old splits
df = df.drop(columns=['split'], errors='ignore')

print("\n" + "="*80)
print("CREATING AUTHOR-DISJOINT SPLITS")
print("="*80)

# Group by author
author_labels = df.groupby('author')['label'].agg(lambda x: x.mode()[0])
authors = pd.DataFrame({
    'author': author_labels.index,
    'label': author_labels.values
})

print(f"\nTotal unique authors: {len(authors):,}")
print(f"  Anxiety authors: {(authors['label']==1).sum():,}")
print(f"  Control authors: {(authors['label']==0).sum():,}")

# Split authors (not posts!) into train/val/test
train_authors, temp_authors = train_test_split(
    authors['author'], 
    test_size=0.30,
    random_state=42,
    stratify=authors['label']
)

val_authors, test_authors = train_test_split(
    temp_authors,
    test_size=0.50,
    random_state=42,
    stratify=authors.loc[authors['author'].isin(temp_authors), 'label']
)

print(f"\nAuthor splits:")
print(f"  Train authors: {len(train_authors):,} (70%)")
print(f"  Val authors:   {len(val_authors):,} (15%)")
print(f"  Test authors:  {len(test_authors):,} (15%)")

# Assign splits to posts based on author
df['split'] = 'unknown'
df.loc[df['author'].isin(train_authors), 'split'] = 'train'
df.loc[df['author'].isin(val_authors), 'split'] = 'validation'
df.loc[df['author'].isin(test_authors), 'split'] = 'test'

print(f"\nPost splits:")
for split in ['train', 'validation', 'test']:
    count = (df['split'] == split).sum()
    pct = count / len(df) * 100
    print(f"  {split}: {count:,} posts ({pct:.1f}%)")

# Verify no overlap
train_auth_set = set(df[df['split'] == 'train']['author'].unique())
val_auth_set = set(df[df['split'] == 'validation']['author'].unique())
test_auth_set = set(df[df['split'] == 'test']['author'].unique())

overlap_train_val = len(train_auth_set & val_auth_set)
overlap_train_test = len(train_auth_set & test_auth_set)
overlap_val_test = len(val_auth_set & test_auth_set)

print("\n" + "="*80)
print("VERIFICATION")
print("="*80)
print(f"Train-Val overlap: {overlap_train_val} authors")
print(f"Train-Test overlap: {overlap_train_test} authors")
print(f"Val-Test overlap: {overlap_val_test} authors")

if overlap_train_val == 0 and overlap_train_test == 0 and overlap_val_test == 0:
    print("\n✅ SUCCESS: All splits are author-disjoint!")
else:
    print("\n🚨 ERROR: Overlaps still exist!")

# Check class balance
for split in ['train', 'validation', 'test']:
    split_data = df[df['split'] == split]
    balance = split_data['label'].value_counts()
    pct_pos = balance[1] / len(split_data) * 100
    print(f"\n{split}: {pct_pos:.1f}% positive class")

# Save fixed dataset
output_path = 'data/processed/posts_with_features_fixed.csv'
df.to_csv(output_path, index=False)
print(f"\n✅ Saved fixed dataset to: {output_path}")
print(f"   Shape: {df.shape}")

