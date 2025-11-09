"""
Build unified RMHD dataset from raw CSV files.
Processes each CSV individually before combining.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

ANXIETY_SUBS = {
    'anxiety', 'socialanxiety', 'panicdisorder', 
    'anxietysupport', 'anxietyhelp', 'healthanxiety'
}

CONTROL_SUBS = {
    'askreddit', 'ask_reddit', 'movies', 'technology', 'gaming', 'books',
    'music', 'aww', 'pics', 'funny', 'food', 'lifehacks'
}

EXCLUDED_SUBS = {
    'depression', 'lonely', 'mentalhealth', 'schizophrenia'
}


def normalize_subreddit(sub):
    if pd.isna(sub):
        return None
    sub = str(sub).lower().strip()
    if sub.startswith('r/'):
        sub = sub[2:]
    if sub == 'ask_reddit':
        sub = 'askreddit'
    return sub


def normalize_single_csv(df, filename):
    """Normalize one CSV file."""
    norm = pd.DataFrame()
    
    # ID
    for col in ['id', 'post_id', 'submission_id']:
        if col in df.columns:
            norm['id'] = df[col].astype(str)
            break
    if 'id' not in norm.columns:
        norm['id'] = [f"{filename}_{i}" for i in range(len(df))]
    
    # Author
    for col in ['author', 'username']:
        if col in df.columns:
            norm['author'] = df[col].astype(str)
            break
    if 'author' not in norm.columns:
        norm['author'] = 'user_' + norm['id']
    
    # Text
    text_found = False
    for col in ['post', 'body', 'text', 'selftext']:
        if col in df.columns:
            norm['text'] = df[col].fillna('').astype(str)
            text_found = True
            break
    
    if 'title' in df.columns:
        title = df['title'].fillna('').astype(str)
        if text_found:
            body = norm['text']
            norm['text'] = (title.str.strip() + ' ' + body.str.strip()).str.strip()
        else:
            norm['text'] = title.str.strip()
            text_found = True
    
    if not text_found:
        raise ValueError(f"No text column in {filename}")
    
    # Timestamp - try each column for THIS file
    ts_found = False
    for col in ['created', 'created_utc', 'date', 'timestamp']:
        if col not in df.columns:
            continue
        
        try:
            col_data = df[col]
            if col_data.isna().all():
                continue
            
            # Numeric timestamp
            if pd.api.types.is_numeric_dtype(col_data):
                ts = pd.to_numeric(col_data, errors='coerce')
                valid = ts.notna()
                if valid.sum() > len(ts) * 0.5:
                    # Check if in reasonable range
                    valid_vals = ts[valid]
                    in_range = ((valid_vals > 946684800) & (valid_vals < 1893456000)).sum()
                    if in_range > valid.sum() * 0.5:
                        norm['timestamp'] = ts
                        ts_found = True
                        break
            
            # String timestamp
            else:
                parsed = pd.to_datetime(col_data, errors='coerce')
                if parsed.notna().sum() > len(parsed) * 0.5:
                    norm['timestamp'] = parsed.astype('int64') / 10**9
                    ts_found = True
                    break
        except:
            continue
    
    if not ts_found:
        raise ValueError(f"No valid timestamp in {filename}")
    
    # Subreddit
    if 'subreddit' in df.columns:
        norm['subreddit'] = df['subreddit']
    else:
        # Infer from filename
        name = filename.lower().replace('_', '').replace('-', '')
        for prefix in ['features', 'tfidf', '256', '2018', '2019', '2020', 'pre', 'post']:
            name = name.replace(prefix, '')
        name = name.strip('_- ')
        norm['subreddit'] = name if name else 'unknown'
    
    return norm


def load_and_normalize_all(src_dir):
    """Load each CSV, normalize it, then combine."""
    src_path = Path(src_dir)
    if not src_path.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")
    
    csv_files = list(src_path.rglob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {src_dir}")
    
    print(f"Found {len(csv_files)} CSV files\n")
    
    all_normalized = []
    stats = {}
    
    for csv_file in tqdm(csv_files, desc="Processing CSVs"):
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            initial_rows = len(df)
            
            # Normalize this CSV
            df_norm = normalize_single_csv(df, csv_file.stem)
            
            all_normalized.append(df_norm)
            stats[csv_file.name] = initial_rows
            
        except Exception as e:
            print(f"\nWarning: Skipped {csv_file.name}: {e}")
    
    if not all_normalized:
        raise ValueError("Failed to process any CSV files")
    
    combined = pd.concat(all_normalized, ignore_index=True)
    
    print(f"\nLoaded {len(combined):,} total rows from {len(all_normalized)} files")
    print(f"\nTop 5 files by size:")
    for name, count in sorted(stats.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {name}: {count:,} rows")
    
    return combined


def clean_data(df):
    print("\nCleaning data...")
    initial = len(df)
    
    df = df[~df['text'].str.lower().isin(['[deleted]', '[removed]'])]
    df = df[df['timestamp'].notna() & (df['timestamp'] > 0)]
    df = df[df['text'].str.len() >= 10]
    
    print(f"  {initial:,} -> {len(df):,} rows ({initial - len(df):,} removed)")
    return df


def assign_labels(df):
    print("\nAssigning labels...")
    
    df['subreddit_norm'] = df['subreddit'].apply(normalize_subreddit)
    
    print(f"Unique subreddits: {df['subreddit_norm'].nunique()}")
    for sub in sorted(df['subreddit_norm'].value_counts().head(10).items(), key=lambda x: x[1], reverse=True):
        print(f"  {sub[0]}: {sub[1]:,}")
    
    anxiety_mask = df['subreddit_norm'].isin(ANXIETY_SUBS)
    control_mask = df['subreddit_norm'].isin(CONTROL_SUBS)
    excluded_mask = df['subreddit_norm'].isin(EXCLUDED_SUBS)
    
    print(f"\nCategories:")
    print(f"  Anxiety: {anxiety_mask.sum():,}")
    print(f"  Control: {control_mask.sum():,}")
    print(f"  Excluded: {excluded_mask.sum():,}")
    
    df = df[anxiety_mask | control_mask].copy()
    df['label'] = df['subreddit_norm'].apply(lambda x: 1 if x in ANXIETY_SUBS else 0)
    
    print(f"\nFinal:")
    print(f"  Anxiety (1): {(df['label']==1).sum():,}")
    print(f"  Control (0): {(df['label']==0).sum():,}")
    
    if (df['label']==0).sum() == 0:
        raise ValueError("No control posts! Check AskReddit CSV processing.")
    
    return df


def process_users(df, max_posts=20, min_posts=3):
    print(f"\nProcessing users (max={max_posts}, min={min_posts})...")
    
    df = df.sort_values(['author', 'timestamp']).reset_index(drop=True)
    df['posts_seen'] = df.groupby('author').cumcount() + 1
    df = df[df['posts_seen'] <= max_posts].copy()
    
    counts = df.groupby('author').size()
    valid = counts[counts >= min_posts].index
    df = df[df['author'].isin(valid)].copy()
    
    print(f"  Users: {df['author'].nunique():,}, Posts: {len(df):,}")
    return df


def balance_users(df):
    print("\nBalancing classes...")
    
    user_labels = df.groupby('author')['label'].agg(lambda x: x.mode()[0])
    anx = user_labels[user_labels == 1].index.tolist()
    ctrl = user_labels[user_labels == 0].index.tolist()
    
    print(f"  Anxiety: {len(anx):,}, Control: {len(ctrl):,}")
    
    min_users = min(len(anx), len(ctrl))
    if min_users == 0:
        raise ValueError("Cannot balance - one class has 0 users")
    
    np.random.seed(42)
    balanced = list(np.random.choice(anx, min_users, False)) + \
               list(np.random.choice(ctrl, min_users, False))
    
    df = df[df['author'].isin(balanced)].copy()
    print(f"  Balanced: {len(balanced):,} users ({min_users:,} per class)")
    return df


def create_splits(df):
    print("\nCreating splits...")
    
    user_labels = df.groupby('author')['label'].agg(lambda x: x.mode()[0])
    anx = user_labels[user_labels == 1].index.tolist()
    ctrl = user_labels[user_labels == 0].index.tolist()
    
    np.random.seed(42)
    
    def split(users):
        n = len(users)
        users = np.random.permutation(users)
        return users[:int(0.7*n)], users[int(0.7*n):int(0.85*n)], users[int(0.85*n):]
    
    a_tr, a_val, a_te = split(anx)
    c_tr, c_val, c_te = split(ctrl)
    
    train = set(list(a_tr) + list(c_tr))
    val = set(list(a_val) + list(c_val))
    test = set(list(a_te) + list(c_te))
    
    df['split'] = df['author'].map(lambda a: 'train' if a in train else ('val' if a in val else 'test'))
    
    print(f"  Train: {len(train):,}, Val: {len(val):,}, Test: {len(test):,}")
    return df


def save_outputs(df, posts_path, users_path):
    print("\nSaving...")
    
    Path(posts_path).parent.mkdir(parents=True, exist_ok=True)
    
    cols = ['id', 'author', 'text', 'timestamp', 'subreddit', 'label', 'posts_seen', 'split']
    df[cols].to_csv(posts_path, index=False)
    
    users = df.groupby('author').agg({
        'label': lambda x: x.mode()[0],
        'split': 'first',
        'posts_seen': 'max',
        'timestamp': ['min', 'max']
    }).reset_index()
    users.columns = ['author', 'label', 'split', 'num_posts', 'first_post', 'last_post']
    users.to_csv(users_path, index=False)
    
    print(f"  Posts: {posts_path}")
    print(f"  Users: {users_path}")
    print(f"\n  Final: {len(df):,} posts, {df['author'].nunique():,} users")
    print(f"  Anxiety: {(df['label']==1).sum():,}, Control: {(df['label']==0).sum():,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', required=True)
    parser.add_argument('--out_posts', required=True)
    parser.add_argument('--out_users', required=True)
    parser.add_argument('--max_posts_per_user', type=int, default=20)
    parser.add_argument('--min_posts_per_user', type=int, default=3)
    args = parser.parse_args()
    
    print("="*60)
    print("Building RMHD Dataset")
    print("="*60)
    
    df = load_and_normalize_all(args.src_dir)
    df = clean_data(df)
    df = assign_labels(df)
    df = process_users(df, args.max_posts_per_user, args.min_posts_per_user)
    df = balance_users(df)
    df = create_splits(df)
    save_outputs(df, args.out_posts, args.out_users)
    
    print("\n" + "="*60)
    print("Complete!")
    print("="*60)


if __name__ == '__main__':
    main()