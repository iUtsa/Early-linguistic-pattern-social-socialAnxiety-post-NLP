"""
Text embedding utilities using Sentence Transformers.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import hashlib
import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
try:
    from .research_validation import validate_author_metadata, validate_finite_matrix
except ImportError:
    from research_validation import validate_author_metadata, validate_finite_matrix


class EmbeddingCache:
    """Cache for embeddings to avoid recomputation."""
    
    def __init__(self, cache_dir='data/processed'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_path(self, model_name, dataset_name='rmhd'):
        """Get path to cache file."""
        safe_model_name = model_name.replace('/', '_')
        return self.cache_dir / f'embeddings_{safe_model_name}_{dataset_name}.npy'
    
    def load(self, model_name, dataset_name='rmhd', fingerprint=None):
        """Load cached embeddings."""
        cache_path = self.get_cache_path(model_name, dataset_name)
        metadata_path = cache_path.with_suffix('.json')
        if cache_path.exists() and metadata_path.exists() and fingerprint is not None:
            metadata = json.loads(metadata_path.read_text())
            if metadata.get('fingerprint') != fingerprint or metadata.get('model') != model_name:
                return None
            if hashlib.sha256(cache_path.read_bytes()).hexdigest() != metadata.get('sha256'):
                raise ValueError('Embedding cache checksum mismatch')
            print(f"Loading cached embeddings from {cache_path}")
            return np.load(cache_path, allow_pickle=False)
        return None
    
    def save(self, embeddings, model_name, dataset_name='rmhd', fingerprint=None):
        """Save embeddings to cache."""
        cache_path = self.get_cache_path(model_name, dataset_name)
        np.save(cache_path, embeddings)
        cache_path.with_suffix('.json').write_text(json.dumps({
            'model': model_name, 'fingerprint': fingerprint,
            'sha256': hashlib.sha256(cache_path.read_bytes()).hexdigest()
        }))
        print(f"Saved embeddings to {cache_path}")


def encode_texts(texts, model_name='all-MiniLM-L6-v2', batch_size=32, 
                 show_progress=True, use_cache=True, cache_name='rmhd'):
    """
    Encode texts using Sentence Transformer.
    
    Args:
        texts: List or Series of texts
        model_name: Name of sentence transformer model
        batch_size: Batch size for encoding
        show_progress: Show progress bar
        use_cache: Use cached embeddings if available
        cache_name: Name for cache file
        
    Returns:
        np.ndarray: Embeddings (n_texts, embedding_dim)
    """
    texts = [str(text) if text is not None else '' for text in texts]
    fingerprint = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()
    # Try loading from cache; equal row counts alone do not establish identity.
    if use_cache:
        cache = EmbeddingCache()
        cached_embeds = cache.load(model_name, cache_name, fingerprint=fingerprint)
        if cached_embeds is not None and len(cached_embeds) == len(texts):
            return cached_embeds
    
    print(f"Encoding {len(texts)} texts with {model_name}...")
    
    # Load model
    model = SentenceTransformer(model_name)
    
    # Convert to list if needed
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    
    # Ensure all texts are strings
    texts = [str(text) if text else "" for text in texts]
    
    # Encode
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True
    )
    
    # Save to cache
    if use_cache:
        cache = EmbeddingCache()
        cache.save(embeddings, model_name, cache_name, fingerprint=fingerprint)
    
    return embeddings


def aggregate_user_embeddings(embeddings, posts_df, aggregation='mean'):
    """
    Aggregate post-level embeddings to user level.
    
    Args:
        embeddings: Post-level embeddings (n_posts, dim)
        posts_df: DataFrame with author, label, split
        aggregation: Aggregation method ('mean', 'max', or 'last')
        
    Returns:
        tuple: (user_embeddings, user_labels, user_splits)
    """
    print(f"Aggregating embeddings to user level (method: {aggregation})...")
    
    validate_author_metadata(posts_df)
    embeddings = validate_finite_matrix(embeddings, len(posts_df))
    # Keep metadata and embeddings in their original common row order.
    df = posts_df[['author', 'label', 'split']].copy()
    if aggregation == 'last':
        if 'posts_seen' not in posts_df:
            raise ValueError('Last-post aggregation requires observed ordering in posts_seen')
        df['posts_seen'] = posts_df['posts_seen'].to_numpy()
    
    df = df.reset_index(drop=True)
    
    # Group by author
    unique_authors = sorted(df['author'].unique())
    user_embeddings = []
    user_labels = []
    user_splits = []
    
    for author in tqdm(unique_authors, desc="Aggregating users"):
        mask = df['author'] == author
        user_posts_embeds = embeddings[mask]
        
        # Aggregate
        if aggregation == 'mean':
            user_embed = user_posts_embeds.mean(axis=0)
        elif aggregation == 'max':
            user_embed = user_posts_embeds.max(axis=0)
        elif aggregation == 'last':
            positions = np.flatnonzero(mask.to_numpy())
            last_position = positions[np.argmax(df.loc[mask, 'posts_seen'].to_numpy())]
            user_embed = embeddings[last_position]
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        user_embeddings.append(user_embed)
        
        # Get label and split (should be same for all posts)
        user_labels.append(df[mask]['label'].iloc[0])
        user_splits.append(df[mask]['split'].iloc[0])
    
    user_embeddings = np.array(user_embeddings)
    user_labels = np.array(user_labels)
    user_splits = np.array(user_splits)
    
    print(f"Aggregated to {len(user_embeddings)} users")
    
    return user_embeddings, user_labels, user_splits


def reduce_dimensions(embeddings, n_components=300, method='pca', training_data=None):
    """
    Reduce embedding dimensions using PCA.
    
    Args:
        embeddings: Input embeddings (n_samples, dim)
        n_components: Target dimensions
        method: 'pca' (more options could be added)
        
    Returns:
        tuple: (reduced_embeddings, reducer_model)
    """
    from sklearn.decomposition import PCA
    
    if embeddings.shape[1] <= n_components:
        print(f"Embeddings already {embeddings.shape[1]}D, skipping reduction")
        return embeddings, None
    
    print(f"Reducing embeddings from {embeddings.shape[1]}D to {n_components}D using {method.upper()}...")
    
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
        fit_data = embeddings if training_data is None else training_data
        reducer.fit(fit_data)
        reduced = reducer.transform(embeddings)
        variance_explained = reducer.explained_variance_ratio_.sum()
        print(f"Variance explained: {variance_explained:.3f}")
        return reduced, reducer
    else:
        raise ValueError(f"Unknown reduction method: {method}")


def split_embeddings(embeddings, labels, splits):
    """
    Split embeddings into train/val/test sets.
    
    Args:
        embeddings: Embeddings array
        labels: Labels array
        splits: Splits array ('train', 'val', 'test')
        
    Returns:
        dict: Dictionary with train/val/test embeddings and labels
    """
    data = {}
    
    for split in ['train', 'val', 'test']:
        mask = splits == split
        data[f'{split}_X'] = embeddings[mask]
        data[f'{split}_y'] = labels[mask]
        print(f"{split}: {data[f'{split}_X'].shape[0]} samples")
    
    return data
