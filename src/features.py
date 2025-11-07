"""
Linguistic feature extraction.
"""

import re
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import emoji
from tqdm import tqdm


# Initialize VADER
vader = SentimentIntensityAnalyzer()


# First-person singular pronouns
FP_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'myself',
    "i'm", "i've", "i'll", "i'd"
}


def extract_vader_sentiment(text):
    """
    Extract VADER sentiment scores.
    
    Args:
        text: Input text
        
    Returns:
        dict: Sentiment scores (neg, neu, pos, compound)
    """
    if not isinstance(text, str) or len(text) == 0:
        return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    
    scores = vader.polarity_scores(text)
    return scores


def extract_textblob_sentiment(text):
    """
    Extract TextBlob sentiment scores.
    
    Args:
        text: Input text
        
    Returns:
        dict: Sentiment scores (polarity, subjectivity)
    """
    if not isinstance(text, str) or len(text) == 0:
        return {'polarity': 0.0, 'subjectivity': 0.0}
    
    try:
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    except:
        return {'polarity': 0.0, 'subjectivity': 0.0}


def extract_pronoun_features(text):
    """
    Extract first-person pronoun usage.
    
    Args:
        text: Input text
        
    Returns:
        dict: Pronoun features
    """
    if not isinstance(text, str) or len(text) == 0:
        return {'fp_pronoun_rate': 0.0, 'fp_pronoun_count': 0}
    
    # Tokenize by whitespace and punctuation
    tokens = re.findall(r'\b\w+\b', text.lower())
    
    if len(tokens) == 0:
        return {'fp_pronoun_rate': 0.0, 'fp_pronoun_count': 0}
    
    # Count first-person pronouns
    fp_count = sum(1 for token in tokens if token in FP_PRONOUNS)
    fp_rate = fp_count / len(tokens)
    
    return {
        'fp_pronoun_rate': fp_rate,
        'fp_pronoun_count': fp_count
    }


def extract_style_features(text):
    """
    Extract stylistic features.
    
    Args:
        text: Input text
        
    Returns:
        dict: Style features
    """
    if not isinstance(text, str):
        text = ""
    
    # Character and word counts
    char_count = len(text)
    word_count = len(text.split())
    
    # Average word length
    words = text.split()
    avg_word_len = np.mean([len(w) for w in words]) if words else 0.0
    
    # Punctuation density
    punct_count = len(re.findall(r'[.!?]', text))
    punct_density = punct_count / char_count if char_count > 0 else 0.0
    
    # Emoji count
    emoji_count = emoji.emoji_count(text)
    
    return {
        'char_count': char_count,
        'word_count': word_count,
        'avg_word_length': avg_word_len,
        'punct_density': punct_density,
        'emoji_count': emoji_count
    }


def extract_all_features(text):
    """
    Extract all linguistic features from text.
    
    Args:
        text: Input text
        
    Returns:
        dict: All features combined
    """
    features = {}
    
    # Sentiment
    vader_scores = extract_vader_sentiment(text)
    features.update({f'vader_{k}': v for k, v in vader_scores.items()})
    
    textblob_scores = extract_textblob_sentiment(text)
    features.update({f'textblob_{k}': v for k, v in textblob_scores.items()})
    
    # Pronouns
    pronoun_feats = extract_pronoun_features(text)
    features.update(pronoun_feats)
    
    # Style
    style_feats = extract_style_features(text)
    features.update(style_feats)
    
    return features


def extract_features_batch(texts, show_progress=True):
    """
    Extract features for a batch of texts.
    
    Args:
        texts: List or Series of texts
        show_progress: Show progress bar
        
    Returns:
        pd.DataFrame: Features DataFrame
    """
    iterator = tqdm(texts, desc="Extracting features") if show_progress else texts
    
    features_list = []
    for text in iterator:
        features = extract_all_features(text)
        features_list.append(features)
    
    return pd.DataFrame(features_list)


def get_feature_names():
    """
    Get list of all feature names.
    
    Returns:
        list: Feature names
    """
    return [
        'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound',
        'textblob_polarity', 'textblob_subjectivity',
        'fp_pronoun_rate', 'fp_pronoun_count',
        'char_count', 'word_count', 'avg_word_length',
        'punct_density', 'emoji_count'
    ]


def aggregate_user_features(features_df, posts_df):
    """
    Aggregate post-level features to user level.
    
    Args:
        features_df: DataFrame of post-level features
        posts_df: DataFrame with author and label info
        
    Returns:
        tuple: (user_features_df, user_labels, user_splits)
    """
    # Combine features with post metadata
    combined = pd.concat([posts_df[['author', 'label', 'split']].reset_index(drop=True), 
                          features_df.reset_index(drop=True)], axis=1)
    
    # Get feature columns
    feature_cols = get_feature_names()
    
    # Aggregate by mean
    user_features = combined.groupby('author')[feature_cols].mean().reset_index()
    
    # Get user labels and splits
    user_labels = combined.groupby('author')['label'].first().values
    user_splits = combined.groupby('author')['split'].first().values
    
    return user_features.drop('author', axis=1), user_labels, user_splits