"""
Text preprocessing utilities.
"""

import re
import ftfy
import spacy
from tqdm import tqdm


# Load spacy model (lazy loading)
_nlp = None

def get_nlp():
    """Lazy load spacy model."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        except:
            print("Warning: spacy model not found. Run: python -m spacy download en_core_web_sm")
            _nlp = None
    return _nlp


def clean_text(text, lowercase=True, remove_urls=True, fix_unicode=True):
    """
    Basic text cleaning.
    
    Args:
        text: Input text
        lowercase: Convert to lowercase
        remove_urls: Remove URLs
        fix_unicode: Fix unicode encoding issues
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Fix unicode encoding
    if fix_unicode:
        text = ftfy.fix_text(text)
    
    # Remove URLs
    if remove_urls:
        text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove Reddit markdown
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # [text](url)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Lowercase
    if lowercase:
        text = text.lower()
    
    return text


def lemmatize_text(text):
    """
    Lemmatize text using spaCy.
    
    Args:
        text: Input text
        
    Returns:
        str: Lemmatized text
    """
    nlp = get_nlp()
    if nlp is None:
        return text
    
    doc = nlp(text)
    lemmatized = ' '.join([token.lemma_ for token in doc if not token.is_space])
    
    return lemmatized


def preprocess_corpus(texts, use_lemmatization=False, show_progress=True):
    """
    Preprocess a corpus of texts.
    
    Args:
        texts: List or Series of texts
        use_lemmatization: Whether to lemmatize
        show_progress: Show progress bar
        
    Returns:
        list: Preprocessed texts
    """
    iterator = tqdm(texts, desc="Preprocessing") if show_progress else texts
    
    cleaned = []
    for text in iterator:
        text = clean_text(text)
        if use_lemmatization and len(text) > 0:
            text = lemmatize_text(text)
        cleaned.append(text)
    
    return cleaned


def remove_stopwords(text, custom_stopwords=None):
    """
    Remove stopwords using spaCy.
    
    Args:
        text: Input text
        custom_stopwords: Optional set of additional stopwords
        
    Returns:
        str: Text with stopwords removed
    """
    nlp = get_nlp()
    if nlp is None:
        return text
    
    doc = nlp(text)
    
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue
        if custom_stopwords and token.text.lower() in custom_stopwords:
            continue
        filtered_tokens.append(token.text)
    
    return ' '.join(filtered_tokens)


def normalize_text_for_features(text):
    """
    Normalize text specifically for feature extraction.
    Keeps more information than standard cleaning.
    
    Args:
        text: Input text
        
    Returns:
        str: Normalized text
    """
    if not isinstance(text, str):
        return ""
    
    # Fix unicode
    text = ftfy.fix_text(text)
    
    # Remove URLs but keep placeholder
    text = re.sub(r'http\S+|www\.\S+', ' URL ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text