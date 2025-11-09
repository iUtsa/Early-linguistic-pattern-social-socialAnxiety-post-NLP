"""
Keyword lists for anxiety detection and ablation studies.
"""

import re

# Anxiety-related terms for pseudo-onset detection
ANXIETY_TERMS = {
    'panic', 'panicking', 'panicked', 'panic attack', 'panic attacks',
    'anxious', 'anxiety', 'anxieties', 'anxiously',
    'worry', 'worried', 'worrying', 'worries', 'worrisome',
    'nervous', 'nervousness', 'nervously',
    'fear', 'afraid', 'scared', 'frightened', 'terrified',
    'stress', 'stressed', 'stressful', 'stressing',
    'overwhelm', 'overwhelmed', 'overwhelming',
    'dread', 'dreading', 'dreaded',
    'overthink', 'overthinking', 'overthought',
    'racing thoughts', 'racing heart', 'heart racing', 'heart pounding',
    'can\'t breathe', 'cannot breathe', 'difficulty breathing',
    'shaking', 'trembling', 'shaky', 'tremors',
    'sweating', 'sweaty', 'cold sweats',
    'nausea', 'nauseous', 'sick to my stomach',
    'dizzy', 'dizziness', 'lightheaded',
    'insomnia', 'can\'t sleep', 'cannot sleep', 'trouble sleeping',
    'restless', 'restlessness', 'on edge',
    'tension', 'tense', 'uptight',
    'hyperventilate', 'hyperventilating',
    'catastrophize', 'catastrophizing',
    'what if', 'worst case',
}

# Terms that might leak label info (for ablation experiments)
LEAK_TERMS = {
    'anxiety', 'anxious', 'panic', 'diagnosed', 'disorder',
    'therapy', 'therapist', 'psychiatrist', 'medication',
    'ssri', 'benzodiazepine', 'xanax', 'zoloft', 'lexapro',
    'mental health', 'mental illness',
}


def is_anxiety_term(text):
    """
    Check if text contains any anxiety-related terms.
    
    Args:
        text (str): Text to check
        
    Returns:
        bool: True if any anxiety term is found
    """
    if not isinstance(text, str):
        return False
    
    text_lower = text.lower()
    
    # Check for each term (using word boundaries for accuracy)
    for term in ANXIETY_TERMS:
        # Escape special regex characters
        escaped_term = re.escape(term)
        # Use word boundaries for single words, but allow phrase matches
        if ' ' in term:
            # Multi-word phrase
            pattern = r'\b' + escaped_term.replace(r'\ ', r'\s+') + r'\b'
        else:
            # Single word with word boundaries
            pattern = r'\b' + escaped_term + r'\b'
        
        if re.search(pattern, text_lower):
            return True
    
    return False


def contains_leak_terms(text):
    """
    Check if text contains label-leaking terms (for ablation).
    
    Args:
        text (str): Text to check
        
    Returns:
        bool: True if any leak term is found
    """
    if not isinstance(text, str):
        return False
    
    text_lower = text.lower()
    
    for term in LEAK_TERMS:
        escaped_term = re.escape(term)
        if ' ' in term:
            pattern = r'\b' + escaped_term.replace(r'\ ', r'\s+') + r'\b'
        else:
            pattern = r'\b' + escaped_term + r'\b'
        
        if re.search(pattern, text_lower):
            return True
    
    return False


def get_anxiety_keywords():
    """Return the set of anxiety terms."""
    return ANXIETY_TERMS.copy()


def get_leak_keywords():
    """Return the set of leak terms."""
    return LEAK_TERMS.copy()


if __name__ == '__main__':
    # Test cases
    test_texts = [
        "I'm feeling really anxious today",
        "Having a panic attack right now",
        "Just watched a great movie",
        "My heart is racing and I can't breathe",
        "What if something goes wrong?",
    ]
    
    print("Testing anxiety term detection:")
    for text in test_texts:
        result = is_anxiety_term(text)
        print(f"  '{text[:40]}...' -> {result}")
    
    print("\nTesting leak term detection:")
    leak_tests = [
        "I have anxiety disorder",
        "My therapist recommended this",
        "Just a normal day at work",
    ]
    for text in leak_tests:
        result = contains_leak_terms(text)
        print(f"  '{text}' -> {result}")