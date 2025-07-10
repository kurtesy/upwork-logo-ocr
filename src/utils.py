import difflib

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculates the similarity ratio between two strings."""
    if not text1 and not text2: # Both empty
        return 1.0
    if not text1 or not text2: # One empty
        return 0.0
    if text1.lower() in text2.lower():
        return 1.0
    return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()