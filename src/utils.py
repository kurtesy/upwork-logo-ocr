import difflib
import logging
import httpx
from fastapi import Request
from thefuzz import fuzz
import jellyfish


logger = logging.getLogger(__name__)

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculates the similarity ratio between two strings."""
    if not text1 and not text2: # Both empty
        return 1.0
    if not text1 or not text2: # One empty
        return 0.0
    if text1.lower() in text2.lower():
        return 1.0
    return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def new_calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculates a composite similarity score between two strings using multiple algorithms.

    This function combines:
    1. A quick check for exact substring matches.
    2. Levenshtein distance (via thefuzz library) for edit-distance similarity.
    3. SequenceMatcher (Gestalt Pattern Matching) for block-based similarity.
    4. Soundex phonetic matching as a booster for similar-sounding words.

    The final score is a weighted average to provide a more robust similarity
    measure for noisy text data like OCR results.
    """
    if not text1 and not text2:  # Both empty
        return 1.0
    if not text1 or not text2:  # One empty
        return 0.0

    text1_lower = text1.lower()
    text2_lower = text2.lower()

    # 1. Quick check for exact substring match (strongest signal)
    if text1_lower in text2_lower or text2_lower in text1_lower:
        return 1.0

    # 2. Levenshtein-based similarity (good for typos)
    # fuzz.ratio returns an int 0-100
    lev_score = fuzz.ratio(text1_lower, text2_lower) / 100.0

    # 3. SequenceMatcher similarity (good for finding common blocks)
    sm_score = difflib.SequenceMatcher(None, text1_lower, text2_lower).ratio()

    # 4. Soundex phonetic similarity (good for similar-sounding names)
    # We compare the soundex of the words without spaces to handle multi-word text.
    text1_no_space = "".join(text1_lower.split())
    text2_no_space = "".join(text2_lower.split())
    soundex_score = 0.0
    if text1_no_space and text2_no_space:
        try:
            soundex1 = jellyfish.soundex(text1_no_space)
            soundex2 = jellyfish.soundex(text2_no_space)
            if soundex1 == soundex2:
                soundex_score = 1.0
        except Exception:
            # jellyfish can sometimes fail on weird characters, so we ignore errors.
            pass

    # 5. Combine scores
    # We take the maximum of Levenshtein and SequenceMatcher to be robust against
    # different types of string variations (e.g., typos vs. word order/spacing).
    primary_score = max(lev_score, sm_score)

    # If the strings are phonetically similar, give a small boost.
    # The boost is proportional to the primary score, pushing it closer to 1.0.
    final_score = primary_score
    if soundex_score > 0:
        final_score += (1.0 - primary_score) * 0.25  # Boost by 25% of the remaining gap to 1.0

    return min(final_score, 1.0)

def get_client_location(request: Request) -> str:
    """
    Determines client location based on IP address using an external service.
    Returns 'internal' for private IPs and 'unknown' on failure. This is a
    synchronous function and will block the event loop.
    """
    client_ip = request.client.host # type: ignore
    if not client_ip or client_ip == "127.0.0.1" or client_ip.startswith("192.168.") or client_ip.startswith("10."):
        return "internal"

    try:
        with httpx.Client(timeout=1.0) as client:
            # Using a free, no-key-required geolocation API
            response = client.get(f"http://ip-api.com/json/{client_ip}?fields=status,message,countryCode")
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "success" and data.get("countryCode"):
                return data["countryCode"]
            else:
                logger.warning(f"Geolocation lookup for {client_ip} failed: {data.get('message')}")
    except httpx.RequestError as e:
        logger.warning(f"Geolocation lookup for {client_ip} failed with network error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during geolocation lookup for {client_ip}: {e}", exc_info=True)

    return "unknown"