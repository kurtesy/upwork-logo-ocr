import difflib
import logging
import httpx
from fastapi import Request

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

def get_client_location(request: Request) -> str:
    """
    Determines client location based on IP address using an external service.
    Returns 'internal' for private IPs and 'unknown' on failure. This is a
    synchronous function and will block the event loop.
    """
    client_ip = request.client.host
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