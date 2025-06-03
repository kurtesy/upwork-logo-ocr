import logging
import easyocr

logger = logging.getLogger(__name__)

EASYOCR_API_READER = None
try:
    EASYOCR_API_READER = easyocr.Reader(['en'], gpu=False, verbose=False)
    logger.info("EasyOCR reader initialized for API usage in ocr_service.")
except ImportError:
    logger.warning("EasyOCR library not found. Text extraction for image matching will be limited/unavailable.")
except Exception as e:
    logger.error(f"Failed to initialize EasyOCR reader for API in ocr_service: {e}", exc_info=True)

def extract_text_from_image_bytes_api(image_bytes: bytes) -> str:
    """Extracts text from image bytes using the API's EasyOCR reader."""
    if not EASYOCR_API_READER:
        logger.error("EasyOCR API Reader not available for text extraction.")
        return ""
    try:
        result = EASYOCR_API_READER.readtext(image_bytes)
        text = " ".join([item[1] for item in result]) # type: ignore
        logger.info(f"API Text Extraction (EasyOCR): Extracted text: '{text.strip()}'")
        return text.strip()
    except Exception as e:
        logger.error(f"Error during EasyOCR text extraction in API: {e}", exc_info=True)
        return ""