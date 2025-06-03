import os
import logging
import uvicorn
from fastapi import FastAPI

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Import configurations and modules ---
# Ensure config is imported first if other modules depend on its loaded values at import time
from . import config # To ensure S3_CLIENT and other configs are initialized
from .database import check_db_initialized, SQLITE_DB_PATH # SQLITE_DB_PATH for health check
from .routers import text_match, image_match
from .services import ocr_service # To ensure EASYOCR_API_READER is initialized

if not config.API_KEY:
    logger.warning(
        "API_KEY environment variable not set. API authentication will fail if this is not a test environment with a mock key."
    )
if not config.S3_CLIENT:
    logger.warning("S3_CLIENT not initialized. S3-dependent operations might fail.")
if not ocr_service.EASYOCR_API_READER:
    logger.warning("EASYOCR_API_READER not initialized in ocr_service. Text extraction for image matching might fail.")

# --- FastAPI App Initialization ---
app = FastAPI(title="S3 Image OCR Service", version="1.0.0")

# Call check on startup (optional, for early warning)
check_db_initialized()

# --- Include Routers ---
app.include_router(text_match.router)
app.include_router(image_match.router)

@app.get("/", summary="Service Health Check", tags=['Default'])
def root():
    """Provides a simple health check message."""
    db_status = "not found"
    if os.path.exists(SQLITE_DB_PATH):
        db_status = "found"
    return {"message": "Image OCR Service is running.", "database_status": f"SQLite DB at {SQLITE_DB_PATH} {db_status}"}

if __name__ == "__main__":
    logger.info("Starting Uvicorn server for Image OCR Service...")
    uvicorn.run(
        "main:app", # Use string "module:app" for reload
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=True if os.getenv("AUTORELOAD", "false").lower() == "true" else False
    )
