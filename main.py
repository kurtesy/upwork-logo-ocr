import os
import logging
import uvicorn
import time
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# --- Import configurations and modules ---
from src import config # To ensure S3_CLIENT and other configs are initialized
from src.database import check_db_initialized, SQLITE_DB_PATH
from src.rate_limiter import RateLimiter # type: ignore
from routers import text_match, image_match, admin

# --- Logging Setup ---
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname).1s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
# Create a RateLimiter instance: 100 requests per 60 seconds per client IP.
limiter = RateLimiter(max_requests=100, time_window=60)

# --- Lifespan Functions ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    logger.info("Application startup...")
    # Initialize services that require loading models or connecting to resources
    image_match.initialize_image_similarity_service()
    try:
        from services import ocr_service
        if not ocr_service.EASYOCR_API_READER:
            logger.warning("EASYOCR_API_READER not initialized in ocr_service. Text extraction for image matching might fail.")
    except ImportError:
        logger.error("Could not import 'services.ocr_service'. Text extraction for image matching will likely fail.")
    # Check for database
    check_db_initialized()
    # Check for necessary configs
    if not config.API_KEY:
        logger.warning("API_KEY environment variable not set. API authentication will fail if this is not a test environment with a mock key.")
    if not config.S3_CLIENT:
        logger.warning("S3_CLIENT not initialized. S3-dependent operations might fail.")
    logger.info("Application startup complete.")
    yield
    # Code to run on shutdown
    logger.info("Application shutdown.")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="TM Image OCR Service",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def rate_limit_and_log_requests(request: Request, call_next):
    """
    Middleware to apply rate limiting and then log API requests.
    """
    try:
        await asyncio.to_thread(limiter, request)
    except HTTPException as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={'detail': exc.detail}
        )

    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(
        f'"{request.method} {request.url.path}" {response.status_code} {process_time:.4f}s - {request.client.host}') # type: ignore
    return response

# --- Include Routers ---
app.include_router(text_match.router)
app.include_router(image_match.router)
app.include_router(admin.router)

@app.get("/", summary="Service Health Check", tags=["Default"])
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
