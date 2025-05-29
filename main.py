import os
import io
import logging
from typing import List
from typing import List, Tuple # Added Tuple for helper return types
import sqlite3
import difflib

from fastapi import FastAPI, HTTPException, Query, File, UploadFile, Body
from pydantic import BaseModel 
from dotenv import load_dotenv
import uvicorn
import boto3
from botocore.exceptions import ClientError
import numpy as np
import cv2

# Assuming utils.py is in the same directory or a discoverable path
try:
    from .utils import calculate_image_similarity
except ImportError:
    # Fallback for direct execution if utils.py is in the same directory
    # and the script is run directly, not as part of a package.
    try:
        from utils import calculate_image_similarity
    except ImportError:
        calculate_image_similarity = None
        logging.getLogger(__name__).error("Could not import calculate_image_similarity from utils. Image similarity endpoint will not work.")


# Load environment variables from .env file (if it exists)
load_dotenv()

# --- Configuration ---
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")

# Bucket and prefix for the grayscale images to compare against in S3
GRAYSCALE_BUCKET_NAME = os.getenv("GRAYSCALE_BUCKET_NAME", "newbucket-trademark") # Default to your existing bucket
GRAYSCALE_S3_PREFIX = os.getenv("GRAYSCALE_S3_PREFIX", "images/grayscale/") # Default to your existing grayscale prefix

SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "ocr_results.db") # Same as in the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"SQLite DB Path: {SQLITE_DB_PATH}")
logger.info(f"Target S3 Grayscale Bucket: {GRAYSCALE_BUCKET_NAME}, Prefix: {GRAYSCALE_S3_PREFIX}")

# --- S3 Client Initialization ---
S3_CLIENT = None
if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and AWS_DEFAULT_REGION:
    try:
        S3_CLIENT = boto3.client(
            's3', aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=AWS_DEFAULT_REGION)
        logger.info("S3 client initialized successfully.")
    except Exception as e:
        logger.error("Failed to initialize S3 client.", exc_info=True)
else:
    logger.warning("AWS credentials or region not fully configured. S3 operations might fail for S3-dependent endpoints.")

# --- FastAPI App Initialization ---
app = FastAPI(title="S3 Image OCR Service", version="1.0.0")

# --- Text Similarity Helper ---
def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculates the similarity ratio between two strings."""
    if not text1 and not text2: # Both empty
        return 1.0
    if not text1 or not text2: # One empty
        return 0.0
    return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

# --- SQLite DB Helper ---
def get_db_connection():
    """Establishes a connection to the SQLite database."""
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False) # check_same_thread=False for FastAPI
        conn.row_factory = sqlite3.Row # Access columns by name
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error connecting to SQLite database {SQLITE_DB_PATH}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not connect to the database: {e}")

def check_db_initialized():
    if not os.path.exists(SQLITE_DB_PATH):
        logger.warning(f"SQLite database file not found at {SQLITE_DB_PATH}. Please run the OCR processing script to create and populate it.")

# Call check on startup (optional, for early warning)
check_db_initialized()

# --- API Endpoints ---
class LogoMatchResponse(BaseModel):
    query_text: str
    similarity_threshold: float
    matching_logos: List[str]
    processed_ocr_files: int
    errors: List[str] = []

class ImageSimilarityInfo(BaseModel):
    s3_image_key: str
    color_similarity: float
    shape_similarity: float
    combined_similarity: float

class FindSimilarImagesResponse(BaseModel):
    uploaded_filename: str
    similar_images: List[ImageSimilarityInfo]
    errors: List[str] = []

class BulkLogoMatchQuery(BaseModel):
    query_text: str

class BulkLogoMatchRequest(BaseModel):
    queries: List[BulkLogoMatchQuery] = Body(..., max_items=100)
    similarity_threshold: float = 0.7

class BulkLogoMatchResult(BaseModel):
    query_text: str
    matching_logos: List[str]
    processed_ocr_files: int # This might be the total in DB, or specific to this query's context
    errors: List[str] = []

class BulkLogoMatchResponse(BaseModel):
    results: List[BulkLogoMatchResult]

@app.get("/", summary="Service Health Check", tags=['Default'])
def root():
    """Provides a simple health check message."""
    db_status = "not found"
    if os.path.exists(SQLITE_DB_PATH):
        db_status = "found"
    return {"message": "Image OCR Service is running.", "database_status": f"SQLite DB at {SQLITE_DB_PATH} {db_status}"}

def _get_ocr_text_matches(
    db_cursor: sqlite3.Cursor,
    query_text: str,
    similarity_threshold: float
) -> Tuple[List[str], int, List[str]]:
    """
    Helper function to find OCR text matches for a single query.
    Returns: (matching_logos, processed_ocr_files_count, errors)
    """
    logger.debug(f"Helper: Finding matches for query: '{query_text}', threshold: {similarity_threshold}")
    matching_logos: List[str] = []
    errors: List[str] = []
    processed_ocr_files_count = 0
    
    conn = None
    try:
        conn = get_db_connection()
        # Use the passed cursor if available, otherwise create one (for single endpoint)
        cursor = db_cursor if db_cursor else conn.cursor()

        # Get total count of OCR entries
        cursor.execute("SELECT COUNT(*) FROM ocr_results")
        count_result = cursor.fetchone()
        if count_result:
            processed_ocr_files_count = count_result[0]
        # Fetch all OCR results
        cursor.execute("SELECT image_identifier, extracted_text FROM ocr_results") # This could be slow for large DBs
        ocr_entries = cursor.fetchall()

        for entry in ocr_entries:
            image_identifier = entry["image_identifier"]
            ocr_content = entry["extracted_text"]

            if ocr_content is None: # Handle cases where text might be NULL
                ocr_content = ""

            similarity = calculate_text_similarity(query_text, ocr_content)

            if similarity >= similarity_threshold: # type: ignore
                matching_logos.append(image_identifier)
                logger.debug(f"Match found: {image_identifier} (Similarity: {similarity:.2f})")

    except sqlite3.Error as e:
        err_msg = f"SQLite database error: {e}"
        logger.error(err_msg, exc_info=True)
        errors.append(err_msg)
    except Exception as e: # Catch any other unexpected errors
        err_msg = f"An unexpected error occurred while querying the database: {e}"
        logger.error(err_msg, exc_info=True)
        errors.append(err_msg)
    finally:
        # Only close connection if it was created within this helper
        if conn and not db_cursor:
            conn.close()
    
    if not ocr_entries and not errors: # type: ignore # ocr_entries might not be defined if exception before assignment
        logger.info(f"No OCR entries found in the database {SQLITE_DB_PATH}.")
    
    return matching_logos, processed_ocr_files_count, errors

@app.get("/ocr/logo-match", response_model=LogoMatchResponse,
         summary="Find logos by OCR text similarity",
         tags=["OCR Match"])
async def find_matching_logos(
    query_text: str = Query(..., min_length=1, description="The text to search for in OCR results."),
    similarity_threshold: float = Query(0.7, ge=0.0, le=1.0, description="Minimum similarity ratio (0.0 to 1.0) to consider a match.")
):
    """
    Searches through processed OCR results to find logos whose extracted text
    matches the provided query_text with at least the given similarity threshold.
    """
    logger.info(f"Finding matching logos for query: '{query_text}' with threshold: {similarity_threshold} from SQLite DB.")
    
    # For single query, we don't pass a cursor, _get_ocr_text_matches will create its own connection
    matching_logos, processed_ocr_files_count, errors = _get_ocr_text_matches(
        db_cursor=None, # type: ignore
        query_text=query_text,
        similarity_threshold=similarity_threshold
    )
        # errors.append("No OCR data found in the database. Please run the processing script.")

    return LogoMatchResponse(
        query_text=query_text,
        similarity_threshold=similarity_threshold,
        matching_logos=matching_logos,
        processed_ocr_files=processed_ocr_files_count,
        errors=errors
    )

def _process_single_image_similarity(
    uploaded_image_np: np.ndarray,
    uploaded_filename: str,
    similarity_threshold: float
) -> Tuple[List[ImageSimilarityInfo], List[str]]:
    """
    Helper function to find similar images in S3 for a single uploaded image.
    Returns: (similar_images_found, errors)
    """
    logger.debug(f"Helper: Processing image similarity for {uploaded_filename}, threshold: {similarity_threshold}")
    if not calculate_image_similarity:
        raise HTTPException(status_code=501, detail="Image similarity utility not available.")

    errors: List[str] = []
    similar_images_found: List[ImageSimilarityInfo] = []

    try:
        paginator = S3_CLIENT.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=GRAYSCALE_BUCKET_NAME, Prefix=GRAYSCALE_S3_PREFIX)

        for page in page_iterator:
            if "Contents" not in page:
                continue
            for s3_object_summary in page["Contents"]:
                s3_key = s3_object_summary["Key"]
                if s3_key.endswith('/') or not any(s3_key.lower().endswith(ext) for ext in ['.jpeg', '.jpg', '.png']):
                    logger.debug(f"Skipping non-image or directory in S3: {s3_key}")
                    continue

                try:
                    logger.debug(f"Comparing {uploaded_filename} with S3 image: {s3_key}")
                    s3_response = S3_CLIENT.get_object(Bucket=GRAYSCALE_BUCKET_NAME, Key=s3_key)
                    s3_image_bytes = s3_response['Body'].read()
                    s3_image_np = cv2.imdecode(np.frombuffer(s3_image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)

                    if s3_image_np is None or s3_image_np.size == 0:
                        logger.warning(f"Could not decode S3 image: {s3_key}. Skipping.")
                        errors.append(f"Could not decode S3 image: {s3_key}")
                        continue

                    similarity_scores = calculate_image_similarity(uploaded_image_np, s3_image_np)

                    if similarity_scores and similarity_scores["combined_similarity"] >= similarity_threshold:
                        similar_images_found.append(ImageSimilarityInfo(s3_image_key=s3_key, **similarity_scores))
                        logger.debug(f"Found similar image for {uploaded_filename}: {s3_key} with combined similarity: {similarity_scores['combined_similarity']:.4f}")

                except ClientError as e:
                    err_msg = f"S3 ClientError processing {s3_key}: {e.response.get('Error', {}).get('Message', str(e))}"
                    logger.error(err_msg)
                    errors.append(err_msg)
                except Exception as e:
                    err_msg = f"Unexpected error processing S3 image {s3_key}: {str(e)}"
                    logger.error(err_msg, exc_info=True)
                    errors.append(err_msg)

    except ClientError as e:
        err_msg = f"S3 ClientError listing objects from s3://{GRAYSCALE_BUCKET_NAME}/{GRAYSCALE_S3_PREFIX}: {e.response.get('Error', {}).get('Message', str(e))}"
        logger.error(err_msg)
        errors.append(err_msg) # Add to response errors, could also raise HTTPException

    return similar_images_found, errors

@app.post("/ocr/image-match", response_model=FindSimilarImagesResponse,
          summary="Compare uploaded image with S3 grayscale images",
          tags=["OCR Match"])
async def find_similar_images_in_s3(
    uploaded_file: UploadFile = File(..., description="Grayscale image (JPEG or PNG) to compare."),
    similarity_threshold: float = Query(0.6, ge=0.0, le=1.0, description="Minimum combined similarity score to consider a match.")
):
    """
    Upload a grayscale image and compare it against a collection of grayscale images
    stored in an S3 bucket. Returns a list of S3 images that are similar based on
    color and shape analysis.
    """
    if not S3_CLIENT:
        raise HTTPException(status_code=503, detail="S3 client not available. Check AWS configuration.")

    try:
        contents = await uploaded_file.read()
        uploaded_image_np = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_UNCHANGED)
        if uploaded_image_np is None or uploaded_image_np.size == 0:
            raise HTTPException(status_code=400, detail="Invalid or unsupported image format for uploaded file.")
        logger.info(f"Successfully decoded uploaded image: {uploaded_file.filename}, shape: {uploaded_image_np.shape}")

    except Exception as e:
        logger.error(f"Error processing uploaded file {uploaded_file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Could not process uploaded image: {str(e)}")

    similar_images, errors = _process_single_image_similarity(
        uploaded_image_np=uploaded_image_np,
        uploaded_filename=uploaded_file.filename or "N/A",
        similarity_threshold=similarity_threshold
    )
    return FindSimilarImagesResponse(uploaded_filename=uploaded_file.filename or "N/A", similar_images=similar_images, errors=errors)

@app.post("/ocr/bulk-logo-match", response_model=BulkLogoMatchResponse,
          summary="Bulk find logos by OCR text similarity (max 100 queries)",
          tags=["OCR Match"])
async def bulk_find_matching_logos(request_data: BulkLogoMatchRequest):
    """
    Processes a list of up to 100 text queries to find matching logos based on OCR results.
    """
    logger.info(f"Bulk logo match request received for {len(request_data.queries)} queries.")
    results: List[BulkLogoMatchResult] = []
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        for query_item in request_data.queries:
            matching_logos, processed_files, errors = _get_ocr_text_matches(
                db_cursor=cursor,
                query_text=query_item.query_text,
                similarity_threshold=request_data.similarity_threshold
            )
            results.append(BulkLogoMatchResult(
                query_text=query_item.query_text,
                matching_logos=matching_logos,
                processed_ocr_files=processed_files, # This will be the total DB count for each query
                errors=errors
            ))
    finally:
        if conn:
            conn.close()
    return BulkLogoMatchResponse(results=results)

@app.post("/ocr/bulk-image-match",
          summary="Bulk compare uploaded images with S3 (max 100 images)",
          tags=["OCR Match"]) # Response model can be a list of FindSimilarImagesResponse
async def bulk_find_similar_images_in_s3(
    uploaded_files: List[UploadFile] = File(..., description="List of up to 100 grayscale images (JPEG or PNG) to compare."),
    similarity_threshold: float = Query(0.6, ge=0.0, le=1.0, description="Minimum combined similarity score to consider a match.")
):
    if len(uploaded_files) > 100:
        raise HTTPException(status_code=413, detail="Too many files. Maximum 100 images allowed per request.")
    if not S3_CLIENT:
        raise HTTPException(status_code=503, detail="S3 client not available. Check AWS configuration.")

    results = []
    for uploaded_file in uploaded_files:
        # Essentially call the single image match logic for each file
        # This reuses the logic including error handling for individual file processing
        single_result = await find_similar_images_in_s3(uploaded_file, similarity_threshold)
        results.append(single_result) # single_result is already a FindSimilarImagesResponse
    return results # Returns a list of FindSimilarImagesResponse objects

if __name__ == "__main__":
    # This block will only execute when the script is run directly (e.g., python main.py)
    # It will not run when main.py is imported as a module by another script (like your trigger_ocr.py).
    logger.info("Starting Uvicorn server for Image OCR Service...")
    uvicorn.run(
        "main:app", # type: ignore # Use string "module:app" for reload
        host=os.getenv("HOST", "0.0.0.0"), 
        port=int(os.getenv("PORT", 8000)),
        reload=True # Enable auto-reload for development
    )
