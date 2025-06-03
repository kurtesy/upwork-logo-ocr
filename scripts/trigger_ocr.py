import os
import io
import logging
import sys
import json
import sqlite3
from datetime import datetime
import argparse # For command-line arguments
from typing import Dict, Any # For type hinting

import boto3
from botocore.exceptions import ClientError
from PIL import Image, ImageOps
import pytesseract
from dotenv import load_dotenv
import cv2 # For image decoding for RapidOCR
import numpy as np # For image decoding for RapidOCR

# --- OCR Library Imports (add as needed) ---
try:
    import easyocr
    EASYOCR_AVAILABLE = True
    # Consider initializing the reader later or making it configurable
    EASYOCR_READER = easyocr.Reader(['en', 'hi'], gpu=True, quantize=True)
except ImportError:
    EASYOCR_AVAILABLE = False
    easyocr = None # type: ignore
    EASYOCR_READER = None

try:
    from rapidocr_onnxruntime import RapidOCR
    RAPIDOCR_AVAILABLE = True
    RAPIDOCR_ENGINE = RapidOCR()
except ImportError:
    RAPIDOCR_AVAILABLE = False
    RapidOCR = None # type: ignore
    RAPIDOCR_ENGINE = None

# --- Load environment variables ---
load_dotenv()

# --- Configuration (moved from main.py) ---
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")

SOURCE_BUCKET_NAME = os.getenv("SOURCE_BUCKET_NAME", "newbucket-trademark")
SOURCE_PREFIX = os.getenv("SOURCE_PREFIX", "images/original/")
DESTINATION_BUCKET_NAME = os.getenv("DESTINATION_BUCKET_NAME", "newbucket-trademark")
GRAYSCALE_DESTINATION_PREFIX = os.getenv("GRAYSCALE_DESTINATION_PREFIX", "images/grayscale/")
VECTOR_DESTINATION_PREFIX = os.getenv("VECTOR_DESTINATION_PREFIX", "images/vectors/")

PROCESSING_MODE = os.getenv("PROCESSING_MODE", "S3").upper()
LOCAL_IMAGE_SOURCE_DIR = os.getenv("LOCAL_IMAGE_SOURCE_DIR", "./local_images/")
LOCAL_GRAYSCALE_DESTINATION_DIR = os.getenv("LOCAL_GRAYSCALE_DESTINATION_DIR", "./local_images_grayscale/")
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "ocr_results.db")

TESSERACT_CMD = os.getenv("TESSERACT_CMD")
OCR_ENGINE_PREFERENCE = os.getenv("OCR_ENGINE_PREFERENCE", "RAPIDOCR").upper() # RAPIDOCR, EASYOCR, PYTESSERACT
if TESSERACT_CMD:
    pytesseract.tesseract_cmd = TESSERACT_CMD # type: ignore


# --- Logging Setup (moved from main.py) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Using __name__ for the logger

logger.info(f"--- OCR Script Configuration ---")
logger.info(f"Processing Mode: {PROCESSING_MODE}")
logger.info(f"SQLite DB Path: {SQLITE_DB_PATH}")

# --- S3 Client Initialization (moved from main.py) ---
S3_CLIENT = None
if PROCESSING_MODE == "S3":
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and AWS_DEFAULT_REGION:
        try:
            S3_CLIENT = boto3.client(
                's3',
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=AWS_DEFAULT_REGION
            )
            logger.info("S3 client initialized successfully for S3 mode.")
        except Exception:
            logger.error("Failed to initialize S3 client for S3 mode.", exc_info=True)
            # The script will exit later if S3_CLIENT is needed but not initialized
    else:
        logger.warning("AWS credentials or region not fully configured for S3 mode. S3 operations might fail.")
elif PROCESSING_MODE == "LOCAL":
    logger.info("Processing mode set to LOCAL. S3 client will not be initialized.")
    
# --- SQLite Database Setup ---
def init_sqlite_db():
    """Initializes the SQLite database and creates the ocr_results table if it doesn't exist."""
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ocr_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_identifier TEXT UNIQUE NOT NULL,
                extracted_text TEXT,
                source_type TEXT NOT NULL, -- 'S3' or 'LOCAL'
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Add index on image_identifier for faster lookups
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_ocr_results_extracted_text ON ocr_results (extracted_text);
        ''')
        # Create table for S3 image bytes cache
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS s3_image_cache (
                s3_key TEXT PRIMARY KEY NOT NULL,
                image_bytes BLOB NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        logger.info(f"SQLite database initialized successfully at {SQLITE_DB_PATH}")
    except sqlite3.Error as e:
        logger.error(f"Error initializing SQLite database: {e}", exc_info=True)
        sys.exit(1) # Exit if DB cannot be initialized
    finally:
        if conn:
            conn.close()
            
def convert_to_grayscale(image_bytes: bytes) -> bytes:
    """
    Converts an image from bytes to grayscale.

    Args:
        image_bytes: The image content as bytes.

    Returns:
        The grayscale image as bytes in JPEG format.
        Returns original bytes if conversion fails.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        grayscale_image = ImageOps.grayscale(image)
        
        # Save the grayscale image to a bytes buffer
        buffer = io.BytesIO()
        grayscale_image.save(buffer, format="JPEG")
        return buffer.getvalue()
    except Exception as e:
        logger.error(f"Error converting image to grayscale: {e}", exc_info=True)
        return image_bytes # Return original if error

# --- OCR Helper Function (EasyOCR) ---
def extract_text_with_easyocr(image_bytes: bytes) -> str:
    """
    Extracts text from image bytes using EasyOCR.
    Installation: pip install easyocr

    Args:
        image_bytes: The image content as bytes.

    Returns:
        The extracted text as a string. Returns an empty string if OCR fails or no text is found.
    """
    if not EASYOCR_AVAILABLE or easyocr is None:
        logger.error("EasyOCR library is not installed or failed to import.")
        return ""
    try:
        result = EASYOCR_READER.readtext(image_bytes) # type: ignore
        text = " ".join([item[1] for item in result]) # type: ignore
        return text
    except Exception as e:
        logger.error(f"Error during EasyOCR: {e}", exc_info=True)
        return ""

# --- OCR Helper Functions (moved and adapted from main.py) ---
def extract_text_with_pytesseract(image_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        logger.error(f"Error during Pytesseract OCR: {e}", exc_info=True)
        return ""

def extract_text_with_rapidocr(image_bytes: bytes) -> str:
    """
    Extracts text from image bytes using RapidOCR.
    Requires opencv-python for image decoding.
    """
    if not RAPIDOCR_AVAILABLE or RAPIDOCR_ENGINE is None:
        logger.error("RapidOCR library is not installed or engine failed to initialize.")
        return ""
    try:
        image_np_array = np.frombuffer(image_bytes, np.uint8)
        img_cv2 = cv2.imdecode(image_np_array, cv2.IMREAD_COLOR) # RapidOCR prefers color images

        if img_cv2 is None:
            logger.error("Failed to decode image for RapidOCR (cv2.imdecode returned None).")
            return ""

        result, _ = RAPIDOCR_ENGINE(img_cv2) # type: ignore
        if result:
            text = " ".join([res[1] for res in result])
            return text
        return ""
    except Exception as e:
        logger.error(f"Error during RapidOCR: {e}", exc_info=True)
        return ""

def extract_text_from_image(image_bytes: bytes, preferred_engine: str = OCR_ENGINE_PREFERENCE) -> str:
    """
    Extracts text from image bytes using the specified OCR engine, with fallbacks.
    """
    engines_order = {
        "RAPIDOCR": [extract_text_with_rapidocr, extract_text_with_easyocr, extract_text_with_pytesseract],
        "EASYOCR": [extract_text_with_easyocr, extract_text_with_rapidocr, extract_text_with_pytesseract],
        "PYTESSERACT": [extract_text_with_pytesseract, extract_text_with_easyocr, extract_text_with_rapidocr],
    }

    # Determine the order of attempts
    attempt_order = engines_order.get(preferred_engine, engines_order["EASYOCR"]) # Default to EasyOCR order

    for i, engine_func in enumerate(attempt_order):
        engine_name = engine_func.__name__.replace("extract_text_with_", "").upper()
        is_preferred = (i == 0 and engine_name == preferred_engine)
        
        # Skip unavailable engines unless it's the only one left or specifically Pytesseract (which doesn't have an explicit init check here)
        if engine_name == "RAPIDOCR" and not RAPIDOCR_AVAILABLE: continue
        if engine_name == "EASYOCR" and not EASYOCR_AVAILABLE: continue

        logger.info(f"Attempting OCR with {'preferred ' if is_preferred else ''}{engine_name}...")
        text = engine_func(image_bytes)
        if text.strip():
            logger.info(f"Successfully extracted text using {engine_name}, text: {text}.")
            return text
        logger.warning(f"{engine_name} did not extract text or returned empty.")

    logger.error("All configured OCR engines failed to extract text for an image.")
    return ""

# --- Processing Task Functions (moved and adapted from main.py) ---
def process_images_from_s3_task(
    source_bucket: str, source_prefix: str,
    start_file_number: int = 5000000  # Default starting number
) -> Dict[str, Any]:
    logger.info("Starting OCR process for S3 bucket: %s, prefix: %s", source_bucket, source_prefix)
    processed_files_count = 0
    failed_files_count = 0
    error_message = None

    if not S3_CLIENT: # Uses the S3_CLIENT defined in this script
        logger.error("S3 client not initialized. Cannot process images.")
        error_message = "S3 client not initialized."
        return {"processed_files": processed_files_count, "failed_files": failed_files_count, "error": error_message}
    
    conn = None
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
    except sqlite3.Error as e:
        logger.error(f"Failed to connect to SQLite DB: {e}")
        return {"processed_files": 0, "failed_files": 0, "error": f"Failed to connect to SQLite DB: {e}"}

    current_file_number = start_file_number  # Use the provided or default starting number
    # Safety break for the loop, e.g., if 'NoSuchKey' is never hit for some reason,
    # or if there are large gaps in the sequence.
    MAX_SEQUENTIAL_FILES_TO_CHECK = 2000000 # Example: Check up to 20,000 files
    files_checked_in_sequence = 0

    logger.info(f"Starting sequential S3 scan from file number {current_file_number} under prefix {source_prefix}")

    try:
        while files_checked_in_sequence < MAX_SEQUENTIAL_FILES_TO_CHECK:
            s3_filename = f"{current_file_number}.jpeg"
            # Construct the full object key using the source_prefix
            # Ensure source_prefix ends with a slash if it's a directory prefix and not empty
            if source_prefix and not source_prefix.endswith('/'):
                normalized_source_prefix = source_prefix + '/'
            else:
                normalized_source_prefix = source_prefix
            
            object_key = f"{normalized_source_prefix}{s3_filename}"
            
            try:
                # Check if the image has already been processed
                cursor.execute("SELECT 1 FROM ocr_results WHERE image_identifier = ?", (s3_filename,))
                if cursor.fetchone():
                    logger.info(f"Image {s3_filename} (object key: {object_key}) already processed. Skipping.")
                    # No need to increment processed_files_count here as it was already processed.
                    # The finally block will handle counter increments for files_checked_in_sequence.
                    current_file_number += 1
                    files_checked_in_sequence += 1
                    continue

                logger.info("Processing image: s3://%s/%s", source_bucket, object_key)
                response = S3_CLIENT.get_object(Bucket=source_bucket, Key=object_key) # type: ignore
                image_bytes = response['Body'].read()

                # Convert to grayscale
                grayscale_image_bytes = convert_to_grayscale(image_bytes)

                # Determine relative path for destination (this will be just s3_filename if prefix is used correctly)
                # The image_identifier should be the unique part of the name, e.g., "5865351.0.jpeg"
                image_identifier = s3_filename 
                base_filename, _ = os.path.splitext(s3_filename) # Get filename without extension "5865351.0"

                # Save grayscale image to S3
                grayscale_s3_key = f"{GRAYSCALE_DESTINATION_PREFIX.rstrip('/')}/{base_filename}.jpeg"
                try:
                    S3_CLIENT.put_object( # type: ignore
                        Bucket=DESTINATION_BUCKET_NAME,
                        Key=grayscale_s3_key,
                        Body=grayscale_image_bytes,
                        ContentType='image/jpeg'
                    )
                    logger.info(f"Saved grayscale image to s3://{DESTINATION_BUCKET_NAME}/{grayscale_s3_key}")
                except ClientError as e_save:
                    logger.error(f"Failed to save grayscale image {grayscale_s3_key} to S3: {e_save}")
                    # Decide if this failure should count as a failed file or just log and continue OCR

                # Perform OCR on grayscale image
                extracted_text = extract_text_from_image(grayscale_image_bytes)

                if not extracted_text.strip():
                    logger.warning("No text extracted from %s (identifier: %s), or OCR returned empty.", object_key, image_identifier)
                
                # Insert/Replace into SQLite
                cursor.execute('''
                    INSERT OR REPLACE INTO ocr_results (image_identifier, extracted_text, source_type, timestamp)
                    VALUES (?, ?, ?, ?)
                ''', (image_identifier, extracted_text, 'S3', datetime.now()))
                conn.commit()
                
                logger.info(
                    "Successfully processed and stored OCR text for S3 image %s (identifier: %s) in SQLite.", 
                    object_key, image_identifier
                )                    
                processed_files_count += 1
            
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code")
                if error_code == 'NoSuchKey':
                    logger.info(f"S3 object not found: {object_key}. Continuing to check next file number.")
                    # The finally block will increment current_file_number and files_checked_in_sequence.
                    continue # Continue to the next iteration of the while loop
                elif error_code == 'AccessDenied':
                    err_msg = f"S3 Access Denied for {object_key}: {e.response.get('Error', {}).get('Message', str(e))}"
                    logger.error(err_msg)
                    error_message = err_msg # Store the first critical error
                    break # Critical error, stop processing this sequence
                else:
                    logger.error(f"S3 ClientError processing {object_key}: {e.response.get('Error', {}).get('Message', str(e))} (Code: {error_code})")
                    failed_files_count += 1
            except Exception as e_proc:
                logger.error(f"Unexpected error processing {object_key}: {e_proc}", exc_info=True)
                failed_files_count += 1
            finally:
                current_file_number += 1
                files_checked_in_sequence += 1

        if files_checked_in_sequence >= MAX_SEQUENTIAL_FILES_TO_CHECK:
            warn_msg = f"Reached maximum sequential file check limit ({MAX_SEQUENTIAL_FILES_TO_CHECK}) for prefix {source_prefix}."
            logger.warning(warn_msg)
            if not error_message: # Don't overwrite a more critical error
                error_message = warn_msg 

        logger.info("Sequential S3 OCR process finished. Attempted: %d, Processed successfully: %d, Failed: %d", files_checked_in_sequence, processed_files_count, failed_files_count)
    except ClientError as e:
        # This would catch errors from S3_CLIENT.get_object if not caught by inner try-except,
        # or other S3 client issues not related to a specific object.
        error_message = f"S3 ClientError during S3 processing task for s3://{source_bucket}/{source_prefix}: {e}"
        logger.error(error_message)
    except Exception:
        error_message = f"Unexpected error during S3 OCR task for s3://{source_bucket}/{source_prefix}"
        logger.error(error_message, exc_info=True)
    finally:
        if conn:
            conn.close()
    return {"processed_files": processed_files_count, "failed_files": failed_files_count, "error": error_message}

def process_images(source_dir: str) -> Dict[str, Any]:
    logger.info("Starting LOCAL OCR process for source directory: %s", source_dir)
    processed_files_count = 0
    failed_files_count = 0
    error_message = None

    if not os.path.isdir(source_dir):
        logger.error("Local source directory not found: %s", source_dir)
        return {"processed_files": 0, "failed_files": 0, "error": f"Local source directory not found: {source_dir}"}

    conn = None
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
    except sqlite3.Error as e:
        logger.error(f"Failed to connect to SQLite DB: {e}")
        return {"processed_files": 0, "failed_files": 0, "error": f"Failed to connect to SQLite DB: {e}"}

    # Ensure the grayscale destination directory exists
    try:
        os.makedirs(LOCAL_GRAYSCALE_DESTINATION_DIR, exist_ok=True)
    except OSError as e:
        logger.error(f"Could not create grayscale directory {LOCAL_GRAYSCALE_DESTINATION_DIR}: {e}", exc_info=True)
        # Decide if this is a fatal error for the whole process or just for saving grayscale images
    try:
        image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']
        
        for filename in os.listdir(source_dir):
            source_file_path = os.path.join(source_dir, filename)
            if not os.path.isfile(source_file_path) or not any(filename.lower().endswith(ext) for ext in image_extensions):
                logger.debug("Skipping non-image file or directory: %s", source_file_path)
                continue
            
            image_identifier = filename

            logger.info("Processing local image: %s", source_file_path)
            try:
                with open(source_file_path, 'rb') as f:
                    image_bytes = f.read()

                # Convert to grayscale and save
                grayscale_image_bytes = convert_to_grayscale(image_bytes)

                base_filename, _ = os.path.splitext(filename)
                grayscale_image_path = os.path.join(LOCAL_GRAYSCALE_DESTINATION_DIR, f"{base_filename}.jpeg")
                
                try:
                    with open(grayscale_image_path, 'wb') as gf:
                        gf.write(grayscale_image_bytes)
                    logger.info(f"Saved grayscale image to {grayscale_image_path}")
                except IOError as e:
                    logger.error(f"IOError saving grayscale image {grayscale_image_path}: {e}") # Perform OCR on grayscale
                extracted_text = extract_text_from_image(grayscale_image_bytes)

                if not extracted_text.strip():
                    logger.warning("No text extracted from %s, or OCR returned empty.", filename)

                # Insert/Replace into SQLite
                cursor.execute('''
                    INSERT OR REPLACE INTO ocr_results (image_identifier, extracted_text, source_type, timestamp)
                    VALUES (?, ?, ?, ?)
                ''', (image_identifier, extracted_text, 'LOCAL', datetime.now()))
                conn.commit()

                logger.info("Successfully processed and stored OCR text for local image %s in SQLite.", filename)
                processed_files_count += 1
            except FileNotFoundError:
                logger.error("File not found during processing: %s", source_file_path)
                failed_files_count += 1
            except IOError as e:
                logger.error("IOError processing local file %s: %s", filename, e)
                failed_files_count += 1
            except Exception as e:
                logger.error("Unexpected error processing local file %s: %s", filename, e, exc_info=True)
                failed_files_count += 1
        logger.info("Local OCR process finished. Processed: %d, Failed: %d", processed_files_count, failed_files_count)
    except Exception:
        error_message = f"Unexpected error during local OCR task for source: {source_dir}"
        logger.error(error_message, exc_info=True)
    finally:
        if conn:
            conn.close()
    return {"processed_files": processed_files_count, "failed_files": failed_files_count, "error": error_message}

# --- Main Function ---
def trigger_ocr_processing(args: argparse.Namespace):
    """
    Triggers a synchronous process to perform OCR on images from the configured source
    (S3 or local) and saves results to the configured destination.
    Prints the outcome to the console.
    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    # start_file_number will be the value provided by the user,
    # or the default specified in argparse if not provided.
    start_file_num = args.start_file_number 
    
    init_sqlite_db()
    logger.info("OCR processing script started. Mode: %s", PROCESSING_MODE) # Use local logger and PROCESSING_MODE
    response_details = {}
    success = False

    try:
        if PROCESSING_MODE == "S3": # Use local PROCESSING_MODE
            if not S3_CLIENT: # Use local S3_CLIENT
                print("Error: S3 client not available. Check AWS configuration for S3 mode.", file=sys.stderr)
                sys.exit(1)
            if not SOURCE_BUCKET_NAME or not DESTINATION_BUCKET_NAME: # Use local config vars
                print("Error: S3 bucket names not configured for S3 mode.", file=sys.stderr)
                sys.exit(1)
            
            logger.info(f"S3 mode: Effective starting file number for sequential processing: {start_file_num}")
            logger.info("Starting S3 OCR processing synchronously.")
            processing_stats = process_images_from_s3_task( # Call local function
                SOURCE_BUCKET_NAME,
                SOURCE_PREFIX,
                start_file_number=start_file_num # Pass the determined start_file_num
            )
            logger.info("S3 OCR processing finished.")

            response_details = {
                "mode": "S3",
                "source_bucket": SOURCE_BUCKET_NAME,
                "source_prefix": SOURCE_PREFIX,
                "destination_bucket": DESTINATION_BUCKET_NAME,
                "destination_prefix": GRAYSCALE_DESTINATION_PREFIX
            }
            response_details.update(processing_stats)
            message = "S3 OCR processing completed."
            success = processing_stats.get("error") is None


        elif PROCESSING_MODE == "LOCAL": # Use local PROCESSING_MODE
            if not LOCAL_IMAGE_SOURCE_DIR:
                print("Error: Local source or destination directory not configured for LOCAL mode.", file=sys.stderr)
                sys.exit(1)
            
            logger.info("Starting local OCR processing synchronously.")
            processing_stats = process_images(LOCAL_IMAGE_SOURCE_DIR)
            logger.info("Local OCR processing finished.")

            response_details = {
                    "mode": "LOCAL",
                    "source_directory": LOCAL_IMAGE_SOURCE_DIR,
                    "database_path": SQLITE_DB_PATH
            }
            response_details.update(processing_stats)
            message = "Local OCR processing completed."
            success = processing_stats.get("error") is None

        else:
            print(f"Error: Invalid PROCESSING_MODE: {PROCESSING_MODE}. Must be 'S3' or 'LOCAL'.", file=sys.stderr)
            sys.exit(1)

        final_output = {"message": message, "details": response_details}
        print(json.dumps(final_output, indent=2))
        
        if not success and response_details.get("error"):
             print(f"\nProcessing encountered an error: {response_details['error']}", file=sys.stderr)
             sys.exit(1)
        elif not success:
             print(f"\nProcessing may have failed for some files. Check logs for details.", file=sys.stderr)
             sys.exit(1)


    except Exception as e:
        logger.error(f"An unexpected error occurred in standalone OCR script: {e}", exc_info=True) # Use local logger
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trigger OCR processing for images.")
    parser.add_argument(
        "start_file_number",  # Positional argument
        type=int,
        nargs='?',             # Makes it optional
        default=5000000,       # Default value if not provided by the user
        help="Optional: The starting file number for S3 sequential processing (default: 5000000)."
    )
    cli_args = parser.parse_args()
    trigger_ocr_processing(cli_args)
