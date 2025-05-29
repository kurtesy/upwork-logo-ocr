import os
import io
import logging
import sys
import json
import sqlite3
from datetime import datetime
from typing import Dict, Any # For type hinting

import boto3
from botocore.exceptions import ClientError
from PIL import Image, ImageOps
import pytesseract
# Keras-OCR is imported within its helper function to handle potential ImportError
from dotenv import load_dotenv

# --- OCR Library Imports (add as needed) ---
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    easyocr = None # type: ignore

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
if TESSERACT_CMD:
    pytesseract.tesseract_cmd = TESSERACT_CMD # type: ignore

# --- Logging Setup (moved from main.py) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Using __name__ for the logger

logger.info(f"Processing Mode: {PROCESSING_MODE}")

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
        reader = easyocr.Reader(['en', 'hi'], gpu=False) # Specify gpu=True if a GPU is available and configured
        result = reader.readtext(image_bytes)
        text = " ".join([item[1] for item in result]) # type: ignore
        return text
    except Exception:
        logger.error("Error during EasyOCR", exc_info=True)
        return ""

# --- OCR Helper Functions (moved and adapted from main.py) ---
def extract_text_with_pytesseract(image_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        return text
    except Exception:
        logger.error("Error during Pytesseract OCR", exc_info=True)
        return ""

# def extract_text_with_kerasocr(image_bytes: bytes) -> str:
#     try:
#         import keras_ocr # Import here to keep dependency local to function
#         pipeline = keras_ocr.pipeline.Pipeline()
#         image = keras_ocr.tools.read(image_bytes) # type: ignore
#         prediction_groups = pipeline.recognize([image])
#         text = " ".join([word_info[0] for word_info in prediction_groups[0]])
#         return text
#     except ImportError:
#         logger.error("Keras-OCR library is not installed. Cannot use Keras-OCR.")
#         return ""
#     except Exception:
#         logger.error("Error during Keras-OCR", exc_info=True)
#         return ""

# --- Processing Task Functions (moved and adapted from main.py) ---
def process_images_from_s3_task(
    source_bucket: str, source_prefix: str
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


    try:
        paginator = S3_CLIENT.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=source_bucket, Prefix=source_prefix)

        for page in page_iterator:
            if "Contents" not in page:
                continue
            for obj in page["Contents"]:
                object_key = obj["Key"]
                if object_key.endswith('/') or not any(object_key.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']):
                    logger.debug("Skipping non-image or directory: %s", object_key)
                    continue

                logger.info("Processing image: s3://%s/%s", source_bucket, object_key)
                try:
                    response = S3_CLIENT.get_object(Bucket=source_bucket, Key=object_key)
                    image_bytes = response['Body'].read()

                    # Convert to grayscale
                    grayscale_image_bytes = convert_to_grayscale(image_bytes)

                    # Determine relative path for destination
                    relative_path = object_key[len(source_prefix):] if object_key.startswith(source_prefix) else object_key
                    base_filename, _ = os.path.splitext(relative_path) # Get filename without extension

                    # Save grayscale image to S3
                    grayscale_s3_key = f"{GRAYSCALE_DESTINATION_PREFIX.rstrip('/')}/{base_filename}.jpeg"
                    try:
                        S3_CLIENT.put_object(
                            Bucket=DESTINATION_BUCKET_NAME, # Assuming same bucket for grayscale, or use a specific one
                            Key=grayscale_s3_key,
                            Body=grayscale_image_bytes,
                            ContentType='image/jpeg'
                        )
                        logger.info(f"Saved grayscale image to s3://{DESTINATION_BUCKET_NAME}/{grayscale_s3_key}")
                    except ClientError as e:
                        logger.error(f"Failed to save grayscale image {grayscale_s3_key} to S3: {e}")

                    # Perform OCR on grayscale image
                    extracted_text = extract_text_with_easyocr(grayscale_image_bytes) # Using EasyOCR for consistency, or Pytesseract

                    if not extracted_text.strip():
                        logger.warning("No text extracted from %s, or OCR returned empty.", object_key)
                    image_identifier = relative_path
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
                    logger.error("S3 ClientError processing %s: %s", object_key, e)
                    failed_files_count += 1
                except Exception as e:
                    logger.error("Unexpected error processing %s: %s", object_key, e, exc_info=True)
                    failed_files_count += 1
        logger.info("OCR process finished. Processed: %d, Failed: %d", processed_files_count, failed_files_count)
    except ClientError as e:
        error_message = f"S3 ClientError during listing objects from s3://{source_bucket}/{source_prefix}: {e}"
        logger.error(error_message)
    except Exception:
        error_message = f"Unexpected error during OCR task for s3://{source_bucket}/{source_prefix}"
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
                    logger.error(f"IOError saving grayscale image {grayscale_image_path}: {e}")
                extracted_text = extract_text_with_easyocr(grayscale_image_bytes) # Perform OCR on grayscale

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
def trigger_ocr_processing():
    """
    Triggers a synchronous process to perform OCR on images from the configured source
    (S3 or local) and saves results to the configured destination.
    Prints the outcome to the console.
    """
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
            
            logger.info("Starting S3 OCR processing synchronously.")
            processing_stats = process_images_from_s3_task( # Call local function
                SOURCE_BUCKET_NAME,
                SOURCE_PREFIX
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
    trigger_ocr_processing()
