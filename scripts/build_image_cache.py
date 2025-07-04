import os
import logging
import sqlite3
from datetime import datetime
import argparse
from typing import Optional

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

# --- Configuration ---
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")

# Configuration for the source of grayscale images
# These should point to where your grayscale images are stored in S3
# (e.g., the destination for grayscale images in trigger_ocr.py)
GRAYSCALE_IMAGE_BUCKET = os.getenv("GRAYSCALE_DESTINATION_BUCKET_NAME", # From trigger_ocr.py
                                   os.getenv("GRAYSCALE_BUCKET_NAME", "newbucket-trademark"))
GRAYSCALE_IMAGE_PREFIX = os.getenv("GRAYSCALE_DESTINATION_PREFIX", # From trigger_ocr.py
                                   os.getenv("GRAYSCALE_S3_PREFIX", "images/grayscale/"))

SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "ocr_results.db")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- S3 Client Initialization ---
S3_CLIENT = None
if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and AWS_DEFAULT_REGION:
    try:
        S3_CLIENT = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_DEFAULT_REGION
        )
        logger.info("S3 client initialized successfully.")
    except Exception as e:
        logger.error("Failed to initialize S3 client.", exc_info=True)
else:
    logger.warning("AWS credentials or region not fully configured. S3 operations will fail.")

def init_sqlite_db_for_cache():
    """Initializes the SQLite database and creates the s3_image_cache table if it doesn't exist."""
    conn = None
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        # Create table for S3 image bytes cache (idempotent)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS s3_image_cache (
                s3_key TEXT PRIMARY KEY NOT NULL,
                image_bytes BLOB NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Optional: Index for faster lookups if needed elsewhere, though PRIMARY KEY is indexed.
        # cursor.execute('''
        #     CREATE INDEX IF NOT EXISTS idx_s3_image_cache_timestamp ON s3_image_cache (timestamp);
        # ''')
        conn.commit()
        logger.info(f"SQLite database and s3_image_cache table ensured at {SQLITE_DB_PATH}")
    except sqlite3.Error as e:
        logger.error(f"Error initializing SQLite database for cache: {e}", exc_info=True)
        if conn:
            conn.close()
        raise  # Re-raise to stop script if DB init fails
    finally:
        if conn:
            conn.close()

def populate_s3_image_cache(
    bucket_name: str,
    s3_prefix: str,
    max_images: Optional[int] = None
):
    """
    Lists images from S3, downloads them, and stores them in the SQLite s3_image_cache.

    Args:
        bucket_name: The S3 bucket name where grayscale images are stored.
        s3_prefix: The S3 prefix (folder) for grayscale images.
        max_images: Optional limit on the number of images to cache.
    """
    if not S3_CLIENT:
        logger.error("S3 client not initialized. Cannot populate cache.")
        return

    if not bucket_name or not s3_prefix:
        logger.error("S3 bucket name or prefix for grayscale images is not configured.")
        return

    logger.info(f"Starting S3 image cache population from s3://{bucket_name}/{s3_prefix}")
    logger.info(f"Database: {SQLITE_DB_PATH}")
    if max_images is not None:
        logger.info(f"Will cache a maximum of {max_images} images.")

    conn = None
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
    except sqlite3.Error as e:
        logger.error(f"Failed to connect to SQLite DB: {e}")
        return

    processed_count = 0
    cached_count = 0
    error_count = 0
    continuation_token = None

    try:
        while True:
            list_kwargs = {'Bucket': bucket_name, 'Prefix': s3_prefix}
            if continuation_token:
                list_kwargs['ContinuationToken'] = continuation_token

            response = S3_CLIENT.list_objects_v2(**list_kwargs) # type: ignore
            
            if 'Contents' not in response:
                logger.info("No objects found in S3 at the specified prefix.")
                break

            for s3_object in response['Contents']:
                if max_images is not None and processed_count >= max_images:
                    logger.info(f"Reached maximum image limit of {max_images}. Stopping.")
                    break # Break from inner loop

                s3_key = s3_object['Key']
                processed_count += 1

                # Skip if it's a "directory" object or not an image (basic check)
                if s3_key.endswith('/') or not (s3_key.lower().endswith('.jpg') or s3_key.lower().endswith('.jpeg') or s3_key.lower().endswith('.png')):
                    logger.debug(f"Skipping non-image or directory object: {s3_key}")
                    continue

                logger.info(f"Processing S3 object: {s3_key} ({processed_count} processed)")

                try:
                    # Check if already cached and recent (optional optimization)
                    # For simplicity, we'll just INSERT OR REPLACE to ensure freshness
                    # cursor.execute("SELECT timestamp FROM s3_image_cache WHERE s3_key = ?", (s3_key,))
                    # existing = cursor.fetchone()
                    # if existing and (datetime.now() - datetime.fromisoformat(existing[0])).days < 1:
                    #     logger.info(f"Image {s3_key} already in cache and recent. Skipping download.")
                    #     continue

                    s3_response = S3_CLIENT.get_object(Bucket=bucket_name, Key=s3_key) # type: ignore
                    image_bytes = s3_response['Body'].read()

                    cursor.execute(
                        "INSERT OR REPLACE INTO s3_image_cache (s3_key, image_bytes, timestamp) VALUES (?, ?, ?)",
                        (s3_key, image_bytes, datetime.now())
                    )
                    conn.commit()
                    cached_count +=1
                    logger.info(f"Successfully cached image: {s3_key} ({len(image_bytes) / 1024:.2f} KB)")

                except ClientError as e:
                    logger.error(f"S3 ClientError processing {s3_key}: {e}", exc_info=True)
                    error_count += 1
                except sqlite3.Error as e:
                    logger.error(f"SQLite error caching {s3_key}: {e}", exc_info=True)
                    error_count += 1
                    # If DB error, might be good to break or handle more gracefully
                except Exception as e:
                    logger.error(f"Unexpected error processing {s3_key}: {e}", exc_info=True)
                    error_count += 1
            
            if max_images is not None and processed_count >= max_images:
                break # Break from outer while loop

            if response.get('IsTruncated'):
                continuation_token = response.get('NextContinuationToken')
                logger.info("Fetching next page of S3 objects...")
            else:
                break # No more objects

    except ClientError as e:
        logger.error(f"S3 error during listing objects: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred during cache population: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()
        logger.info(
            f"Cache population finished. Total objects processed: {processed_count}. "
            f"Images newly cached/updated: {cached_count}. Errors: {error_count}."
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Populate S3 image cache in SQLite.")
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process and cache. If not set, processes all images under the prefix."
    )
    args = parser.parse_args()

    if not S3_CLIENT:
        logger.error("S3 client failed to initialize. Exiting.")
    else:
        try:
            init_sqlite_db_for_cache() # Ensure DB and table exist
            populate_s3_image_cache(
                bucket_name=GRAYSCALE_IMAGE_BUCKET,
                s3_prefix=GRAYSCALE_IMAGE_PREFIX,
                max_images=args.max_images
            )
        except Exception as e:
            logger.fatal(f"Unhandled exception in main execution: {e}", exc_info=True)