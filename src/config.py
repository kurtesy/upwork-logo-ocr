import os
import logging
from dotenv import load_dotenv
import boto3

# Load environment variables from .env file (if it exists)
load_dotenv()

# --- Core Configuration ---
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
API_KEY = os.getenv("API_KEY")
API_KEY_NAME = "X-API-KEY"

SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "ocr_results.db")

# --- S3 Bucket/Prefix Configuration ---
GRAYSCALE_BUCKET_NAME = os.getenv("GRAYSCALE_BUCKET_NAME", "newbucket-trademark")
GRAYSCALE_S3_PREFIX = os.getenv("GRAYSCALE_S3_PREFIX", "images/grayscale/")

OCR_SOURCE_BUCKET_NAME = os.getenv("SOURCE_BUCKET_NAME", "newbucket-trademark")
OCR_SOURCE_S3_PREFIX = os.getenv("SOURCE_PREFIX", "images/original/")

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "image_index.faiss.tmp")
KEY_MAP_PATH = os.getenv("KEY_MAP_PATH", "index_to_key_map.pkl.tmp")

# --- S3 Client Initialization ---
S3_CLIENT = None
if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and AWS_DEFAULT_REGION:
    try:
        S3_CLIENT = boto3.client(
            's3', aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=AWS_DEFAULT_REGION)
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to initialize S3 client in config: {e}", exc_info=True)