import logging
import os
import pickle
import sys

import boto3
import faiss
import numpy as np
from dotenv import load_dotenv
from botocore.exceptions import ClientError

# Add the project root directory to the Python path to resolve module imports.
# This allows the script to find the 'services' module when run directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.image_similarity_service import ImageSimilarityService

# --- Load environment variables ---
load_dotenv()

# --- S3 Client Initialization ---
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")

# --- Configuration ---
# These should point to the S3 location of your original, high-quality images.
ORIGINAL_IMAGE_BUCKET = os.getenv("ORIGINAL_IMAGE_BUCKET", "newbucket-trademark")
ORIGINAL_IMAGE_PREFIX = os.getenv("ORIGINAL_IMAGE_PREFIX", "images/original/")

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "image_index.faiss")
KEY_MAP_PATH = os.getenv("KEY_MAP_PATH", "index_to_key_map.pkl")
BATCH_SIZE = 1000  # Process 1000 images at a time to conserve memory
CHECKPOINT_INTERVAL = 10  # Save a checkpoint to disk every 10 batches

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

S3_CLIENT = None
if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and AWS_DEFAULT_REGION:
    try:
        S3_CLIENT = boto3.client('s3')
        logger.info("S3 client initialized successfully.")
    except Exception as e:
        logger.error("Failed to initialize S3 client.", exc_info=True)
else:
    logger.warning("AWS credentials or region not fully configured. S3 operations will fail.")


def _save_checkpoint(index: faiss.Index, key_map: list, index_path: str, map_path: str):
    """
    Saves the current index and key map to temporary files for resilience.
    This prevents loss of progress on very long indexing jobs.
    """
    tmp_index_path = index_path + ".tmp"
    tmp_map_path = map_path + ".tmp"

    logger.info(f"Saving checkpoint with {index.ntotal} vectors to {tmp_index_path}...")
    faiss.write_index(index, tmp_index_path)

    with open(tmp_map_path, 'wb') as f:
        pickle.dump(key_map, f)
    logger.info("Checkpoint saved successfully.")


def build_index():
    """
    Reads images directly from an S3 bucket, extracts features using CLIP, and builds a Faiss index.
    """
    logger.info(f"Starting image index build from s3://{ORIGINAL_IMAGE_BUCKET}/{ORIGINAL_IMAGE_PREFIX}")
    if not S3_CLIENT:
        logger.error("S3 client not initialized. Cannot build index.")
        return

    # 1. Initialize the similarity service to get access to the CLIP model
    similarity_service = ImageSimilarityService()
    paginator = S3_CLIENT.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=ORIGINAL_IMAGE_BUCKET, Prefix=ORIGINAL_IMAGE_PREFIX)

    index = None
    s3_key_map = []
    batch_features = []
    batch_keys = []
    batch_num = 0
    processed_count = 0

    try:
        for page in pages:
            if 'Contents' not in page:
                continue

            for s3_object in page['Contents']:
                s3_key = s3_object['Key']
                # Skip "directory" objects and non-images
                if s3_key.endswith('/') or not (s3_key.lower().endswith(('.jpg', '.jpeg', '.png'))):
                    continue

                processed_count += 1
                logger.info(f"Processing S3 object #{processed_count}: {s3_key}")

                try:
                    s3_response = S3_CLIENT.get_object(Bucket=ORIGINAL_IMAGE_BUCKET, Key=s3_key)
                    image_bytes = s3_response['Body'].read()
                    features = similarity_service.extract_features(image_bytes)

                    # Initialize index with dimension from first successful feature extraction
                    if index is None:
                        dimension = features.shape[0]
                        logger.info(f"CLIP feature dimension is {dimension}.")
                        index = faiss.IndexFlatL2(dimension)

                    batch_features.append(features)
                    batch_keys.append(s3_key)

                    # When batch is full, add to index and reset
                    if len(batch_features) >= BATCH_SIZE:
                        features_np = np.array(batch_features).astype('float32')
                        index.add(features_np) # type: ignore
                        s3_key_map.extend(batch_keys)
                        logger.info(f"Added {len(batch_features)} vectors to index. Total vectors: {index.ntotal}")

                        batch_features.clear()
                        batch_keys.clear()
                        batch_num += 1

                        if batch_num % CHECKPOINT_INTERVAL == 0:
                            _save_checkpoint(index, s3_key_map, FAISS_INDEX_PATH, KEY_MAP_PATH)

                except Exception as e:
                    logger.error(f"Could not process image {s3_key}: {e}")

        # Add any remaining features from the last partial batch
        if index and batch_features:
            features_np = np.array(batch_features).astype('float32')
            index.add(features_np) # type: ignore
            s3_key_map.extend(batch_keys)
            logger.info(f"Added final {len(batch_features)} vectors to index. Total vectors: {index.ntotal}")

        # Save the final index
        if index and index.ntotal > 0:
            logger.info("Processing complete. Saving final index and map...")
            _save_checkpoint(index, s3_key_map, FAISS_INDEX_PATH, KEY_MAP_PATH)
            os.rename(FAISS_INDEX_PATH + ".tmp", FAISS_INDEX_PATH)
            os.rename(KEY_MAP_PATH + ".tmp", KEY_MAP_PATH)
            logger.info(f"Final index with {index.ntotal} vectors successfully saved to {FAISS_INDEX_PATH}")
        else:
            logger.warning("No images were processed or no features were added to the index. Nothing to save.")
    except ClientError as e:
        logger.error(f"A critical S3 error occurred during index build: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred during index build: {e}", exc_info=True)

if __name__ == "__main__":
    build_index()