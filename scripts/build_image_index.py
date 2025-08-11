import logging
import os
import pickle
import sys

import boto3
import faiss
import numpy as np
from dotenv import load_dotenv
from botocore.exceptions import ClientError
from src.config import (OCR_SOURCE_S3_PREFIX, FAISS_INDEX_PATH, KEY_MAP_PATH,
                        OCR_SOURCE_BUCKET_NAME)

# Add the project root directory to the Python path to resolve module imports.
# This allows the script to find the 'services' module when run directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.image_similarity_service import ImageSimilarityService

# --- Load environment variables ---
load_dotenv()

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IndexBuilder:
    """
    Encapsulates the logic for building a Faiss index from images stored in S3.
    """
    def __init__(self):
        """Initializes the IndexBuilder, loading configuration and services."""
        # --- Configuration ---
        self.original_image_bucket = OCR_SOURCE_BUCKET_NAME
        self.original_image_prefix = OCR_SOURCE_S3_PREFIX
        self.faiss_index_path = FAISS_INDEX_PATH
        self.key_map_path = KEY_MAP_PATH
        self.batch_size = int(os.getenv("BATCH_SIZE", 1000))
        self.checkpoint_interval = int(os.getenv("CHECKPOINT_INTERVAL", 10))

        # --- Service and Client Initialization ---
        self.s3_client = self._initialize_s3_client()
        self.similarity_service = ImageSimilarityService()

        # --- State Variables ---
        self.index = None
        self.s3_key_map = []
        self.batch_features = []
        self.batch_keys = []
        self.batch_num = 0
        self.processed_count = 0

    def _initialize_s3_client(self):
        """Initializes and returns a Boto3 S3 client."""
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_default_region = os.getenv("AWS_DEFAULT_REGION")

        if not (aws_access_key_id and aws_secret_access_key and aws_default_region):
            logger.warning("AWS credentials or region not fully configured. S3 operations might fail.")
            return None
        try:
            client = boto3.client('s3')
            logger.info("S3 client initialized successfully.")
            return client
        except Exception as e:
            logger.error("Failed to initialize S3 client.", exc_info=True)
            return None

    def _load_checkpoint(self):
        """Loads index and key map from checkpoint files if they exist."""
        tmp_index_path = self.faiss_index_path + ".tmp"
        tmp_map_path = self.key_map_path + ".tmp"

        if os.path.exists(tmp_index_path) and os.path.exists(tmp_map_path):
            try:
                logger.info(f"Resuming from checkpoint: loading {tmp_index_path} and {tmp_map_path}")
                self.index = faiss.read_index(tmp_index_path)
                with open(tmp_map_path, 'rb') as f:
                    self.s3_key_map = pickle.load(f)

                # Update state to reflect resumed progress
                self.processed_count = self.index.ntotal
                self.batch_num = self.processed_count // self.batch_size

                logger.info(f"Resumed successfully. Index has {self.index.ntotal} vectors. Starting from batch number {self.batch_num + 1}.")
                return set(self.s3_key_map)
            except Exception as e:
                logger.error(f"Failed to load checkpoint files: {e}. Starting from scratch.", exc_info=True)
                self.index = None
                self.s3_key_map = []

    def _save_index_and_map(self, final: bool = False):
        """
        Saves the current index and key map. If final, renames temp files to permanent ones.
        """
        if not self.index or self.index.ntotal == 0:
            logger.warning("Index is empty or not initialized. Nothing to save.")
            return

        tmp_index_path = self.faiss_index_path + ".tmp"
        tmp_map_path = self.key_map_path + ".tmp"

        action = "Saving final index" if final else "Saving checkpoint"
        logger.info(f"{action} with {self.index.ntotal} vectors to {tmp_index_path}...")

        faiss.write_index(self.index, tmp_index_path)
        with open(tmp_map_path, 'wb') as f:
            pickle.dump(self.s3_key_map, f)

        if final:
            os.rename(tmp_index_path, self.faiss_index_path)
            os.rename(tmp_map_path, self.key_map_path)
            logger.info(f"Final index with {self.index.ntotal} vectors successfully saved to {self.faiss_index_path}")
        else:
            logger.info("Checkpoint saved successfully.")

    def _process_batch(self):
        """Processes the current batch of features and adds them to the Faiss index."""
        if not self.batch_features:
            return

        assert self.index is not None, "Index should be initialized before processing a batch"

        try:
            features_np = np.array(self.batch_features).astype('float32')
            self.index.add(features_np) # type: ignore
            self.s3_key_map.extend(self.batch_keys)
            logger.info(f"Added {len(self.batch_features)} vectors to index. Total vectors: {self.index.ntotal}")

            self.batch_features.clear()
            self.batch_keys.clear()
            self.batch_num += 1

            if self.batch_num % self.checkpoint_interval == 0:
                self._save_index_and_map(final=False)
        except Exception as e:
            logger.error(f"Failed to process batch: {e}", exc_info=True)

    def _process_s3_object(self, s3_key: str):
        """Downloads an S3 object, extracts features, and adds to the current batch."""
        try:
            self.processed_count += 1
            logger.info(f"Processing S3 object #{self.processed_count}: {s3_key}")

            s3_response = self.s3_client.get_object(Bucket=self.original_image_bucket, Key=s3_key) # type: ignore
            image_bytes = s3_response['Body'].read()
            features = self.similarity_service.extract_features(image_bytes)

            if self.index is None:
                dimension = features.shape[0]
                logger.info(f"Initializing index with feature dimension {dimension}.")
                self.index = faiss.IndexFlatL2(dimension)

            self.batch_features.append(features)
            self.batch_keys.append(s3_key)
        except Exception as e:
            logger.error(f"Could not process image {s3_key}: {e}")

    def run(self):
        """
        Main method to build the Faiss index from images in S3.
        """
        if not self.s3_client:
            logger.error("S3 client not initialized. Cannot build index.")
            return

        # Load from checkpoint if available and get a set of already processed keys
        self._load_checkpoint()
        processed_keys = self.s3_key_map
        if processed_keys:
            logger.info(f"Skipping {len(processed_keys)} already processed S3 objects found in checkpoint.")

        logger.info(f"Starting image index build from s3://{self.original_image_bucket}/{self.original_image_prefix}")

        
        paginator = self.s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self.original_image_bucket, Prefix=self.original_image_prefix)

        try:
            total_s3_objects = 0
            for page in pages:
                if 'Contents' not in page:
                    continue

                for s3_object in page['Contents']:
                    total_s3_objects += 1
                    s3_key = s3_object['Key']
                    if s3_key.endswith('/') or not any(s3_key.lower().endswith(ext) for ext in ('.jpg', '.jpeg', '.png')):
                        continue

                    if s3_key in processed_keys:
                        continue

                    self._process_s3_object(s3_key)

                    if len(self.batch_features) >= self.batch_size:
                        self._process_batch()

            # Process any remaining features in the last batch
            if self.batch_features:
                self._process_batch()

            # Save the final index
            self._save_index_and_map(final=True)
            if not self.index or self.index.ntotal == len(processed_keys):
                logger.warning(f"No new images were found to be indexed out of {total_s3_objects} S3 objects checked.")

        except ClientError as e:
            logger.error(f"A critical S3 error occurred during index build: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"An unexpected error occurred during index build: {e}", exc_info=True)


if __name__ == "__main__":
    builder = IndexBuilder()
    builder.run()