import os
import io
import logging
import sys
import json
import sqlite3
from datetime import datetime, timezone
import argparse # For command-line arguments
from typing import Dict, Any # For type hinting

import boto3
from botocore.exceptions import ClientError
from PIL import Image, ImageOps
import pytesseract
from dotenv import load_dotenv
import cv2 # For image decoding for RapidOCR
import numpy as np # For image decoding for RapidOCR

# --- Load environment variables ---
load_dotenv()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# --- Logging Setup (moved from main.py) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Using __name__ for the logger

class OcrProcessor:
    """Encapsulates the logic for processing images for OCR from various sources."""

    def __init__(self, start_file_number: int = 5000000, resume: bool = False):
        """Initializes the processor, loading configuration and setting up clients."""
        self._load_config()
        self.start_file_number = start_file_number
        self.resume = resume

        self.s3_client = self._initialize_s3_client()
        self.db_conn = self._initialize_db()
        self.cursor = self.db_conn.cursor()

        # Lazy load OCR engines
        self._easyocr_reader = None
        self._rapidocr_engine = None

        if self.resume and self.processing_mode == "S3":
            self.start_file_number = self._get_start_file_number_from_db()

        logger.info(f"--- OCR Script Configuration ---")
        logger.info(f"Processing Mode: {self.processing_mode}")
        logger.info(f"SQLite DB Path: {self.sqlite_db_path}")

    def _load_config(self):
        """Loads configuration from environment variables."""
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_default_region = os.getenv("AWS_DEFAULT_REGION")
        self.source_bucket_name = os.getenv("SOURCE_BUCKET_NAME", "newbucket-trademark")
        self.source_prefix = os.getenv("SOURCE_PREFIX", "images/real/")
        self.destination_bucket_name = os.getenv("DESTINATION_BUCKET_NAME", "newbucket-trademark")
        self.grayscale_destination_prefix = os.getenv("GRAYSCALE_DESTINATION_PREFIX", "images/grayscale/")
        self.processing_mode = os.getenv("PROCESSING_MODE", "S3").upper()
        self.local_image_source_dir = os.getenv("LOCAL_IMAGE_SOURCE_DIR", "./local_images/")
        self.local_grayscale_destination_dir = os.getenv("LOCAL_GRAYSCALE_DESTINATION_DIR", "./local_images_grayscale/")
        self.sqlite_db_path = os.getenv("SQLITE_DB_PATH", "ocr_results.db")
        self.ocr_engine_preference = os.getenv("OCR_ENGINE_PREFERENCE", "RAPIDOCR").upper()
        tesseract_cmd = os.getenv("TESSERACT_CMD")
        if tesseract_cmd:
            pytesseract.tesseract_cmd = tesseract_cmd # type: ignore

    def _initialize_s3_client(self):
        """Initializes and returns a Boto3 S3 client if in S3 mode."""
        if self.processing_mode != "S3":
            logger.info("Processing mode is not S3. S3 client will not be initialized.")
            return None
        if self.aws_access_key_id and self.aws_secret_access_key and self.aws_default_region:
            try:
                client = boto3.client('s3')
                logger.info("S3 client initialized successfully.")
                return client
            except Exception:
                logger.error("Failed to initialize S3 client.", exc_info=True)
        else:
            logger.warning("AWS credentials or region not fully configured. S3 operations will fail.")
        return None

    def _initialize_db(self) -> sqlite3.Connection:
        """Initializes the SQLite database and returns a connection."""
        try:
            conn = sqlite3.connect(self.sqlite_db_path)
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
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ocr_results_extracted_text ON ocr_results (extracted_text);')
            conn.commit()
            logger.info(f"SQLite database initialized successfully at {self.sqlite_db_path}")
            return conn
        except sqlite3.Error as e:
            logger.error(f"Error initializing SQLite database: {e}", exc_info=True)
            sys.exit(1)

    def _get_start_file_number_from_db(self) -> int:
        """Queries DB for the last processed file number to resume from."""
        logger.info("Resume flag is set. Attempting to find last processed file number from the database.")
        try:
            self.cursor.execute("""
                SELECT MAX(CAST(REPLACE(image_identifier, '.jpeg', '') AS INTEGER))
                FROM ocr_results
                WHERE source_type = 'S3' AND image_identifier GLOB '[0-9]*.jpeg'
            """)
            result = self.cursor.fetchone()
            if result and result[0] is not None:
                last_processed_num = result[0]
                new_start_num = last_processed_num + 1
                logger.info(f"Resuming from DB. Last processed file was {last_processed_num}. Starting from {new_start_num}.")
                return new_start_num
            else:
                logger.info("Could not determine last processed file from DB. Using provided start number: %d", self.start_file_number)
        except sqlite3.Error as e:
            logger.error(f"Could not query DB to resume: {e}. Using provided start number: %d", self.start_file_number)
        return self.start_file_number

    @property
    def easyocr_reader(self):
        if self._easyocr_reader is None:
            try:
                import easyocr
                logger.info("Initializing EasyOCR reader...")
                self._easyocr_reader = easyocr.Reader(['en', 'hi'], gpu=False, quantize=False)
            except ImportError:
                logger.warning("EasyOCR library is not installed. It will not be available.")
        return self._easyocr_reader

    @property
    def rapidocr_engine(self):
        if self._rapidocr_engine is None:
            try:
                from rapidocr import RapidOCR
                logger.info("Initializing RapidOCR engine...")
                self._rapidocr_engine = RapidOCR()
            except ImportError:
                logger.warning("RapidOCR library is not installed. It will not be available.")
        return self._rapidocr_engine

    def _convert_to_grayscale(self, image_bytes: bytes) -> bytes:
        """Converts an image from bytes to grayscale JPEG bytes."""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            grayscale_image = ImageOps.grayscale(image)
            buffer = io.BytesIO()
            grayscale_image.save(buffer, format="JPEG")
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Error converting image to grayscale: {e}", exc_info=True)
            return image_bytes

    def _extract_text_with_pytesseract(self, image_bytes: bytes) -> str:
        try:
            image = Image.open(io.BytesIO(image_bytes))
            return pytesseract.image_to_string(image)
        except Exception as e:
            logger.error(f"Error during Pytesseract OCR: {e}", exc_info=True)
            return ""

    def _extract_text_with_easyocr(self, image_bytes: bytes) -> str:
        if not self.easyocr_reader: return ""
        try:
            result = self.easyocr_reader.readtext(image_bytes)
            return " ".join([item[1] for item in result]) # type: ignore
        except Exception as e:
            logger.error(f"Error during EasyOCR: {e}", exc_info=True)
            return ""

    def _extract_text_with_rapidocr(self, image_bytes: bytes) -> str:
        if not self.rapidocr_engine: return ""
        try:
            image_np_array = np.frombuffer(image_bytes, np.uint8)
            img_cv2 = cv2.imdecode(image_np_array, cv2.IMREAD_COLOR)
            if img_cv2 is None:
                logger.error("Failed to decode image for RapidOCR.")
                return ""
            result = self.rapidocr_engine(img_cv2)
            # The result from rapidocr is an object with a .txts attribute, not an iterable.
            txts = getattr(result, 'txts', None)
            return " ".join(txts) if txts else ""
        except Exception as e:
            logger.error(f"Error during RapidOCR: {e}", exc_info=True)
            return ""

    def _extract_text_from_image(self, image_bytes: bytes) -> str:
        """Extracts text using the preferred OCR engine, with fallbacks."""
        engines_order = {
            "RAPIDOCR": [self._extract_text_with_rapidocr, self._extract_text_with_easyocr, self._extract_text_with_pytesseract],
            "EASYOCR": [self._extract_text_with_easyocr, self._extract_text_with_rapidocr, self._extract_text_with_pytesseract],
            "PYTESSERACT": [self._extract_text_with_pytesseract, self._extract_text_with_easyocr, self._extract_text_with_rapidocr],
        }
        attempt_order = engines_order.get(self.ocr_engine_preference, engines_order["EASYOCR"])

        for engine_func in attempt_order:
            engine_name = engine_func.__name__.replace("_extract_text_with_", "").upper()
            logger.info(f"Attempting OCR with {engine_name}...")
            text = engine_func(image_bytes)
            if text.strip():
                logger.info(f"Successfully extracted text using {engine_name}.")
                return text
            logger.warning(f"{engine_name} did not extract text.")
        logger.error("All configured OCR engines failed to extract text.")
        return ""

    def _process_image(self, image_identifier: str, image_bytes: bytes, source_type: str) -> bool:
        """Processes a single image: grayscale, OCR, and save."""
        try:
            grayscale_bytes = self._convert_to_grayscale(image_bytes)
            base_filename, _ = os.path.splitext(image_identifier)

            if source_type == 'S3':
                grayscale_s3_key = f"{self.grayscale_destination_prefix.rstrip('/')}/{base_filename}.jpeg"
                self.s3_client.put_object( # type: ignore
                    Bucket=self.destination_bucket_name,
                    Key=grayscale_s3_key,
                    Body=grayscale_bytes,
                    ContentType='image/jpeg'
                )
                logger.info(f"Saved grayscale image to s3://{self.destination_bucket_name}/{grayscale_s3_key}")
            else: # LOCAL
                os.makedirs(self.local_grayscale_destination_dir, exist_ok=True)
                grayscale_path = os.path.join(self.local_grayscale_destination_dir, f"{base_filename}.jpeg")
                with open(grayscale_path, 'wb') as gf:
                    gf.write(grayscale_bytes)
                logger.info(f"Saved grayscale image to {grayscale_path}")

            extracted_text = self._extract_text_from_image(grayscale_bytes)
            if not extracted_text.strip():
                logger.warning("No text extracted from %s.", image_identifier)

            self.cursor.execute('''
                INSERT OR REPLACE INTO ocr_results (image_identifier, extracted_text, source_type, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (image_identifier, extracted_text, source_type, datetime.now(timezone.utc).isoformat()))
            self.db_conn.commit()

            logger.info("Successfully processed and stored OCR for %s.", image_identifier)
            return True
        except Exception as e:
            logger.error(f"Unexpected error processing {image_identifier}: {e}", exc_info=True)
            return False

    def _iterate_s3_images(self):
        """Generator that yields (image_identifier, image_bytes) from S3."""
        if not self.s3_client:
            raise ConnectionError("S3 client not initialized.")

        current_file_num = self.start_file_number
        MAX_SEQUENTIAL_CHECKS = 100000000
        checked_count = 0

        logger.info(f"Starting sequential S3 scan from file number {current_file_num} under prefix {self.source_prefix}")

        while checked_count < MAX_SEQUENTIAL_CHECKS:
            s3_filename = f"{current_file_num}.jpeg"
            object_key = f"{self.source_prefix.rstrip('/')}/{s3_filename}"
            checked_count += 1
            current_file_num += 1

            self.cursor.execute("SELECT 1 FROM ocr_results WHERE image_identifier = ?", (s3_filename,))
            if self.cursor.fetchone():
                logger.info(f"Image {s3_filename} already processed. Skipping.")
                continue

            try:
                logger.info("Processing image: s3://%s/%s", self.source_bucket_name, object_key)
                response = self.s3_client.get_object(Bucket=self.source_bucket_name, Key=object_key)
                yield s3_filename, response['Body'].read()
            except ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchKey':
                    continue
                logger.error(f"S3 ClientError for {object_key}: {e}")
                yield s3_filename, None # Yield None to signal failure
            except Exception as e:
                logger.error(f"Unexpected error fetching {object_key}: {e}", exc_info=True)
                yield s3_filename, None # Yield None to signal failure

        if checked_count >= MAX_SEQUENTIAL_CHECKS:
            logger.warning(f"Reached maximum sequential file check limit ({MAX_SEQUENTIAL_CHECKS}).")

    def _iterate_local_images(self):
        """Generator that yields (image_identifier, image_bytes) from local directory."""
        if not os.path.isdir(self.local_image_source_dir):
            raise FileNotFoundError(f"Local source directory not found: {self.local_image_source_dir}")

        image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']
        for filename in os.listdir(self.local_image_source_dir):
            if not any(filename.lower().endswith(ext) for ext in image_extensions):
                continue

            self.cursor.execute("SELECT 1 FROM ocr_results WHERE image_identifier = ?", (filename,))
            if self.cursor.fetchone():
                logger.info(f"Image {filename} already processed. Skipping.")
                continue

            file_path = os.path.join(self.local_image_source_dir, filename)
            try:
                with open(file_path, 'rb') as f:
                    yield filename, f.read()
            except (IOError, FileNotFoundError) as e:
                logger.error(f"Error reading local file {file_path}: {e}")
                yield filename, None

    def run(self) -> Dict[str, Any]:
        """Main method to run the OCR processing job."""
        processed_count = 0
        failed_count = 0
        error_message = None

        try:
            if self.processing_mode == "S3":
                image_iterator = self._iterate_s3_images()
            elif self.processing_mode == "LOCAL":
                image_iterator = self._iterate_local_images()
            else:
                raise ValueError(f"Invalid PROCESSING_MODE: {self.processing_mode}")

            for identifier, image_bytes in image_iterator:
                if image_bytes is None:
                    failed_count += 1
                    continue

                if self._process_image(identifier, image_bytes, self.processing_mode):
                    processed_count += 1
                else:
                    failed_count += 1

        except Exception as e:
            error_message = f"A critical error occurred during processing: {e}"
            logger.error(error_message, exc_info=True)
        finally:
            self.close()

        logger.info("OCR process finished. Processed: %d, Failed: %d", processed_count, failed_count)
        return {"processed_files": processed_count, "failed_files": failed_count, "error": error_message}

    def close(self):
        """Closes database connection."""
        if self.db_conn:
            self.db_conn.close()
            logger.info("Database connection closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trigger OCR processing for images.")
    parser.add_argument(
        "start_file_number",  # Positional argument
        type=int,
        nargs='?',             # Makes it optional
        default=5000000,       # Default value if not provided by the user
        help="Optional: The starting file number for S3 sequential processing (default: 5000000). This is ignored if --resume is used."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If specified for S3 mode, the script will attempt to automatically resume from the last processed file number found in the database."
    )
    cli_args = parser.parse_args()

    processor = OcrProcessor(start_file_number=cli_args.start_file_number, resume=cli_args.resume)
    results = processor.run()

    final_output = {"message": "OCR processing completed.", "details": results}
    print(json.dumps(final_output, indent=2))

    if results.get("error"):
        print(f"\nProcessing encountered an error: {results['error']}", file=sys.stderr)
        sys.exit(1)
    elif results['failed_files'] > 0:
        print(f"\nProcessing failed for {results['failed_files']} files. Check logs for details.", file=sys.stderr)
        sys.exit(1)
