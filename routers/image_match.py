import logging
import sqlite3
from typing import List, Tuple
import os

import numpy as np
import cv2
from fastapi import APIRouter, Query, File, UploadFile, Depends, HTTPException
from botocore.exceptions import ClientError

from .. import models
from ..config import S3_CLIENT, GRAYSCALE_BUCKET_NAME, GRAYSCALE_S3_PREFIX
from ..database import get_db_connection
from ..dependencies import get_current_api_key
from ..services.ocr_service import extract_text_from_image_bytes_api
from ..utils import calculate_image_similarity, calculate_text_similarity

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/ocr",
    tags=["OCR Match"],
)

def _get_ocr_matching_identifiers(
    db_cursor: sqlite3.Cursor, query_text: str, text_similarity_threshold: float
) -> Tuple[List[str], int, List[str]]:
    logger.debug(f"DB Helper: Finding matching identifiers for query: '{query_text}', threshold: {text_similarity_threshold}")
    matching_identifiers: List[str] = []
    errors: List[str] = []
    total_ocr_entries_in_db = 0
    try:
        db_cursor.execute("SELECT COUNT(*) FROM ocr_results")
        count_result = db_cursor.fetchone()
        if count_result: total_ocr_entries_in_db = count_result[0]

        db_cursor.execute("SELECT image_identifier, extracted_text FROM ocr_results WHERE extracted_text IS NOT NULL AND extracted_text != ''")
        for entry in db_cursor.fetchall():
            similarity = calculate_text_similarity(query_text, entry["extracted_text"])
            if similarity >= text_similarity_threshold:
                matching_identifiers.append(entry["image_identifier"])
    except sqlite3.Error as e:
        errors.append(f"SQLite error in _get_ocr_matching_identifiers: {e}")
        logger.error(errors[-1], exc_info=True)
    return matching_identifiers, total_ocr_entries_in_db, errors

def _find_similar_images_from_s3_candidates(
    uploaded_image_np: np.ndarray, uploaded_filename: str, candidate_s3_keys: List[str],
    image_similarity_threshold: float, max_results_cap: int = 20
) -> Tuple[List[models.ImageSimilarityInfo], List[str]]:
    if not calculate_image_similarity:
        raise HTTPException(status_code=501, detail="Image similarity utility not available.")
    if not S3_CLIENT:
        return [], ["S3 client not initialized. Check AWS configuration."]

    errors: List[str] = []
    similar_images_found: List[models.ImageSimilarityInfo] = []

    for s3_key in candidate_s3_keys:
        if len(similar_images_found) >= max_results_cap: break
        try:
            s3_image_bytes = None
            conn_cache = get_db_connection()
            try:
                cursor_cache = conn_cache.cursor()
                cursor_cache.execute("SELECT image_bytes FROM s3_image_cache WHERE s3_key = ?", (s3_key,))
                cached = cursor_cache.fetchone()
                if cached: s3_image_bytes = cached["image_bytes"]
                else:
                    s3_response = S3_CLIENT.get_object(Bucket=GRAYSCALE_BUCKET_NAME, Key=s3_key)
                    s3_image_bytes = s3_response['Body'].read()
                    cursor_cache.execute("INSERT OR REPLACE INTO s3_image_cache (s3_key, image_bytes, timestamp) VALUES (?, ?, CURRENT_TIMESTAMP)", (s3_key, s3_image_bytes))
                    conn_cache.commit()
            except sqlite3.Error as db_err:
                logger.error(f"SQLite cache error for {s3_key}: {db_err}", exc_info=True)
                if s3_image_bytes is None: # Fallback if cache failed before fetch
                    s3_response = S3_CLIENT.get_object(Bucket=GRAYSCALE_BUCKET_NAME, Key=s3_key)
                    s3_image_bytes = s3_response['Body'].read()
            finally:
                if conn_cache: conn_cache.close()

            if not s3_image_bytes:
                errors.append(f"Could not obtain image bytes for S3 key: {s3_key}"); continue

            s3_image_np = cv2.imdecode(np.frombuffer(s3_image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
            if s3_image_np is None or s3_image_np.size == 0:
                errors.append(f"Could not decode S3 image: {s3_key}"); continue

            similarity_scores = calculate_image_similarity(uploaded_image_np, s3_image_np)
            if similarity_scores and similarity_scores["combined_similarity"] >= image_similarity_threshold:
                key_for_response = s3_key.replace(GRAYSCALE_S3_PREFIX, "", 1).replace("grayscale/", "original/", 1) # Adjust based on actual prefix structure
                # A more robust way to get original key might be needed if prefixes are complex
                original_identifier = os.path.basename(s3_key) # Assuming s3_key is like 'images/grayscale/filename.jpeg'
                key_for_response = f"{OCR_SOURCE_S3_PREFIX.rstrip('/')}/{original_identifier}".replace(GRAYSCALE_S3_PREFIX, OCR_SOURCE_S3_PREFIX,1)

                similar_images_found.append(models.ImageSimilarityInfo(s3_image_key=key_for_response, **similarity_scores))
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == 'NoSuchKey': logger.info(f"S3 object not found: {s3_key}")
            else: errors.append(f"S3 ClientError for {s3_key}: {e}"); logger.error(errors[-1])
        except Exception as e:
            errors.append(f"Unexpected error processing S3 image {s3_key}: {e}"); logger.error(errors[-1], exc_info=True)
    return similar_images_found, errors

@router.post("/image-match", response_model=models.FindSimilarImagesResponse)
async def find_similar_images_in_s3(
    uploaded_file: UploadFile = File(..., description="Grayscale image (JPEG or PNG) to compare."),
    image_similarity_threshold: float = Query(0.6, ge=0.0, le=1.0, alias="imageSimilarityThreshold"),
    text_match_similarity_threshold: float = Query(0.5, ge=0.0, le=1.0, alias="textMatchSimilarityThreshold"),
    api_key: str = Depends(get_current_api_key)
):
    if not S3_CLIENT: raise HTTPException(status_code=503, detail="S3 client not available.")
    uploaded_filename = uploaded_file.filename or "N/A"
    errors: List[str] = []
    similar_images: List[models.ImageSimilarityInfo] = []
    conn = None
    try:
        contents = await uploaded_file.read()
        query_text_from_upload = extract_text_from_image_bytes_api(contents)
        candidate_s3_keys: List[str] = []

        if query_text_from_upload:
            conn = get_db_connection()
            matched_ids, _, db_errs = _get_ocr_matching_identifiers(
                conn.cursor(), query_text_from_upload, text_match_similarity_threshold
            )
            errors.extend(db_errs)
            for identifier in matched_ids:
                candidate_s3_keys.append(f"{GRAYSCALE_S3_PREFIX.rstrip('/')}/{identifier}")

        uploaded_image_np = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_UNCHANGED)
        if uploaded_image_np is None or uploaded_image_np.size == 0:
            raise HTTPException(status_code=400, detail="Invalid image format for uploaded file.")

        if candidate_s3_keys:
            s3_sim_imgs, s3_errs = _find_similar_images_from_s3_candidates(
                uploaded_image_np, uploaded_filename, candidate_s3_keys, image_similarity_threshold
            )
            similar_images.extend(s3_sim_imgs); errors.extend(s3_errs)
    except Exception as e:
        logger.error(f"Error in find_similar_images_in_s3 for {uploaded_filename}: {e}", exc_info=True)
        errors.append(f"Could not process request: {str(e)}") # Avoid raising HTTPException here to return partial results/errors
    finally:
        if conn: conn.close()
    return models.FindSimilarImagesResponse(uploaded_filename=uploaded_filename, similar_images=similar_images, errors=errors)

@router.post("/bulk-image-match", response_model=List[models.FindSimilarImagesResponse])
async def bulk_find_similar_images_in_s3(
    uploaded_files: List[UploadFile] = File(..., description="List of up to 100 images."),
    image_similarity_threshold: float = Query(0.6, ge=0.0, le=1.0, alias="imageSimilarityThreshold"),
    text_match_similarity_threshold: float = Query(0.5, ge=0.0, le=1.0, alias="textMatchSimilarityThreshold"),
    api_key: str = Depends(get_current_api_key)
):
    if len(uploaded_files) > 100: raise HTTPException(status_code=413, detail="Max 100 images.")
    results = []
    for up_file in uploaded_files:
        results.append(await find_similar_images_in_s3(
            up_file, image_similarity_threshold, text_match_similarity_threshold, api_key # api_key passed for consistency, though Depends handles it
        ))
    return results