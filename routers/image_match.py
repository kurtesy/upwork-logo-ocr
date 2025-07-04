import logging
import sqlite3
from typing import List, Tuple

import numpy as np
import cv2
from fastapi import APIRouter, Query, File, UploadFile, Depends, HTTPException

import src.models as models
from src.config import S3_CLIENT, OCR_SOURCE_S3_PREFIX, FAISS_INDEX_PATH, KEY_MAP_PATH
from src.database import get_db_connection
from src.dependencies import get_current_api_key
from services.ocr_service import extract_text_from_image_bytes_api
from src.utils import calculate_text_similarity
from services.image_similarity_service import ImageSimilarityService

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/ocr",
    tags=["OCR Match"],
)

# --- Initialize Image Similarity Service ---
# This service is loaded once at startup and shared across requests.
image_similarity_service = ImageSimilarityService()
image_similarity_service.load_index(FAISS_INDEX_PATH, KEY_MAP_PATH)


def _get_ocr_matching_identifiers(
    db_cursor: sqlite3.Cursor, query_text: str, text_similarity_threshold: float,
    max_results_cap: int = 20
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
            if len(matching_identifiers) >= max_results_cap: break
            if query_text in entry["extracted_text"]:
                matching_identifiers.append(entry["image_identifier"])
                continue
            similarity = calculate_text_similarity(query_text, entry["extracted_text"])
            if similarity >= text_similarity_threshold:
                matching_identifiers.append(entry["image_identifier"])
    except sqlite3.Error as e:
        errors.append(f"SQLite error in _get_ocr_matching_identifiers: {e}")
        logger.error(errors[-1], exc_info=True)
    return matching_identifiers, total_ocr_entries_in_db, errors

@router.post("/image-match", response_model=models.FindSimilarImagesResponse)
async def find_similar_images_in_s3(
    uploaded_file: UploadFile = File(..., description="Image (JPEG or PNG) to find matches for."),
    top_k: int = Query(10, ge=1, le=50, description="Number of similar images to return."),
    api_key: str = Depends(get_current_api_key)
):
    if not image_similarity_service.index:
        raise HTTPException(status_code=503, detail="Image similarity search is not available. Index not loaded.")

    uploaded_filename = uploaded_file.filename or "N/A"
    errors: List[str] = []
    similar_images: List[models.ImageSimilarityInfo] = []

    try:
        contents = await uploaded_file.read()
        # 1. Extract features from the uploaded image
        query_features = image_similarity_service.extract_features(contents)

        # 2. Search for the top_k most similar images
        search_results = image_similarity_service.search(query_features, top_k=top_k)

        # 3. Format the results for the response
        for s3_key, distance in search_results:
            # For normalized vectors, cosine similarity = 1 - (distance^2 / 2)
            # This converts the L2 distance from Faiss back to a cosine similarity score [0, 1]
            similarity_score = 1 - (distance**2) / 2

            # Reconstruct the original S3 key from the grayscale key
            original_identifier = s3_key.split('/')[-1]
            key_for_response = f"{OCR_SOURCE_S3_PREFIX.rstrip('/')}/{original_identifier}"

            similar_images.append(models.ImageSimilarityInfo(
                s3_image_key=key_for_response,
                combined_similarity=similarity_score,
                shape_similarity=similarity_score, # Placeholder, as ResNet combines features
                color_similarity=similarity_score  # Placeholder
            ))

    except Exception as e:
        logger.error(f"Error in find_similar_images_in_s3 for {uploaded_filename}: {e}", exc_info=True)
        errors.append(f"Could not process request: {str(e)}")

    return models.FindSimilarImagesResponse(uploaded_filename=uploaded_filename, similar_images=similar_images, errors=errors)

@router.post("/bulk-image-match", response_model=List[models.FindSimilarImagesResponse])
async def bulk_find_similar_images_in_s3(
    uploaded_files: List[UploadFile] = File(..., description="List of up to 100 images."),
    top_k: int = Query(10, ge=1, le=50, description="Number of similar images to return per uploaded file."),
    api_key: str = Depends(get_current_api_key)
):
    if len(uploaded_files) > 100: raise HTTPException(status_code=413, detail="Max 100 images.")
    results = []
    for up_file in uploaded_files:
        # Call the single-image endpoint with keyword arguments for correctness
        results.append(await find_similar_images_in_s3(
            uploaded_file=up_file,
            top_k=top_k,
            api_key=api_key
        ))
    return results

@router.post("/text-to-image-search", response_model=models.FindSimilarImagesResponse)
async def find_similar_images_from_text(
    query_text: str = Query(..., min_length=3, max_length=100, description="Text description to find matching images for."),
    top_k: int = Query(10, ge=1, le=50, description="Number of similar images to return."),
    api_key: str = Depends(get_current_api_key)
):
    """
    Searches for images that match a given text description using CLIP.
    """
    if not image_similarity_service.index:
        raise HTTPException(status_code=503, detail="Image similarity search is not available. Index not loaded.")

    errors: List[str] = []
    similar_images: List[models.ImageSimilarityInfo] = []

    try:
        # 1. Extract features from the text query using CLIP's text encoder
        query_features = image_similarity_service.extract_text_features(query_text)

        # 2. Search the Faiss index for the most similar image vectors
        search_results = image_similarity_service.search(query_features, top_k=top_k)

        # 3. Format the results for the response
        for s3_key, distance in search_results:
            # Convert L2 distance to cosine similarity
            similarity_score = 1 - (distance**2) / 2

            original_identifier = s3_key.split('/')[-1]
            key_for_response = f"{OCR_SOURCE_S3_PREFIX.rstrip('/')}/{original_identifier}"

            similar_images.append(models.ImageSimilarityInfo(
                s3_image_key=key_for_response,
                combined_similarity=similarity_score,
                shape_similarity=similarity_score,
                color_similarity=similarity_score
            ))
    except Exception as e:
        logger.error(f"Error in find_similar_images_from_text for query '{query_text}': {e}", exc_info=True)
        errors.append(f"Could not process request: {str(e)}")

    return models.FindSimilarImagesResponse(uploaded_filename=f"text_query: '{query_text}'", similar_images=similar_images, errors=errors)