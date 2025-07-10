import asyncio
import logging
from typing import List, Union

from fastapi import APIRouter, Query, File, UploadFile, Depends, HTTPException

import src.models as models
from src.config import OCR_SOURCE_S3_PREFIX, FAISS_INDEX_PATH, KEY_MAP_PATH
from src.dependencies import get_current_api_key
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

async def _perform_search(
    query: Union[UploadFile, str],
    img_cnt: int,
) -> models.FindSimilarImagesResponse:
    """
    Helper to perform image or text search, format results, and handle errors.
    """
    errors: List[str] = []
    similar_images: List[models.ImageSimilarityInfo] = []

    if isinstance(query, UploadFile):
        input_name = query.filename or "N/A"
        log_query_identifier = input_name
    else:
        input_name = f"text_query: '{query}'"
        log_query_identifier = f"query '{query}'"

    try:
        if isinstance(query, UploadFile):
            contents = await query.read()
            query_features = image_similarity_service.extract_features(contents)
        else:  # str
            query_features = image_similarity_service.extract_text_features(query)

        search_results = image_similarity_service.search(query_features, top_k=img_cnt)

        for s3_key, distance in search_results:
            # For normalized vectors, cosine similarity = 1 - (distance^2 / 2)
            # This converts the L2 distance from Faiss back to a cosine similarity score [0, 1]
            similarity_score = 1 - (distance**2) / 2

            # Reconstruct the original S3 key from the grayscale key
            original_identifier = s3_key.split('/')[-1]
            key_for_response = f"{OCR_SOURCE_S3_PREFIX.rstrip('/')}/{original_identifier}"

            similar_images.append(
                models.ImageSimilarityInfo(
                    s3_image_key=key_for_response,
                    combined_similarity=similarity_score,
                    shape_similarity=similarity_score,  # Placeholder, as ResNet combines features
                    color_similarity=similarity_score   # Placeholder
                )
            )
    except Exception as e:
        logger.error(f"Error during similarity search for {log_query_identifier}: {e}", exc_info=True)
        errors.append(f"Could not process request: {str(e)}")

    return models.FindSimilarImagesResponse(
        uploaded_filename=input_name,
        similar_images=similar_images,
        errors=errors
    )


@router.post("/image-match", response_model=models.FindSimilarImagesResponse)
async def find_similar_images_in_s3(
    uploaded_file: UploadFile = File(..., description="Image (JPEG or PNG) to find matches for."),
    img_cnt: int = Query(10, ge=1, le=50, description="Number of similar images to return."),
    api_key: str = Depends(get_current_api_key)
):
    if not image_similarity_service.index:
        raise HTTPException(status_code=503, detail="Image similarity search is not available. Index not loaded.")
    return await _perform_search(query=uploaded_file, img_cnt=img_cnt)


@router.post("/bulk-image-match", response_model=List[models.FindSimilarImagesResponse])
async def bulk_find_similar_images_in_s3(
    uploaded_files: List[UploadFile] = File(..., description="List of up to 100 images."),
    img_cnt: int = Query(10, ge=1, le=50, description="Number of similar images to return per uploaded file."),
    api_key: str = Depends(get_current_api_key)
):
    if len(uploaded_files) > 100:
        raise HTTPException(status_code=413, detail="Max 100 images.")
    if not image_similarity_service.index:
        raise HTTPException(status_code=503, detail="Image similarity search is not available. Index not loaded.")

    # Process files concurrently
    tasks = [_perform_search(query=up_file, img_cnt=img_cnt) for up_file in uploaded_files]
    results = await asyncio.gather(*tasks)
    return results


@router.post("/text-to-image-search", response_model=models.FindSimilarImagesResponse)
async def find_similar_images_from_text(
    query_text: str = Query(..., min_length=3, max_length=100, description="Text description to find matching images for."),
    img_cnt: int = Query(10, ge=1, le=50, description="Number of similar images to return."),
    api_key: str = Depends(get_current_api_key)
):
    """
    Searches for images that match a given text description using CLIP.
    """
    if not image_similarity_service.index:
        raise HTTPException(status_code=503, detail="Image similarity search is not available. Index not loaded.")
    return await _perform_search(query=query_text, img_cnt=img_cnt)