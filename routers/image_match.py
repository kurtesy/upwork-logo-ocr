import asyncio
import logging
import httpx
from typing import List, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Query, File, UploadFile, Depends, HTTPException, Response, Request
import boto3
from botocore.exceptions import ClientError

import src.models as models
from src.config import (OCR_SOURCE_S3_PREFIX, FAISS_INDEX_PATH, KEY_MAP_PATH,
                        OCR_SOURCE_BUCKET_NAME)
from src.dependencies import get_current_api_key
from services.image_similarity_service import ImageSimilarityService

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/ocr/image",
    tags=["Image Match"],
)

SIMILARITY_LEVEL_MAPPING = {
    models.SimilarityLevel.HIGH: 0.8,
    models.SimilarityLevel.MEDIUM: 0.5,
    models.SimilarityLevel.LOW: 0.3,
}
THREAD_COUNT = 10

# --- Initialize Image Similarity Service ---
image_similarity_service = ImageSimilarityService()
s3_client = None

def initialize_image_similarity_service():
    global image_similarity_service, s3_client
    logger.info("Initializing Image Similarity Service...")
    image_similarity_service = ImageSimilarityService()
    try:
        image_similarity_service.load_index(FAISS_INDEX_PATH, KEY_MAP_PATH)
        s3_client = boto3.client("s3")
        logger.info("Image Similarity Service and S3 client initialized successfully.")
    except Exception as e:
        logger.error(f"Image Sim Service init failed: {e}", exc_info=True)
        # This exception will be caught by the lifespan manager and will
        # prevent the app from starting in a broken state.
        raise

def _find_similar_images_with_reranking_sync(
    query_image_bytes: bytes,
    img_cnt: int,
    similarity_threshold: float
) -> Tuple[List[models.ImageSimilarityInfo], List[str]]:
    """
    Finds similar images by performing an initial search with CLIP features and
    then re-ranking the results using color and shape similarity. This is a
    synchronous and potentially long-running function.
    """
    if not s3_client:
        # This should not happen if initialization is correct, but it's a safeguard.
        return [], ["S3 client is not initialized."]

    errors: List[str] = []    
    similar_images: List[models.ImageSimilarityInfo] = []

    try:
        # 1. Extract features from the query image
        query_clip_features = image_similarity_service.extract_features(query_image_bytes)
        query_color_hist = image_similarity_service.extract_color_features(query_image_bytes)
        query_shape_contour = image_similarity_service.extract_shape_features(query_image_bytes)

        # 2. Get initial candidates from Faiss (based on CLIP similarity)
        # Fetch more candidates for re-ranking (e.g., 5x the requested count, capped at 100)
        candidates_to_fetch = min(max(img_cnt * 5, 50), 100)
        search_results = image_similarity_service.search(query_clip_features, top_k=candidates_to_fetch)

        def _process_candidate(candidate_data: Tuple[str, float]):
            s3_key, clip_distance = candidate_data
            source_s3_key = ""
            try:
                # Download candidate image from S3
                original_identifier = s3_key.split('/')[-1]
                source_s3_key = f"{OCR_SOURCE_S3_PREFIX.rstrip('/')}/{original_identifier}"
                
                response = s3_client.get_object(Bucket=OCR_SOURCE_BUCKET_NAME, Key=source_s3_key) # type: ignore
                candidate_image_bytes = response['Body'].read()

                # Extract features for the candidate image
                candidate_color_hist = image_similarity_service.extract_color_features(candidate_image_bytes)
                candidate_shape_contour = image_similarity_service.extract_shape_features(candidate_image_bytes)

                # Calculate individual similarity scores
                clip_similarity = 1 - (clip_distance**2) / 2
                color_similarity = image_similarity_service.compare_color_features(query_color_hist, candidate_color_hist)
                shape_similarity = image_similarity_service.compare_shape_features(query_shape_contour, candidate_shape_contour)

                # Combine scores with weighting. These weights can be tuned.
                combined_similarity = (0.8 * clip_similarity) + (0.1 * color_similarity) + (0.1 * shape_similarity)

                return models.ImageSimilarityInfo(
                    s3_image_key=source_s3_key,
                    combined_similarity=combined_similarity,
                    shape_similarity=shape_similarity,
                    color_similarity=color_similarity,
                    clip_similarity=clip_similarity
                )
            except ClientError as e:
                log_key = source_s3_key if source_s3_key else s3_key
                logger.warning(f"S3 process fail for {log_key}: {e}")
            except Exception as e:
                logger.warning(f"Candidate process error for {s3_key}: {e}")
            return None

        # 3. Re-rank candidates based on combined similarity
        with ThreadPoolExecutor(max_workers=5) as executor:
            results_iterator = executor.map(_process_candidate, search_results)
            reranked_results = [result for result in results_iterator if result is not None]

        # 4. Sort by the new combined similarity and filter by threshold
        reranked_results.sort(key=lambda x: x.combined_similarity, reverse=True)

        for result in reranked_results:
            if result.combined_similarity >= similarity_threshold:
                similar_images.append(result)
            if len(similar_images) >= img_cnt:
                break

    except Exception as e:
        logger.error(f"Similarity search rerank error: {e}", exc_info=True)
        errors.append(f"Could not process request: {e}")

    return similar_images, errors

def _check_services_initialized():
    """Dependency to ensure that required services are available."""
    if not image_similarity_service or not image_similarity_service.index:
        raise HTTPException(status_code=503, detail="Image similarity search is not available. Index not loaded.")
    if not s3_client:
        raise HTTPException(status_code=503, detail="S3 service is not available.")

async def _process_item_for_similarity(
    item: Union[UploadFile, str],
    img_cnt: int,
    similarity_threshold: float,
) -> models.FindSimilarImagesResponse:
    """
    Fetches image bytes from an UploadFile or URL, finds similar images,
    and returns a response model.
    """
    errors: List[str] = []
    similar_images: List[models.ImageSimilarityInfo] = []
    identifier = "N/A"

    try:
        if isinstance(item, UploadFile):
            identifier = item.filename or "N/A"
            image_bytes = await item.read()
        else:  # It's a URL string
            identifier = item
            async with httpx.AsyncClient() as client:
                response = await client.get(item)
                response.raise_for_status()
                image_bytes = response.content

        similar_images, errors = await asyncio.to_thread(
            _find_similar_images_with_reranking_sync,
            query_image_bytes=image_bytes,
            img_cnt=img_cnt,
            similarity_threshold=similarity_threshold
        )
    except httpx.RequestError as e:
        err_msg = f"URL download failed: {identifier}: {e}"
        logger.error(err_msg)
        errors.append(err_msg)
    except Exception as e:
        logger.error(f"Processing error for {identifier}: {e}", exc_info=True)
        errors.append(f"Could not process request for {identifier}: {str(e)}")

    return models.FindSimilarImagesResponse(
        uploaded_filename=identifier, similar_images=similar_images, errors=errors
    )

async def _run_bulk_processing(items, process_coro):
    """Generic helper to run a list of items through a processing coroutine with a semaphore."""
    semaphore = asyncio.Semaphore(THREAD_COUNT)
    async def process_with_semaphore(item):
        async with semaphore:
            return await process_coro(item)
    tasks = [process_with_semaphore(item) for item in items]
    return await asyncio.gather(*tasks)

@router.post("/image-match", response_model=models.FindSimilarImagesResponse)
async def find_similar_images_in_s3(
    request: Request,
    uploaded_file: UploadFile = File(..., description="Image (JPEG or PNG) to find matches for."),
    img_cnt: int = Query(100, ge=1, le=100, description="Number of similar images to return."),
    similarity_level: List[models.SimilarityLevel] = Query(
        default=[models.SimilarityLevel.HIGH],
        description="Similarity level(s) to consider a match. 'high' (0.8), 'medium' (0.5), 'low' (0.3)."
    ),
    api_key: str = Depends(get_current_api_key),
    _service_check: None = Depends(_check_services_initialized),
):
    min_similarity_threshold = min((SIMILARITY_LEVEL_MAPPING[level] for level in similarity_level), default=0.0)
    return await _process_item_for_similarity(uploaded_file, img_cnt, min_similarity_threshold)

@router.post("/image-url-match", response_model=models.FindSimilarImagesResponse)
async def find_similar_images_by_url(
    request: Request,
    image_url: str = Query(..., description="URL of an image (JPEG or PNG) to find matches for."),
    img_cnt: int = Query(100, ge=1, le=100, description="Number of similar images to return."),
    similarity_level: models.SimilarityLevel = Query(
        default=models.SimilarityLevel.HIGH,
        description="Similarity level to consider a match. 'high' (0.8), 'medium' (0.5), 'low' (0.3)."
    ),
    api_key: str = Depends(get_current_api_key),
    _service_check: None = Depends(_check_services_initialized),
):
    similarity_threshold = SIMILARITY_LEVEL_MAPPING[similarity_level]
    return await _process_item_for_similarity(image_url, img_cnt, similarity_threshold)

@router.post("/bulk-image-match", response_model=List[models.FindSimilarImagesResponse])
async def bulk_find_similar_images_in_s3(
    request: Request,
    uploaded_files: List[UploadFile] = File(..., description="List of up to 100 images."),
    img_cnt: int = Query(100, ge=1, le=100, description="Number of similar images to return per uploaded file."),
    similarity_level: List[models.SimilarityLevel] = Query(
        default=[models.SimilarityLevel.HIGH],
        description="Similarity level(s) to consider a match. 'high' (0.8), 'medium' (0.5), 'low' (0.3)."
    ), 
    api_key: str = Depends(get_current_api_key),
    _service_check: None = Depends(_check_services_initialized),
):
    if len(uploaded_files) > 100:
        raise HTTPException(status_code=413, detail="Max 100 images.")
    
    min_similarity_threshold = min((SIMILARITY_LEVEL_MAPPING[level] for level in similarity_level), default=0.0)

    async def process_file(up_file: UploadFile):
        return await _process_item_for_similarity(up_file, img_cnt, min_similarity_threshold)

    return await _run_bulk_processing(uploaded_files, process_file)
    
@router.post("/bulk-image-url-match", response_model=List[models.FindSimilarImagesResponse])
async def bulk_find_similar_images_by_url(
    request_obj: Request,
    request: models.BulkImageUrlRequest,
    img_cnt: int = Query(100, ge=1, le=100, description="Number of similar images to return per URL."),
    similarity_level: models.SimilarityLevel = Query(
        default=models.SimilarityLevel.HIGH,
        description="Similarity level to consider a match. 'high' (0.8), 'medium' (0.5), 'low' (0.3)."
    ),
    api_key: str = Depends(get_current_api_key),
    _service_check: None = Depends(_check_services_initialized),
):
    if len(request.image_urls) > 100:
        raise HTTPException(status_code=413, detail="You can provide a maximum of 100 image URLs.")

    similarity_threshold = SIMILARITY_LEVEL_MAPPING[similarity_level]

    async def process_url(url: str):
        return await _process_item_for_similarity(url, img_cnt, similarity_threshold)

    return await _run_bulk_processing(request.image_urls, process_url)

@router.get("/image/{image_name}",
            responses={
                200: {
                    "content": {"image/jpeg": {}, "image/png": {}},
                    "description": "The image from S3.",
                },
                404: {"description": "Image not found"},
            },
            )
async def get_s3_image(
    request: Request,
     image_name: str,
    api_key: str = Depends(get_current_api_key),
    _service_check: None = Depends(_check_services_initialized)
):
    """Retrieves and displays an image from the S3 bucket by its name."""

    source_s3_key = f"{OCR_SOURCE_S3_PREFIX.rstrip('/')}/{image_name}"
    logger.info(f"S3 get: {source_s3_key}")

    try:
        response = s3_client.get_object(Bucket=OCR_SOURCE_BUCKET_NAME, Key=source_s3_key) # type: ignore

        # Determine media type from S3 object metadata, default to octet-stream
        media_type = response.get("ContentType", "application/octet-stream")

        return Response(content=response['Body'].read(), media_type=media_type)

    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.warning(f"S3 not found: {source_s3_key}")
            raise HTTPException(status_code=404, detail="Image not found")
        else:
            logger.error(f"S3 ClientError for {source_s3_key}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error retrieving image from S3")