import asyncio
import logging
import httpx
from typing import List, Tuple
from io import BytesIO

from fastapi import APIRouter, Query, File, UploadFile, Depends, HTTPException, Response
import boto3
from botocore.exceptions import ClientError

import src.models as models
from src.config import (OCR_SOURCE_S3_PREFIX, FAISS_INDEX_PATH, KEY_MAP_PATH,
                        OCR_SOURCE_BUCKET_NAME)
from src.dependencies import get_current_api_key
from services.image_similarity_service import ImageSimilarityService

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/ocr",
    tags=["Image & Text Match"],
)

SIMILARITY_LEVEL_MAPPING = {
    models.SimilarityLevel.HIGH: 0.8,
    models.SimilarityLevel.MEDIUM: 0.5,
    models.SimilarityLevel.LOW: 0.3,
}

# --- Initialize Image Similarity Service ---
# This service is loaded once at startup and shared across requests.
image_similarity_service = ImageSimilarityService()
image_similarity_service.load_index(FAISS_INDEX_PATH, KEY_MAP_PATH)
s3_client = boto3.client("s3")

async def _find_similar_images_with_reranking(
    query_image_bytes: bytes,
    img_cnt: int,
    similarity_threshold: float
) -> Tuple[List[models.ImageSimilarityInfo], List[str]]:
    """
    Finds similar images by performing an initial search with CLIP features and
    then re-ranking the results using color and shape similarity.
    """    
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

        reranked_results = []

        # 3. Re-rank candidates based on combined similarity
        for s3_key, clip_distance in search_results:
            try:
                # Download candidate image from S3
                original_identifier = s3_key.split('/')[-1]
                source_s3_key = f"{OCR_SOURCE_S3_PREFIX.rstrip('/')}/{original_identifier}"
                
                response = s3_client.get_object(Bucket=OCR_SOURCE_BUCKET_NAME, Key=source_s3_key)
                candidate_image_bytes = response['Body'].read()

                # Extract features for the candidate image
                candidate_color_hist = image_similarity_service.extract_color_features(candidate_image_bytes)
                candidate_shape_contour = image_similarity_service.extract_shape_features(candidate_image_bytes)

                # Calculate individual similarity scores
                clip_similarity = 1 - (clip_distance**2) / 2
                color_similarity = image_similarity_service.compare_color_features(query_color_hist, candidate_color_hist)
                shape_similarity = image_similarity_service.compare_shape_features(query_shape_contour, candidate_shape_contour)

                # Combine scores with weighting. These weights can be tuned.
                combined_similarity = (0.6 * clip_similarity) + (0.2 * color_similarity) + (0.2 * shape_similarity)

                reranked_results.append(models.ImageSimilarityInfo(
                    s3_image_key=source_s3_key,
                    combined_similarity=combined_similarity,
                    shape_similarity=shape_similarity,
                    color_similarity=color_similarity
                ))

            except ClientError as e:
                logger.warning(f"Could not download/process candidate {source_s3_key} from S3: {e}")
            except Exception as e:
                logger.warning(f"Error processing candidate {s3_key}: {e}")

        # 4. Sort by the new combined similarity and filter by threshold
        reranked_results.sort(key=lambda x: x.combined_similarity, reverse=True)

        for result in reranked_results:
            if result.combined_similarity >= similarity_threshold:
                similar_images.append(result)
            if len(similar_images) >= img_cnt:
                break

    except Exception as e:
        logger.error(f"Error in similarity search with reranking: {e}", exc_info=True)
        errors.append(f"Could not process request: {str(e)}")

    return similar_images, errors


@router.post("/image-match", response_model=models.FindSimilarImagesResponse)
async def find_similar_images_in_s3(
    uploaded_file: UploadFile = File(..., description="Image (JPEG or PNG) to find matches for."),
    img_cnt: int = Query(100, ge=1, le=100, description="Number of similar images to return."),
    similarity_level: List[models.SimilarityLevel] = Query(
        default=[models.SimilarityLevel.HIGH],
        description="Similarity level(s) to consider a match. 'high' (0.8), 'medium' (0.5), 'low' (0.3)."
    ),
    api_key: str = Depends(get_current_api_key)
):
    if not image_similarity_service.index:
        raise HTTPException(status_code=503, detail="Image similarity search is not available. Index not loaded.")

    uploaded_filename = uploaded_file.filename or "N/A"
    similarity_thresholds = [SIMILARITY_LEVEL_MAPPING[level] for level in similarity_level]
    min_similarity_threshold = min(similarity_thresholds) if similarity_thresholds else 0.0


    contents = await uploaded_file.read()
    
    similar_images, errors = await _find_similar_images_with_reranking(
        query_image_bytes=contents,
        img_cnt=img_cnt,
        similarity_threshold=min_similarity_threshold
    )

    return models.FindSimilarImagesResponse(uploaded_filename=uploaded_filename, similar_images=similar_images, errors=errors)
    
async def _find_similar_for_url(image_url: str, img_cnt: int, similarity_threshold: float) -> models.FindSimilarImagesResponse:
    """
    Helper to find similar images for a single image URL.
    """
    errors: List[str] = []
    similar_images: List[models.ImageSimilarityInfo] = []

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url)
            response.raise_for_status()
            contents = response.content

        similar_images, search_errors = await _find_similar_images_with_reranking(
            query_image_bytes=contents,
            img_cnt=img_cnt,
            similarity_threshold=similarity_threshold
        )
        errors.extend(search_errors)

    except httpx.RequestError as e:
        err_msg = f"Failed to download image from URL {image_url}: {e}"
        logger.error(err_msg)
        errors.append(err_msg)
    except Exception as e:
        logger.error(f"Error processing URL {image_url}: {e}", exc_info=True)
        errors.append(f"Could not process request for {image_url}: {str(e)}")

    return models.FindSimilarImagesResponse(
        uploaded_filename=image_url,
        similar_images=similar_images,
        errors=errors
    )

@router.post("/bulk-image-match", response_model=List[models.FindSimilarImagesResponse])
async def bulk_find_similar_images_in_s3(
    uploaded_files: List[UploadFile] = File(..., description="List of up to 100 images."),
    img_cnt: int = Query(100, ge=1, le=100, description="Number of similar images to return per uploaded file."),
    similarity_level: List[models.SimilarityLevel] = Query(
        default=[models.SimilarityLevel.HIGH],
        description="Similarity level(s) to consider a match. 'high' (0.8), 'medium' (0.5), 'low' (0.3)."
    ), 
    api_key: str = Depends(get_current_api_key)
):
    if len(uploaded_files) > 100: raise HTTPException(status_code=413, detail="Max 100 images.")
    results = []
    for up_file in uploaded_files:
        # Call the single-image endpoint with keyword arguments for correctness
        results.append(await find_similar_images_in_s3(
            uploaded_file=up_file,
            img_cnt=img_cnt,
            similarity_level=similarity_level,
            api_key=api_key
        ))
    return results

@router.post("/image-url-match", response_model=models.FindSimilarImagesResponse, tags=["Image & Text Match"])
async def find_similar_images_by_url(
    image_url: str = Query(..., description="URL of an image (JPEG or PNG) to find matches for."),
    img_cnt: int = Query(100, ge=1, le=100, description="Number of similar images to return."),
    similarity_level: models.SimilarityLevel = Query(
        default=models.SimilarityLevel.HIGH,
        description="Similarity level to consider a match. 'high' (0.8), 'medium' (0.5), 'low' (0.3)."
    ),
    api_key: str = Depends(get_current_api_key)
):
    """
    Finds similar images to a given image URL.
    """
    if not image_similarity_service.index:
        raise HTTPException(status_code=503, detail="Image similarity search is not available. Index not loaded.")

    similarity_threshold = SIMILARITY_LEVEL_MAPPING[similarity_level]
    return await _find_similar_for_url(image_url=image_url, img_cnt=img_cnt, similarity_threshold=similarity_threshold)

@router.post("/bulk-image-url-match", response_model=List[models.FindSimilarImagesResponse], tags=["Image & Text Match"])
async def bulk_find_similar_images_by_url(
    request: models.BulkImageUrlRequest,
    img_cnt: int = Query(100, ge=1, le=100, description="Number of similar images to return per URL."),
    similarity_level: models.SimilarityLevel = Query(
        default=models.SimilarityLevel.HIGH,
        description="Similarity level to consider a match. 'high' (0.8), 'medium' (0.5), 'low' (0.3)."
    ),
    api_key: str = Depends(get_current_api_key)
):
    """
    Finds similar images for a list of up to 100 image URLs.
    """
    if not image_similarity_service.index:
        raise HTTPException(status_code=503, detail="Image similarity search is not available. Index not loaded.")

    if len(request.image_urls) > 100:
        raise HTTPException(status_code=413, detail="You can provide a maximum of 100 image URLs.")

    similarity_threshold = SIMILARITY_LEVEL_MAPPING[similarity_level]
    tasks = [_find_similar_for_url(url, img_cnt=img_cnt, similarity_threshold=similarity_threshold) for url in request.image_urls]
    results = await asyncio.gather(*tasks)
    return results

@router.get("/image/{image_name}",
            responses={
                200: {
                    "content": {"image/jpeg": {}, "image/png": {}},
                    "description": "The image from S3.",
                },
                404: {"description": "Image not found"},
            },
            tags=["Image & Text Match"])
async def get_s3_image(
    image_name: str,
    api_key: str = Depends(get_current_api_key)
):
    """
    Retrieves and displays an image from the S3 bucket by its name.
    """
    source_s3_key = f"{OCR_SOURCE_S3_PREFIX.rstrip('/')}/{image_name}"
    logger.info(f"Attempting to retrieve image from S3: {source_s3_key}")

    try:
        response = s3_client.get_object(Bucket=OCR_SOURCE_BUCKET_NAME, Key=source_s3_key)

        # Determine media type from S3 object metadata, default to octet-stream
        media_type = response.get("ContentType", "application/octet-stream")

        return Response(content=response['Body'].read(), media_type=media_type)

    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.warning(f"Image not found in S3: {source_s3_key}")
            raise HTTPException(status_code=404, detail="Image not found")
        else:
            logger.error(f"S3 ClientError when retrieving {source_s3_key}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error retrieving image from S3")