import logging
import sqlite3
from typing import List, Tuple

from fastapi import APIRouter, Depends, Query

import src.models as models
from src.config import OCR_SOURCE_BUCKET_NAME, OCR_SOURCE_S3_PREFIX, AWS_DEFAULT_REGION
from src.database import get_db_connection
from src.dependencies import get_current_api_key
from src.utils import calculate_text_similarity

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/ocr",
    tags=["OCR Match"],
)

SIMILARITY_LEVEL_MAPPING = {
    models.SimilarityLevel.HIGH: 0.8,
    models.SimilarityLevel.MEDIUM: 0.5,
    models.SimilarityLevel.LOW: 0.3,
}

def _get_ocr_text_matches(
    db_cursor: sqlite3.Cursor,
    query_text: str,
    similarity_threshold: float,
    max_results_cap: int = 10
) -> Tuple[List[str], int, List[str]]:
    """
    Helper function to find OCR text matches for a single query.
    Returns: (matching_logos, processed_ocr_files_count, errors)
    """
    logger.debug(f"Helper: Finding matches for query: '{query_text}', threshold: {similarity_threshold}")
    matching_logos: List[str] = []
    errors: List[str] = []
    processed_ocr_files_count = 0
    ocr_entries: List[sqlite3.Row] = []

    try:
        cursor = db_cursor # Assumes connection is managed by the caller for bulk, or created for single
        cursor.execute("SELECT COUNT(*) FROM ocr_results")
        count_result = cursor.fetchone()
        if count_result:
            processed_ocr_files_count = count_result[0]

        cursor.execute("SELECT image_identifier, extracted_text, source_type FROM ocr_results")
        ocr_entries = cursor.fetchall()

        for entry in ocr_entries:
            image_identifier = entry["image_identifier"]
            ocr_content = entry["extracted_text"] or ""
            source_type = entry["source_type"]
            similarity = calculate_text_similarity(query_text, ocr_content)

            if similarity >= similarity_threshold:
                if source_type == 'S3' and OCR_SOURCE_BUCKET_NAME and AWS_DEFAULT_REGION:
                    key_prefix = OCR_SOURCE_S3_PREFIX.rstrip('/')
                    object_key = f"{key_prefix}/{image_identifier}" if key_prefix else image_identifier
                    s3_url = f"https://{OCR_SOURCE_BUCKET_NAME}.s3.{AWS_DEFAULT_REGION}.amazonaws.com/{object_key}"
                    matching_logos.append(s3_url)
                else:
                    matching_logos.append(image_identifier)
            if len(matching_logos) > max_results_cap:
                break
    except sqlite3.Error as e:
        err_msg = f"SQLite database error in _get_ocr_text_matches: {e}"
        logger.error(err_msg, exc_info=True)
        errors.append(err_msg)
    except Exception as e:
        err_msg = f"An unexpected error occurred in _get_ocr_text_matches: {e}"
        logger.error(err_msg, exc_info=True)
        errors.append(err_msg)
    return matching_logos, processed_ocr_files_count, errors

@router.get("/text-match", response_model=models.LogoMatchResponse)
async def find_matching_logos(
    query_text: str = Query(..., min_length=1, description="The text to search for in OCR results."),
    similarity_level: models.SimilarityLevel = Query(
        default=models.SimilarityLevel.HIGH,
        description="Similarity level to consider a match. 'high' (0.9), 'medium' (0.7), 'low' (0.5)."
    ),
    api_key: str = Depends(get_current_api_key)
):
    similarity_threshold = SIMILARITY_LEVEL_MAPPING[similarity_level]
    conn = get_db_connection()
    try:
        matching_logos, processed_files, errors = _get_ocr_text_matches(
            db_cursor=conn.cursor(), query_text=query_text, similarity_threshold=similarity_threshold
        )
    finally:
        conn.close()

    return models.LogoMatchResponse(
        query_text=query_text,
        similarity_level=similarity_level,
        similarity_threshold=similarity_threshold,
        matching_logos=matching_logos,
        processed_ocr_files=processed_files,
        errors=errors,
    )

@router.post("/bulk-text-match", response_model=models.BulkLogoMatchResponse)
async def bulk_find_matching_logos(
    request_data: models.BulkLogoMatchRequest, api_key: str = Depends(get_current_api_key)
):
    results: List[models.BulkLogoMatchResult] = []
    similarity_threshold = SIMILARITY_LEVEL_MAPPING[request_data.similarity_level]
    conn = get_db_connection()
    try:
        for query_item in request_data.queries:
            logos, files, errors = _get_ocr_text_matches(
                conn.cursor(), query_item.query_text, similarity_threshold
            )
            results.append(models.BulkLogoMatchResult(
                query_text=query_item.query_text, matching_logos=logos,
                processed_ocr_files=files, errors=errors
            ))
    finally:
        conn.close()
    return models.BulkLogoMatchResponse(results=results)