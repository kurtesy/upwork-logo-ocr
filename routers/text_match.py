import logging
import sqlite3
import asyncio
from typing import List, Tuple

from fastapi import APIRouter, Depends, Query, Body, Request

import src.models as models
from src.config import OCR_SOURCE_BUCKET_NAME, OCR_SOURCE_S3_PREFIX, AWS_DEFAULT_REGION
from src.database import get_db_connection
from src.dependencies import get_current_api_key
from src.utils import calculate_text_similarity

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/ocr/text",
    tags=["Text Match"],
)

SIMILARITY_LEVEL_MAPPING = {
    models.SimilarityLevel.HIGH: 0.8,
    models.SimilarityLevel.MEDIUM: 0.5,
    models.SimilarityLevel.LOW: 0.3,
}


def _execute_match_query(
    db_cursor: sqlite3.Cursor,
    query_text: str,
    similarity_threshold: float,
    max_results_cap: int = 100,
) -> Tuple[List[str], List[str]]:
    """
    Executes a single match query against the database. This version is optimized
    to use an FTS5 virtual table for a fast initial search, followed by a
    more precise similarity calculation in Python for re-ranking.

    NOTE: This function requires an FTS5 table named 'ocr_results_fts' for performance.
    If the table does not exist, it will raise an error.

    Returns a tuple of (matching_logo_urls, errors).
    """
    matching_logos: List[str] = []
    errors: List[str] = []
    try:
        # Step 1: Use FTS5 to get a list of candidate rowids.
        # The FTS MATCH operator is much faster than a full table scan.
        # We fetch more candidates than needed to allow for re-ranking.
        fts_query = "SELECT rowid FROM ocr_results_fts WHERE ocr_results_fts MATCH ? ORDER BY rank LIMIT ?;"
        candidate_limit = max_results_cap * 5  # Fetch 5x candidates for re-ranking
        db_cursor.execute(fts_query, (query_text, candidate_limit))
        candidate_rowids = [row[0] for row in db_cursor.fetchall()]

        if not candidate_rowids:
            return [], []

        # Step 2: Fetch the full data for candidate rows.
        placeholders = ','.join('?' for _ in candidate_rowids)
        sql = f"SELECT image_identifier, source_type, extracted_text FROM ocr_results WHERE rowid IN ({placeholders})"
        db_cursor.execute(sql, candidate_rowids)
        
        candidates = db_cursor.fetchall()
        
        # Step 3: Re-rank in Python using the precise similarity function
        ranked_results = []
        for row in candidates:
            similarity = calculate_text_similarity(query_text, row['extracted_text'])
            if similarity >= similarity_threshold:
                ranked_results.append({
                    "image_identifier": row["image_identifier"],
                    "source_type": row["source_type"],
                    "score": similarity
                })
        
        # Sort and limit
        ranked_results.sort(key=lambda x: x['score'], reverse=True)
        final_results = ranked_results[:max_results_cap]

        # Step 4: Format the final results
        for entry in final_results:
            image_identifier = entry["image_identifier"]
            source_type = entry["source_type"]
            if source_type == 'S3' and OCR_SOURCE_BUCKET_NAME and AWS_DEFAULT_REGION:
                key_prefix = OCR_SOURCE_S3_PREFIX.rstrip('/')
                object_key = f"{key_prefix}/{image_identifier}" if key_prefix else image_identifier
                s3_url = f"https://{OCR_SOURCE_BUCKET_NAME}.s3.{AWS_DEFAULT_REGION}.amazonaws.com/{object_key}"
                logo_url = s3_url
            else:
                logo_url = image_identifier
            matching_logos.append(logo_url)
    except sqlite3.Error as e:
        if "no such table" in str(e) and "ocr_results_fts" in str(e):
            err_msg = "FTS5 table 'ocr_results_fts' not found. Please run the database indexing script to create it for faster queries."
            logger.error(err_msg)
            errors.append(err_msg)
        else:
            err_msg = f"SQLite error in FTS match query: {e}"
            logger.error(err_msg, exc_info=True)
            errors.append(err_msg)
    return matching_logos, errors


@router.get("/text-match", response_model=models.LogoMatchResponse)
async def find_matching_logos(
    request: Request,
    query_text: str = Query(..., min_length=1, description="The text to search for in OCR results."),
    similarity_level: List[models.SimilarityLevel] = Query(
        default=[models.SimilarityLevel.HIGH],
        description="Similarity level(s) to consider a match. 'high' (0.8), 'medium' (0.5), 'low' (0.3)."
    ),
    max_results: int = Query(100, ge=1, le=500, description="Maximum number of matching logos to return."),
    api_key: str = Depends(get_current_api_key) 
):
    similarity_thresholds = [SIMILARITY_LEVEL_MAPPING[level] for level in similarity_level]
    min_similarity_threshold = min(similarity_thresholds) if similarity_thresholds else 0.0
    conn = get_db_connection()
    try:
        # The 'similarity' function is no longer registered on the connection,
        # as the optimized query now performs this logic in Python.
        cursor = conn.cursor()

        # Get total count of processable files for the response.
        cursor.execute("SELECT COUNT(*) FROM ocr_results")
        count_result = cursor.fetchone()
        processed_files = count_result[0] if count_result else 0

        matching_logos, errors = _execute_match_query(
            cursor, query_text, min_similarity_threshold, max_results
        )
    finally:
        conn.close()

    return models.LogoMatchResponse(
        query_text=query_text,
        similarity_level=similarity_level,
        similarity_threshold=min_similarity_threshold,
        matching_logos=matching_logos,
        processed_ocr_files=processed_files,
        errors=errors,
    )

@router.post("/bulk-text-match", response_model=models.BulkLogoMatchResponse)
async def bulk_find_matching_logos(
    request: Request,
    request_data: models.BulkLogoMatchRequest = Body(...),
    max_results_per_query: int = Query(100, ge=1, le=500, description="Maximum number of matching logos to return per query."),
    api_key: str = Depends(get_current_api_key)
):
    # Get total count once for all queries.
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM ocr_results")
        count_result = cursor.fetchone()
        processed_files_count = count_result[0] if count_result else 0
    finally:
        conn.close()

    def _run_single_query(query_item: models.BulkLogoMatchQuery):
        """Wrapper to run a single query in a thread, with its own db connection."""
        # Each thread needs its own DB connection.
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            similarity_thresholds = [SIMILARITY_LEVEL_MAPPING[level] for level in query_item.similarity_level]
            min_similarity_threshold = min(similarity_thresholds) if similarity_thresholds else 0.0
            
            logos, errors = _execute_match_query(
                cursor, query_item.query_text, min_similarity_threshold, max_results_per_query
            )
            return models.BulkLogoMatchResult(
                query_text=query_item.query_text,
                matching_logos=logos,
                processed_ocr_files=processed_files_count,
                errors=errors
            )
        finally:
            conn.close()

    # Run queries concurrently in a thread pool
    loop = asyncio.get_running_loop()
    tasks = [loop.run_in_executor(None, _run_single_query, item) for item in request_data.queries]
    results = await asyncio.gather(*tasks)

    return models.BulkLogoMatchResponse(results=results)