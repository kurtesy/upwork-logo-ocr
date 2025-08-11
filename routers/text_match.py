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
    models.SimilarityLevel.MEDIUM: 0.4,
    models.SimilarityLevel.LOW: 0.1,
}

def _get_fuzzy_parts_for_token(token: str) -> set:
    """Generates a set of FTS query parts for a single token."""
    parts = set()
    if not token:
        return parts

    # 1. Add a prefix search for the whole token.
    parts.add(f'"{token}"*')

    # 2. Add bigrams for more fuzzy matching.
    if len(token) >= 2:
        bigrams = {token[i:i+2] for i in range(len(token) - 1)}
        parts.update(f'"{bg}"' for bg in bigrams)

    # 3. Add trigrams for more precise fuzzy matching.
    if len(token) >= 3:
        trigrams = {token[i:i+3] for i in range(len(token) - 2)}
        parts.update(f'"{tg}"' for tg in trigrams)
    
    return parts


def _generate_fuzzy_query(query_text: str) -> str:
    """
    Generates a more comprehensive fuzzy FTS5 query from a user's query text.
    It creates a query that searches for:
    1. Individual words from the query using prefix, bigram, and trigram matching (joined by AND).
    2. The entire query text with spaces removed, also using prefix, bigram, and trigram matching.
    These two search strategies are joined by OR to maximize recall.

    e.g., "Coca Cola" ->
    '((("coca*" OR ...) AND ("cola*" OR ...))) OR ("cocacola*" OR ...)'
    """
    lower_query = query_text.lower()
    words = [lower_query.replace(" ", "")]

    if not words:
        return ""

    # --- Strategy 1: Match individual words ---
    word_queries = []
    for token in words:
        parts = _get_fuzzy_parts_for_token(token)
        if parts:
            word_queries.append(f'({" OR ".join(sorted(list(parts)))})')

    individual_word_query = ""
    if word_queries:
        # Join word queries with AND to ensure all words are considered.
        individual_word_query = f'({" AND ".join(word_queries)})'

    # --- Strategy 2: Match the whole query as a single token (no spaces) ---
    # This helps find matches where words are concatenated, e.g., "CocaCola".
    combined_query = ""
    if len(words)> 1:
        combined_token = "".join(words)
        parts = _get_fuzzy_parts_for_token(combined_token)
        if parts:
            combined_query = f'({" OR ".join(sorted(list(parts)))})'

    # --- Combine strategies ---
    final_queries = []
    if individual_word_query:
        final_queries.append(individual_word_query)
    if combined_query:
        final_queries.append(combined_query)

    if not final_queries:
        return ""

    return " OR ".join(final_queries)

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
        # Step 1: Use FTS5 with a fuzzy query to get a list of candidate rowids.
        # The FTS MATCH operator is much faster than a full table scan.
        # We generate a fuzzy query to better handle typos and variations.
        fuzzy_query = _generate_fuzzy_query(query_text)
        if not fuzzy_query:
            return [], []

        # We fetch more candidates than needed to allow for re-ranking.
        fts_query = "SELECT rowid FROM ocr_results_fts WHERE ocr_results_fts MATCH ? ORDER BY rank LIMIT ?;"
        # Fetching a larger set of candidates for re-ranking improves accuracy.
        # 100x the final cap is a reasonable trade-off between performance and recall.
        candidate_limit = max_results_cap * 100
        db_cursor.execute(fts_query, (fuzzy_query, candidate_limit))
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
                    "score": similarity,
                    "extracted_text": row["extracted_text"]
                })
        
        # Sort and limit
        ranked_results.sort(key=lambda x: x['score'], reverse=True)
        final_results = ranked_results[:max_results_cap]

        # Step 4: Format the final results
        for entry in final_results:
            image_identifier = entry["image_identifier"]
            source_type = entry["source_type"]
            print(entry["extracted_text"], entry["score"])
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
        default=[models.SimilarityLevel.LOW],
        description="Similarity level(s) to consider a match. 'high' (0.8), 'medium' (0.4), 'low' (0.1)."
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