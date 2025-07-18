from enum import Enum
from typing import List
from pydantic import BaseModel
from fastapi import Body

class SimilarityLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class LogoMatchResponse(BaseModel):
    query_text: str
    similarity_level: SimilarityLevel
    similarity_threshold: float
    matching_logos: List[str]
    processed_ocr_files: int
    errors: List[str] = []

class ImageSimilarityInfo(BaseModel):
    s3_image_key: str
    color_similarity: float
    shape_similarity: float
    combined_similarity: float

class FindSimilarImagesResponse(BaseModel):
    uploaded_filename: str
    similar_images: List[ImageSimilarityInfo]
    errors: List[str] = []

class BulkLogoMatchQuery(BaseModel):
    query_text: str

class BulkLogoMatchRequest(BaseModel):
    queries: List[BulkLogoMatchQuery] = Body(..., max_items=100)
    similarity_level: SimilarityLevel = SimilarityLevel.HIGH

class BulkLogoMatchResult(BaseModel):
    query_text: str
    matching_logos: List[str]
    processed_ocr_files: int # This might be the total in DB, or specific to this query's context
    errors: List[str] = []

class BulkLogoMatchResponse(BaseModel):
    results: List[BulkLogoMatchResult]

class BulkImageUrlRequest(BaseModel):
    image_urls: List[str] = Body(..., max_items=100, description="A list of up to 100 image URLs to search for.")