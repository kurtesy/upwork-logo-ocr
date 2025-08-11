# AI-Powered Image OCR and Similarity Matching Service

## Overview

This service provides a high-performance API for Optical Character Recognition (OCR) and advanced AI-powered image similarity matching. It leverages state-of-the-art machine learning models and algorithms to deliver accurate and efficient results.

The core of the image matching is a sophisticated AI pipeline that combines deep learning embeddings with traditional computer vision techniques for superior performance.

## Key Features

- **AI-Powered Image Matching**: Utilizes OpenAI's **CLIP** model to generate rich, semantic vector embeddings for images. This allows the system to understand the content and context of an image, not just its pixels.
- **High-Speed Search**: Employs **Faiss** from Meta AI for highly efficient similarity search on millions of image vectors.
- **Advanced Re-ranking Algorithm**: Initial search candidates from Faiss are re-ranked using a combination of:
    - **CLIP Similarity**: The primary semantic similarity score.
    - **Color Similarity**: Analysis of color histograms.
    - **Shape Similarity**: Comparison of image contours.
- **Fuzzy Text Search**: For text-based logo matching, the service uses a custom-built fuzzy search algorithm on top of a SQLite FTS5 index. This provides robust matching even with OCR inaccuracies or text variations.
- **OCR Processing**: Extracts text from images using EasyOCR.
- **Bulk Processing**: Offers bulk endpoints for both text and image matching.
- **API Documentation**: Interactive API documentation available via Swagger UI and ReDoc.

## Image Matching AI Pipeline

The image matching process follows a multi-stage pipeline to ensure both speed and accuracy:

1.  **Feature Extraction (CLIP)**: When an image is uploaded, its semantic features are extracted using the CLIP model, generating a dense vector embedding.
2.  **Candidate Retrieval (Faiss)**: The query vector is used to search the Faiss index, which contains the pre-computed vectors of the entire image database. This step rapidly retrieves a set of the most likely candidates.
3.  **Re-ranking**: The candidates from Faiss are then re-ranked using a weighted combination of CLIP similarity, color similarity, and shape similarity. This refines the search results, ensuring that the top matches are not only semantically similar but also visually alike in terms of color and shape.

## API Endpoints

The API server runs by default on `http://127.0.0.1:8000`.

### Authentication

All endpoints require an API key to be provided in the `X-API-KEY` request header.

### Health Check

- **`GET /`**: Provides a simple health check message.

### Text Matching

- **`GET /ocr/text/text-match`**: Find logos by OCR text similarity using fuzzy search.
- **`POST /ocr/text/bulk-text-match`**: Bulk find logos by OCR text similarity.

### Image Matching

- **`POST /ocr/image/image-match`**: Find similar images by uploading an image. The image is processed through the AI pipeline (CLIP -> Faiss -> Re-ranking).
- **`POST /ocr/image/image-url-match`**: Find similar images from an image URL.
- **`POST /ocr/image/bulk-image-match`**: Bulk find similar images by uploading multiple images.
- **`POST /ocr/image/bulk-image-url-match`**: Bulk find similar images from a list of image URLs.
- **`GET /ocr/image/image/{image_name}`**: Retrieve an image from the S3 bucket.

## Setup and Running

### Prerequisites

- Python 3.8+
- Pip
- Git
- An AWS account with S3 credentials configured.

### Environment Variables

Create a `.env` file in the project root with the necessary environment variables. See `src/config.py` for a full list of required variables.

### Installation

1.  Clone the repository.
2.  Create and activate a virtual environment.
3.  Install dependencies: `pip install -r requirements.txt`

### Building the Image Index

Before you can use the image matching features, you need to build the Faiss index from your images in S3.

```bash
python scripts/build_image_index.py
```

This script will:
1.  Download images from your S3 bucket.
2.  Use the CLIP model to extract feature vectors.
3.  Build a Faiss index and save it to disk.

### Running the API Server

```bash
python main.py
```

The server will start on `http://0.0.0.0:8000`.

## API Documentation

- **Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc**: `http://127.0.0.1:8000/redoc`