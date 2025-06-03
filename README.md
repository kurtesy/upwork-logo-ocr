# Image OCR and Similarity Matching Service

## Overview

This service provides an API for performing Optical Character Recognition (OCR) on images and finding similarities between images based on their visual features (color and shape).

The service also includes a standalone script to trigger bulk OCR processing.

## Features

- **OCR Processing**: Extracts text from images using EasyOCR (or Pytesseract as a fallback).
- **Image Source**: Supports processing images from local directories or AWS S3 buckets.
- **Grayscale Conversion**: Converts images to grayscale before OCR and for similarity comparison, potentially improving accuracy and consistency. Grayscale versions can be saved locally or to S3.
- **SQLite Database**: Stores extracted OCR text for efficient querying.
- **Text-Based Logo Matching**:
  - Find logos by text similarity against stored OCR results.
  - Bulk endpoint for matching multiple text queries.
- **Image-Based Similarity Matching**:
  - Compare an uploaded grayscale image against a collection of grayscale images in an S3 bucket.
  - Similarity is determined by color histograms and shape (Hu Moments).
  - Bulk endpoint for matching multiple uploaded images.
- **API Documentation**: Interactive API documentation available via Swagger UI and ReDoc.

## API Endpoints

The API server runs by default on `http://127.0.0.1:8000`.

### Authentication

All endpoints, except for the root health check (`/`), require API key authentication.
The API key must be provided in the `X-API-KEY` request header.
Refer to the "Environment Variables" section for setting up the `API_KEY`.

### Health Check

- **`GET /`**
  - **Summary**: Service Health Check.
  - **Description**: Provides a simple health check message and the status of the SQLite database.
  - **Response**:
    ```json
    {
      "message": "Image OCR Service is running.",
      "database_status": "SQLite DB at ocr_results.db found"
    }
    ```

### OCR Text Matching

- **`GET /ocr/text-match`**

  - **Summary**: Find logos by OCR text similarity.
  - **Description**: Searches through processed OCR results in the SQLite database to find logos whose extracted text matches the provided query text with at least the given similarity threshold.
  - **Query Parameters**:
    - `query_text` (string, required): The text to search for.
    - `similarity_threshold` (float, optional, default: 0.7): Minimum similarity ratio (0.0 to 1.0).
  - **Response (`LogoMatchResponse`)**:
    ```json
    {
      "query_text": "example",
      "similarity_threshold": 0.7,
      "matching_logos": ["image1.jpg.txt", "image2.png.txt"],
      "processed_ocr_files": 150,
      "errors": []
    }
    ```

- **`POST /ocr/bulk-text-match`**
  - **Summary**: Bulk find logos by OCR text similarity (max 100 queries).
  - **Description**: Processes a list of up to 100 text queries to find matching logos.
  - **Request Body (`BulkLogoMatchRequest`)**:
    ```json
    {
      "queries": [
        { "query_text": "logo one" },
        { "query_text": "another brand" }
      ],
      "similarity_threshold": 0.65
    }
    ```
  - **Response (`BulkLogoMatchResponse`)**:
    ```json
    {
      "results": [
        {
          "query_text": "logo one",
          "matching_logos": ["logo_one_variant1.jpg"],
          "processed_ocr_files": 150,
          "errors": []
        },
        {
          "query_text": "another brand",
          "matching_logos": [],
          "processed_ocr_files": 150,
          "errors": ["Some specific error for this query if any"]
        }
      ]
    }
    ```

### Image Similarity Matching

- **`POST /ocr/image-match`**

  - **Summary**: Compare uploaded image with S3 grayscale images.
  - **Description**: Upload a grayscale image and compare it against a collection of grayscale images stored in an S3 bucket. Returns a list of S3 images that are similar based on color and shape analysis.
  - **Query Parameters**:
    - `similarity_threshold` (float, optional, default: 0.6): Minimum combined similarity score.
  - **Request Body**: Form data with `uploaded_file` (file part).
  - **Response (`FindSimilarImagesResponse`)**:
    ```json
    {
      "uploaded_filename": "my_test_image.jpg",
      "similar_images": [
        {
          "s3_image_key": "images/grayscale/similar_image1.jpeg",
          "color_similarity": 0.85,
          "shape_similarity": 0.75,
          "combined_similarity": 0.8
        }
      ],
      "errors": []
    }
    ```

- **`POST /ocr/bulk-image-match`**
  - **Summary**: Bulk compare uploaded images with S3 (max 100 images).
  - **Description**: Upload a list of up to 100 grayscale images and compare each against images in S3.
  - **Query Parameters**:
    - `similarity_threshold` (float, optional, default: 0.6): Minimum combined similarity score.
  - **Request Body**: Form data with `uploaded_files` (multiple file parts).
  - **Response**: A list of `FindSimilarImagesResponse` objects, one for each uploaded image.
    ```json
    [
      {
        "uploaded_filename": "image_a.jpg",
        "similar_images": [
          /* ... */
        ],
        "errors": []
      },
      {
        "uploaded_filename": "image_b.png",
        "similar_images": [
          /* ... */
        ],
        "errors": [
          "Could not decode S3 image: images/grayscale/problem_image.jpeg"
        ]
      }
    ]
    ```

## API Documentation (Swagger UI / ReDoc)

Once the API server is running, interactive API documentation is available at:

- **Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc**: `http://127.0.0.1:8000/redoc`

## Setup and Running the Service

### Prerequisites

- Python 3.8+
- Pip
- Tesseract OCR Engine (and language packs, e.g., `eng`, `hin`)
- Git

### Environment Variables

Create a `.env` file in the project root directory with the following variables (adjust as needed):

```env
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=your_aws_region

SOURCE_BUCKET_NAME=your_source_s3_bucket_for_ocr
SOURCE_PREFIX=images/original/
DESTINATION_BUCKET_NAME=your_destination_s3_bucket # Can be same as source
GRAYSCALE_DESTINATION_PREFIX=images/grayscale/
# VECTOR_DESTINATION_PREFIX=images/vectors/ # If you implement vectorization

PROCESSING_MODE=S3 # or LOCAL
LOCAL_IMAGE_SOURCE_DIR=./local_images/
LOCAL_GRAYSCALE_DESTINATION_DIR=./local_images_grayscale/
SQLITE_DB_PATH=ocr_results.db

TESSERACT_CMD=/usr/bin/tesseract # Optional: Path to Tesseract executable if not in PATH

# For API server (main.py)
GRAYSCALE_BUCKET_NAME=your_s3_bucket_with_grayscale_images_for_matching
API_KEY=your_secret_api_key_here # Add your desired API key
GRAYSCALE_S3_PREFIX=images/grayscale/
HOST=0.0.0.0
PORT=8000
```

### Installation

1.  Clone the repository:
    ```bash
    git clone <your_repository_url>
    cd upwork-logo-ocr
    ```
2.  Create and activate a virtual environment (recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Install Tesseract OCR:
    - On Ubuntu/Debian: `sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin`
    - On macOS (using Homebrew): `brew install tesseract tesseract-lang`
    - For other systems, refer to Tesseract documentation.

### Running the OCR Processing Script

The `scripts/trigger_ocr.py` script processes images (from S3 or local) and populates the SQLite database with OCR results. It also saves grayscale versions of images.

```bash
(venv) python scripts/trigger_ocr.py
```

Ensure `PROCESSING_MODE` and related paths/bucket names are correctly set in your `.env` file.

### Running the API Server

```bash
(venv) python main.py
```

The server will start, typically on `http://0.0.0.0:8000`.

## Testing

The `test.py` script provides examples for testing the `/ocr/text-match` and `/ocr/image-match` endpoints.

1.  Ensure the API server is running.
2.  Modify `test.py` with appropriate query texts, image paths, and thresholds.
3.  Run the test script:
    ```bash
    (venv) python test.py
    ```

You will need a sample grayscale image (e.g., `sample_grayscale_image.jpg` or use one from `local_images_grayscale/`) for `test_image_match_endpoint`.
