import requests
import json
import os

# Define the URL of your FastAPI application's /trigger-ocr endpoint
# Update this if your application is running on a different host or port
BASE_API_URL = "http://3.110.127.175:8000"
# BASE_API_URL = "http://localhost:8000"
LOGO_MATCH_URL = f"{BASE_API_URL}/ocr/text-match" # Corrected from find-match
IMAGE_MATCH_URL = f"{BASE_API_URL}/ocr/image-match"
BULK_LOGO_MATCH_URL = f"{BASE_API_URL}/ocr/bulk-text-match"
BULK_IMAGE_MATCH_URL = f"{BASE_API_URL}/ocr/bulk-image-match"

API_KEY = os.getenv("API_KEY", "your_test_api_key") # Use an env var or a default for testing

def test_find_match_endpoint(query_text: str, similarity_level: str):
    """
    Sends a GET request to the /ocr/text-match endpoint with the given query text and threshold.
    """
    params = {
        "query_text": query_text,
        "similarity_level": similarity_level
    }
    headers = {"X-API-KEY": API_KEY}
    print(f"\nSending GET request to: {LOGO_MATCH_URL} with params: {params}")
    try:
        response = requests.get(LOGO_MATCH_URL, params=params, headers=headers)
        response.raise_for_status()

        print(f"Response Status Code: {response.status_code}")
        try:
            response_json = response.json()
            print("Response JSON:")
            print(json.dumps(response_json, indent=2))
        except json.JSONDecodeError:
            print("Response content is not valid JSON:")
            print(response.text)

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(f"Response Content: {response.content.decode()}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err}")
        print("Please ensure the FastAPI server is running and accessible at the specified URL.")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"An unexpected error occurred with the request: {req_err}")

def test_image_match_endpoint(image_path: str, img_cnt: int = 10):
    """
    Sends a POST request to the /ocr/image-match endpoint with the given image file and count.
    """
    params = {
        "img_cnt": img_cnt
    }
    headers = {"X-API-KEY": API_KEY}
    print(f"\nSending POST request to: {IMAGE_MATCH_URL} with image: {image_path} and params: {params}")
    try:
        with open(image_path, 'rb') as image_file:
            files = {'uploaded_file': (os.path.basename(image_path), image_file, 'image/jpeg')} # Adjust content type if needed
            response = requests.post(IMAGE_MATCH_URL, files=files, params=params, headers=headers)

        response.raise_for_status()

        print(f"Response Status Code: {response.status_code}")
        try:
            response_json = response.json()
            print("Response JSON:")
            print(json.dumps(response_json, indent=2))
        except json.JSONDecodeError:
            print("Response content is not valid JSON:")
            print(response.text)

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(f"Response Content: {response.content.decode()}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err}")
        print("Please ensure the FastAPI server is running and accessible at the specified URL.")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"An unexpected error occurred with the request: {req_err}")

def test_bulk_logo_match_endpoint(queries: list, similarity_level: str):
    """
    Sends a POST request to the /ocr/bulk-text-match endpoint.
    """
    payload = {
        "queries": [{"query_text": q} for q in queries],
        "similarity_level": similarity_level
    }
    headers = {"X-API-KEY": API_KEY, "Content-Type": "application/json"}
    print(f"\nSending POST request to: {BULK_LOGO_MATCH_URL} with payload: {json.dumps(payload, indent=2)}")
    try:
        response = requests.post(BULK_LOGO_MATCH_URL, json=payload, headers=headers)
        response.raise_for_status()
        print(f"Response Status Code: {response.status_code}")
        print("Response JSON:")
        print(json.dumps(response.json(), indent=2))
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(f"Response Content: {response.content.decode()}")
    except Exception as e:
        print(f"An error occurred: {e}")

def test_bulk_image_match_endpoint(image_paths: list, img_cnt: int = 10):
    """
    Sends a POST request to the /ocr/bulk-image-match endpoint.
    """
    params = {"img_cnt": img_cnt}
    headers = {"X-API-KEY": API_KEY}
    files_to_upload = []
    try:
        for image_path in image_paths:
            files_to_upload.append(('uploaded_files', (os.path.basename(image_path), open(image_path, 'rb'), 'image/jpeg')))
        
        print(f"\nSending POST request to: {BULK_IMAGE_MATCH_URL} with {len(image_paths)} images and params: {params}")
        response = requests.post(BULK_IMAGE_MATCH_URL, files=files_to_upload, params=params, headers=headers)
        response.raise_for_status()
        print(f"Response Status Code: {response.status_code}")
        print("Response JSON:")
        print(json.dumps(response.json(), indent=2))
    except FileNotFoundError as fnf_err:
        print(f"Error: Image file not found: {fnf_err}")
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(f"Response Content: {response.content.decode()}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Close all opened files
        for _, file_tuple in files_to_upload:
            if hasattr(file_tuple[1], 'close'):
                file_tuple[1].close()

if __name__ == "__main__":
    # Test for OCR logo text matching
    test_find_match_endpoint(query_text="KAING", similarity_level="low")
    test_find_match_endpoint(query_text="test", similarity_level="high")

    # Test for image similarity matching
    # Make sure 'local_images_grayscale/5866394.jpeg' exists or change the path
    test_image_match_endpoint(image_path="local_images_grayscale/5866394.jpeg", img_cnt=5)

    # Test for bulk OCR logo text matching
    test_bulk_logo_match_endpoint(queries=["KAING", "test", "another one"], similarity_level="medium")

    # Test for bulk image similarity matching
    # Ensure paths are correct and images exist
    # test_bulk_image_match_endpoint(image_paths=["local_images_grayscale/5866394.jpeg", "local_images_grayscale/5866394.jpeg"], img_cnt=5)
