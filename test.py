import requests
import json

# Define the URL of your FastAPI application's /trigger-ocr endpoint
# Update this if your application is running on a different host or port
BASE_API_URL = "http://127.0.0.1:8000"
LOGO_MATCH_URL = f"{BASE_API_URL}/ocr/logo-match"
IMAGE_MATCH_URL = f"{BASE_API_URL}/ocr/image-match"

def test_find_match_endpoint(query_text: str, similarity_threshold: float):
    """
    Sends a GET request to the /ocr/logo-match endpoint with the given query text and threshold.
    """
    params = {
        "query_text": query_text,
        "similarity_threshold": similarity_threshold
    }
    print(f"\nSending GET request to: {LOGO_MATCH_URL} with params: {params}")
    try:
        response = requests.get(LOGO_MATCH_URL, params=params)
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

def test_image_match_endpoint(image_path: str, similarity_threshold: float):
    """
    Sends a POST request to the /ocr/image-match endpoint with the given image file and threshold.
    """
    params = {
        "similarity_threshold": similarity_threshold
    }
    print(f"\nSending POST request to: {IMAGE_MATCH_URL} with image: {image_path} and params: {params}")
    try:
        with open(image_path, 'rb') as image_file:
            files = {'uploaded_file': (image_path, image_file, 'image/jpeg')} # Adjust content type if needed
            response = requests.post(IMAGE_MATCH_URL, files=files, params=params)
        
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

if __name__ == "__main__":
    # Test for OCR logo text matching
    # test_find_match_endpoint(query_text="KAING", similarity_threshold=0.4)
    # test_find_match_endpoint(query_text="test", similarity_threshold=0.8)

    # Test for image similarity matching
    # Make sure 'sample_grayscale_image.jpg' exists or change the path
    test_image_match_endpoint(image_path="local_images_grayscale/5866394.jpeg", similarity_threshold=0.5)
