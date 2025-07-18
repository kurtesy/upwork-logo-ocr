import logging
import pickle
from typing import List, Tuple
from io import BytesIO

import faiss
import clip
import numpy as np
import torch
import cv2
from PIL import Image

logger = logging.getLogger(__name__)

class ImageSimilarityService:
    """
    A service for extracting image features and finding similar images using a Faiss index.
    """
    def __init__(self):
        # Use GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load the OpenAI CLIP model and its corresponding preprocessor
        try:
            # "ViT-B/32" is a good starting point. Other models like "ViT-L/14" are more powerful but slower.
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            self.model.eval() # Set model to evaluation mode
            logger.info("OpenAI CLIP model 'ViT-B/32' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}", exc_info=True)
            self.model = None
            self.preprocess = None

        self.index = None
        self.index_to_key_map = []

    def load_index(self, index_path: str, map_path: str):
        """Loads the Faiss index and the key mapping from disk."""
        try:
            self.index = faiss.read_index(index_path)
            with open(map_path, 'rb') as f:
                self.index_to_key_map = pickle.load(f)
            logger.info(f"Successfully loaded Faiss index with {self.index.ntotal} vectors and key map.")
        except Exception as e:
            logger.error(f"Failed to load Faiss index or map from {index_path}/{map_path}: {e}", exc_info=True)
            self.index = None # Ensure index is None if loading fails

    def _preprocess_for_shape_color(self, image_bytes: bytes) -> np.ndarray:
        """Helper to load image bytes into an OpenCV-readable format."""
        image_np = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image from bytes.")
        return img

    def extract_features(self, image_bytes: bytes) -> np.ndarray:
        """Extracts a feature vector from a single image's bytes."""
        from io import BytesIO
        if not self.model or not self.preprocess:
            raise RuntimeError("CLIP model is not loaded.")

        # The model expects a 3-channel RGB image.
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Use the model's image encoder
            features = self.model.encode_image(image_input)

        # Flatten the features and move to CPU as a numpy array
        # L2-normalize the features for cosine similarity search
        features /= features.norm(dim=-1, keepdim=True)
        return features.squeeze().cpu().numpy()

    def extract_color_features(self, image_bytes: bytes) -> np.ndarray:
        """Extracts a color histogram from an image's bytes."""
        img = self._preprocess_for_shape_color(image_bytes)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Using 50 bins for hue, 60 for saturation
        hist = cv2.calcHist([hsv_img], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    def extract_shape_features(self, image_bytes: bytes) -> np.ndarray:
        """Extracts shape features (main contour) from an image's bytes."""
        img = self._preprocess_for_shape_color(image_bytes)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_img, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.array([])
        return max(contours, key=cv2.contourArea)

    def compare_color_features(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """Compares two color histograms using correlation."""
        if hist1.size == 0 or hist2.size == 0:
            return 0.0
        # HISTCMP_CORREL returns a value between -1 and 1. We normalize it to [0, 1].
        score = cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_CORREL)
        return (score + 1) / 2

    def compare_shape_features(self, contour1: np.ndarray, contour2: np.ndarray) -> float:
        """Compares two contours using cv2.matchShapes."""
        if contour1.size == 0 or contour2.size == 0:
            return 0.0
        # CONTOURS_MATCH_I1 returns a distance. Lower is better.
        distance = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0.0)
        # Convert distance to a similarity score between 0 and 1.
        similarity = 1.0 / (1.0 + distance)
        return similarity

    def extract_text_features(self, text: str) -> np.ndarray:
        """Extracts a feature vector from a text string."""
        if not self.model:
            raise RuntimeError("CLIP model is not loaded.")
        text_inputs = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.squeeze().cpu().numpy()

    def search(self, query_features: np.ndarray, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Searches the Faiss index for the top_k most similar images.

        Returns:
            A list of tuples, where each tuple contains (s3_image_key, distance).
            Lower distance means higher similarity.
        """
        if self.index is None:
            logger.warning("Search attempted but Faiss index is not loaded.")
            return []

        # Faiss requires a 2D array for searching
        query_features_2d = np.expand_dims(query_features, axis=0).astype('float32')

        # When features are L2-normalized, L2 distance is monotonically related to cosine similarity.
        # For normalized vectors, (d^2) = 2 - 2 * cos_sim. So minimizing L2 distance maximizes cosine similarity.
        distances, indices = self.index.search(query_features_2d, top_k)

        results = [(self.index_to_key_map[i], float(d)) for d, i in zip(distances[0], indices[0])]

        return results