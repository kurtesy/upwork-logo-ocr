import logging
import pickle
from typing import List, Tuple

import faiss
import clip
import numpy as np
import torch
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

    def extract_text_features(self, text: str) -> np.ndarray:
        """Extracts a feature vector from a text string."""
        if not self.model:
            raise RuntimeError("CLIP model is not loaded.")
        text_inputs = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.squeeze().cpu().numpy()

    def search(self, query_features: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
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