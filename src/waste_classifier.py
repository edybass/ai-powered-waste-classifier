"""
AI Waste Classifier - Main Module
Real-time waste classification using deep learning
"""

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
import time

class WasteClassifier:
    """Main waste classification class."""

    # Waste categories with recycling info
    CATEGORIES = {
        0: {"name": "cardboard", "type": "recyclable", "bin": "blue"},
        1: {"name": "glass", "type": "recyclable", "bin": "green"},
        2: {"name": "metal", "type": "recyclable", "bin": "yellow"},
        3: {"name": "paper", "type": "recyclable", "bin": "blue"},
        4: {"name": "plastic", "type": "recyclable", "bin": "yellow"},
        5: {"name": "trash", "type": "general", "bin": "black"},
        6: {"name": "organic", "type": "compost", "bin": "brown"},
        7: {"name": "battery", "type": "hazardous", "bin": "red"},
        8: {"name": "clothes", "type": "textile", "bin": "purple"},
        9: {"name": "e-waste", "type": "electronic", "bin": "orange"}
    }

    def __init__(self, model_path=None, confidence_threshold=0.8):
        """Initialize the classifier."""
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.load_model(model_path)

    def load_model(self, model_path=None):
        """Load the pre-trained model."""
        if model_path is None:
            model_path = Path("data/models/waste_classifier_efficientnet.h5")

        print(f"Loading model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")

    def preprocess_image(self, image):
        """Preprocess image for model input."""
        # Resize to model input size
        img_resized = cv2.resize(image, (224, 224))

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # Normalize pixel values
        img_normalized = img_rgb.astype(np.float32) / 255.0

        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)

        return img_batch

    def classify_image(self, image):
        """Classify a single image."""
        # Preprocess
        processed_img = self.preprocess_image(image)

        # Predict
        start_time = time.time()
        predictions = self.model.predict(processed_img, verbose=0)[0]
        inference_time = time.time() - start_time

        # Get top prediction
        top_idx = np.argmax(predictions)
        confidence = predictions[top_idx]

        # Get category info
        category_info = self.CATEGORIES[top_idx]

        result = {
            "category": category_info["name"],
            "type": category_info["type"],
            "bin_color": category_info["bin"],
            "confidence": float(confidence),
            "inference_time": inference_time,
            "all_predictions": {
                self.CATEGORIES[i]["name"]: float(predictions[i]) 
                for i in range(len(predictions))
            }
        }

        return result

    def classify_from_camera(self, camera_id=0, show_preview=True):
        """Classify waste from camera feed."""
        cap = cv2.VideoCapture(camera_id)

        print("Press SPACE to capture and classify, Q to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if show_preview:
                cv2.imshow("Waste Classifier - Press SPACE to classify", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                result = self.classify_image(frame)
                self._display_result(frame, result)
            elif key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def _display_result(self, image, result):
        """Display classification result on image."""
        # Create result image
        result_img = image.copy()

        # Add text overlay
        text = f"{result['category'].upper()} ({result['confidence']:.1%})"
        bin_text = f"Bin: {result['bin_color'].upper()}"

        cv2.putText(result_img, text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(result_img, bin_text, (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Show result
        cv2.imshow("Classification Result", result_img)
        cv2.waitKey(3000)

        return result_img

# Example usage
if __name__ == "__main__":
    classifier = WasteClassifier()

    # Test with camera
    classifier.classify_from_camera()
