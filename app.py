"""
Flask Web Application for Waste Classifier
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
from src.waste_classifier import WasteClassifier
import io
from PIL import Image

app = Flask(__name__)
CORS(app)

# Initialize classifier
classifier = WasteClassifier()

@app.route("/")
def index():
    """Home page."""
    return render_template("index.html")

@app.route("/classify", methods=["POST"])
def classify():
    """Classify uploaded image."""
    try:
        # Get image from request
        data = request.get_json()
        image_data = data["image"]

        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(",")[1])
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to OpenCV format
        img_array = np.array(image)
        img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Classify
        result = classifier.classify_image(img_cv2)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/stats")
def stats():
    """Get classification statistics."""
    # This would connect to a database in production
    stats = {
        "total_classifications": 1234,
        "accuracy_rate": 0.951,
        "recyclables_detected": 892,
        "waste_diverted": "342 kg"
    }
    return jsonify(stats)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
