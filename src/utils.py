"""
Utility functions for waste classifier
"""

import cv2
import numpy as np
from PIL import Image
import requests
import base64
import io
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def preprocess_batch(images: List[np.ndarray], target_size=(224, 224)) -> np.ndarray:
    """Preprocess a batch of images."""
    processed = []

    for img in images:
        # Resize
        img_resized = cv2.resize(img, target_size)

        # Convert BGR to RGB if needed
        if img_resized.shape[2] == 3 and img_resized[:,:,0].mean() > img_resized[:,:,2].mean():
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # Normalize
        img_normalized = img_resized.astype(np.float32) / 255.0
        processed.append(img_normalized)

    return np.array(processed)

def draw_prediction_on_image(image: np.ndarray, prediction: Dict) -> np.ndarray:
    """Draw prediction results on image."""
    img_copy = image.copy()
    height, width = img_copy.shape[:2]

    # Define colors for different waste types
    colors = {
        'recyclable': (0, 255, 0),    # Green
        'organic': (139, 69, 19),      # Brown
        'hazardous': (0, 0, 255),      # Red
        'general': (128, 128, 128),    # Gray
        'electronic': (255, 165, 0),   # Orange
        'textile': (128, 0, 128)       # Purple
    }

    # Get color based on waste type
    color = colors.get(prediction['type'], (255, 255, 255))

    # Draw background rectangle
    cv2.rectangle(img_copy, (10, 10), (width-10, 100), (0, 0, 0), -1)
    cv2.rectangle(img_copy, (10, 10), (width-10, 100), color, 3)

    # Add text
    category_text = f"{prediction['category'].upper()}"
    confidence_text = f"Confidence: {prediction['confidence']:.1%}"
    bin_text = f"Bin: {prediction['bin_color'].upper()}"

    cv2.putText(img_copy, category_text, (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    cv2.putText(img_copy, confidence_text, (20, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(img_copy, bin_text, (20, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    return img_copy

def create_confusion_matrix_plot(y_true: List[int], y_pred: List[int], 
                                class_names: List[str], save_path: str = None):
    """Create and save confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Waste Classification', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def calculate_environmental_impact(classifications: List[Dict]) -> Dict:
    """Calculate environmental impact based on classifications."""
    # Average weights for different waste types (kg)
    avg_weights = {
        'plastic': 0.15,
        'glass': 0.35,
        'metal': 0.25,
        'paper': 0.10,
        'cardboard': 0.20,
        'organic': 0.30,
        'e-waste': 0.50,
        'battery': 0.05,
        'textile': 0.40
    }

    # CO2 savings per kg for recycling (kg CO2)
    co2_savings = {
        'plastic': 1.5,
        'glass': 0.3,
        'metal': 2.0,
        'paper': 0.9,
        'cardboard': 0.9,
        'organic': 0.5,  # Composting
        'e-waste': 3.0,
        'battery': 2.5,
        'textile': 2.0
    }

    total_weight = 0
    total_co2_saved = 0
    recyclables_count = 0

    for classification in classifications:
        category = classification['category']
        if category in avg_weights:
            weight = avg_weights[category]
            total_weight += weight

            if classification['type'] in ['recyclable', 'compost', 'electronic']:
                recyclables_count += 1
                total_co2_saved += weight * co2_savings.get(category, 0)

    recycling_rate = recyclables_count / len(classifications) if classifications else 0

    return {
        'total_items': len(classifications),
        'recyclables_count': recyclables_count,
        'recycling_rate': recycling_rate,
        'estimated_weight_kg': round(total_weight, 2),
        'co2_saved_kg': round(total_co2_saved, 2),
        'trees_equivalent': round(total_co2_saved / 21.77, 2),  # 1 tree absorbs ~21.77 kg CO2/year
        'car_miles_equivalent': round(total_co2_saved / 0.404, 2)  # 1 mile = 0.404 kg CO2
    }

def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def decode_base64_to_image(base64_string: str) -> np.ndarray:
    """Decode base64 string to OpenCV image."""
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def download_sample_dataset():
    """Download sample waste images for testing."""
    urls = {
        'plastic_bottle': 'https://example.com/plastic_bottle.jpg',
        'glass_jar': 'https://example.com/glass_jar.jpg',
        'cardboard_box': 'https://example.com/cardboard_box.jpg',
        'aluminum_can': 'https://example.com/aluminum_can.jpg',
        'food_waste': 'https://example.com/food_waste.jpg'
    }

    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)

    for name, url in urls.items():
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(examples_dir / f"{name}.jpg", "wb") as f:
                    f.write(response.content)
                print(f"✅ Downloaded {name}")
        except:
            print(f"❌ Failed to download {name}")

def generate_synthetic_data(num_samples: int = 100):
    """Generate synthetic waste images for testing."""
    categories = ['plastic', 'glass', 'metal', 'paper', 'organic']
    colors = {
        'plastic': (100, 150, 255),  # Light blue
        'glass': (200, 255, 200),    # Light green
        'metal': (192, 192, 192),    # Silver
        'paper': (245, 245, 220),    # Beige
        'organic': (139, 90, 43)     # Brown
    }

    synthetic_dir = Path("data/synthetic")
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_samples):
        category = np.random.choice(categories)

        # Create synthetic image
        img = np.ones((224, 224, 3), dtype=np.uint8) * 255

        # Add colored shape
        color = colors[category]
        shape_type = np.random.choice(['circle', 'rectangle', 'triangle'])

        if shape_type == 'circle':
            cv2.circle(img, (112, 112), 80, color, -1)
        elif shape_type == 'rectangle':
            cv2.rectangle(img, (62, 62), (162, 162), color, -1)
        else:  # triangle
            pts = np.array([[112, 32], [32, 192], [192, 192]], np.int32)
            cv2.fillPoly(img, [pts], color)

        # Add noise
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)

        # Save
        filename = f"{category}_{i:04d}.jpg"
        cv2.imwrite(str(synthetic_dir / filename), img)

    print(f"✅ Generated {num_samples} synthetic images")

if __name__ == "__main__":
    # Generate some synthetic data for testing
    generate_synthetic_data(50)
