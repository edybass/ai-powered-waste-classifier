"""
Comprehensive tests for waste classifier
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import json
import tempfile
from src.waste_classifier import WasteClassifier
from src.utils import preprocess_batch, calculate_environmental_impact

class TestWasteClassifier:
    """Test suite for waste classifier."""

    @pytest.fixture
    def classifier(self):
        """Create classifier instance for testing."""
        return WasteClassifier(confidence_threshold=0.7)

    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        # Create a synthetic image
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        return img

    def test_classifier_initialization(self, classifier):
        """Test classifier initializes correctly."""
        assert classifier is not None
        assert classifier.confidence_threshold == 0.7
        assert len(classifier.CATEGORIES) == 10
        assert all(cat in classifier.CATEGORIES[0] for cat in ['name', 'type', 'bin'])

    def test_image_preprocessing(self, classifier, sample_image):
        """Test image preprocessing pipeline."""
        processed = classifier.preprocess_image(sample_image)

        # Check output shape
        assert processed.shape == (1, 224, 224, 3)

        # Check data type and range
        assert processed.dtype == np.float32
        assert 0 <= processed.min() <= processed.max() <= 1

    def test_classification_output_format(self, classifier, sample_image):
        """Test classification output structure."""
        # Mock model prediction
        classifier.model = MockModel()

        result = classifier.classify_image(sample_image)

        # Check all required fields
        assert 'category' in result
        assert 'type' in result
        assert 'bin_color' in result
        assert 'confidence' in result
        assert 'inference_time' in result
        assert 'all_predictions' in result

        # Check data types
        assert isinstance(result['category'], str)
        assert isinstance(result['confidence'], float)
        assert 0 <= result['confidence'] <= 1
        assert result['inference_time'] > 0

    def test_batch_preprocessing(self):
        """Test batch preprocessing functionality."""
        images = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(5)]

        batch = preprocess_batch(images)

        assert batch.shape == (5, 224, 224, 3)
        assert batch.dtype == np.float32
        assert np.all((batch >= 0) & (batch <= 1))

    def test_environmental_impact_calculation(self):
        """Test environmental impact calculations."""
        classifications = [
            {'category': 'plastic', 'type': 'recyclable'},
            {'category': 'glass', 'type': 'recyclable'},
            {'category': 'organic', 'type': 'compost'},
            {'category': 'trash', 'type': 'general'},
            {'category': 'metal', 'type': 'recyclable'}
        ]

        impact = calculate_environmental_impact(classifications)

        assert impact['total_items'] == 5
        assert impact['recyclables_count'] == 4
        assert impact['recycling_rate'] == 0.8
        assert impact['estimated_weight_kg'] > 0
        assert impact['co2_saved_kg'] > 0
        assert impact['trees_equivalent'] > 0

    def test_model_loading_error_handling(self):
        """Test error handling for missing model files."""
        with pytest.raises(Exception):
            classifier = WasteClassifier(model_path="nonexistent_model.h5")

    def test_category_consistency(self, classifier):
        """Test that all categories have consistent structure."""
        for idx, category in classifier.CATEGORIES.items():
            assert isinstance(idx, int)
            assert 'name' in category
            assert 'type' in category
            assert 'bin' in category

            # Check valid types
            valid_types = ['recyclable', 'general', 'compost', 'hazardous', 'textile', 'electronic']
            assert category['type'] in valid_types

            # Check valid bin colors
            valid_bins = ['blue', 'green', 'yellow', 'black', 'brown', 'red', 'orange', 'purple']
            assert category['bin'] in valid_bins

class MockModel:
    """Mock model for testing."""

    def predict(self, x, verbose=0):
        """Return mock predictions."""
        # Return random predictions for 10 categories
        predictions = np.random.dirichlet(np.ones(10))
        return np.array([predictions])

@pytest.mark.integration
class TestIntegration:
    """Integration tests requiring actual model."""

    def test_end_to_end_classification(self):
        """Test complete classification pipeline."""
        # This would run with actual model in CI/CD
        pass

    def test_api_integration(self):
        """Test API endpoints."""
        # This would test the Flask API
        pass
