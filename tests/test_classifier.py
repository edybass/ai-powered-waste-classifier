"""
Test cases for waste classifier
"""

import pytest
import numpy as np
from src.waste_classifier import WasteClassifier

class TestWasteClassifier:
    @pytest.fixture
    def classifier(self):
        """Create classifier instance for testing."""
        return WasteClassifier()

    def test_classifier_initialization(self, classifier):
        """Test classifier initializes correctly."""
        assert classifier is not None
        assert classifier.confidence_threshold == 0.8
        assert len(classifier.CATEGORIES) == 10

    def test_image_preprocessing(self, classifier):
        """Test image preprocessing."""
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        processed = classifier.preprocess_image(dummy_image)

        assert processed.shape == (1, 224, 224, 3)
        assert processed.dtype == np.float32
        assert 0 <= processed.min() <= processed.max() <= 1

    def test_categories(self, classifier):
        """Test waste categories are properly defined."""
        for idx, category in classifier.CATEGORIES.items():
            assert "name" in category
            assert "type" in category
            assert "bin" in category
            assert category["type"] in ["recyclable", "general", "compost", "hazardous", "textile", "electronic"]
