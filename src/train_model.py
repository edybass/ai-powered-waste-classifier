"""
Model Training Script for Waste Classifier
Train EfficientNet-B0 on waste classification dataset
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

class WasteClassifierTrainer:
    """Train waste classification models."""

    def __init__(self, data_dir="data/dataset", model_name="efficientnet"):
        self.data_dir = Path(data_dir)
        self.model_name = model_name
        self.img_size = (224, 224)
        self.batch_size = 32
        self.epochs = 100

        # Data augmentation
        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest',
            validation_split=0.2
        )

        self.test_datagen = ImageDataGenerator(rescale=1./255)

    def prepare_data(self):
        """Prepare data generators."""
        self.train_generator = self.train_datagen.flow_from_directory(
            self.data_dir / "train",
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training'
        )

        self.validation_generator = self.train_datagen.flow_from_directory(
            self.data_dir / "train",
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation'
        )

        self.test_generator = self.test_datagen.flow_from_directory(
            self.data_dir / "test",
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )

        # Save class indices
        with open("data/models/class_indices.json", "w") as f:
            json.dump(self.train_generator.class_indices, f)

        self.num_classes = len(self.train_generator.class_indices)
        print(f"Found {self.num_classes} classes")

    def build_model(self):
        """Build the model architecture."""
        if self.model_name == "efficientnet":
            base_model = EfficientNetB0(
                input_shape=(*self.img_size, 3),
                include_top=False,
                weights='imagenet'
            )
            base_model.trainable = False

            inputs = tf.keras.Input(shape=(*self.img_size, 3))
            x = base_model(inputs, training=False)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(256, activation='relu')(x)
            x = layers.Dropout(0.5)(x)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.Dropout(0.3)(x)
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)

            self.model = models.Model(inputs, outputs)

        elif self.model_name == "mobilenet":
            base_model = tf.keras.applications.MobileNetV3Large(
                input_shape=(*self.img_size, 3),
                include_top=False,
                weights='imagenet'
            )
            base_model.trainable = False

            self.model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation='softmax')
            ])

        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        print(self.model.summary())

    def train(self):
        """Train the model."""
        # Callbacks
        checkpoint = ModelCheckpoint(
            f'data/models/best_{self.model_name}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )

        # Train
        history = self.model.fit(
            self.train_generator,
            epochs=self.epochs,
            validation_data=self.validation_generator,
            callbacks=[checkpoint, early_stop, reduce_lr],
            verbose=1
        )

        # Fine-tuning phase
        print("\nStarting fine-tuning phase...")
        base_model = self.model.layers[0] if self.model_name == "mobilenet" else self.model.layers[1]
        base_model.trainable = True

        # Freeze early layers
        for layer in base_model.layers[:100]:
            layer.trainable = False

        # Recompile with lower learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        # Continue training
        history_fine = self.model.fit(
            self.train_generator,
            epochs=30,
            validation_data=self.validation_generator,
            callbacks=[checkpoint, early_stop, reduce_lr],
            verbose=1
        )

        # Save final model
        self.model.save(f'data/models/waste_classifier_{self.model_name}.h5')

        # Plot training history
        self.plot_history(history)

        return history

    def evaluate(self):
        """Evaluate model on test set."""
        results = self.model.evaluate(self.test_generator)
        print(f"\nTest Loss: {results[0]:.4f}")
        print(f"Test Accuracy: {results[1]:.4f}")
        print(f"Test Precision: {results[2]:.4f}")
        print(f"Test Recall: {results[3]:.4f}")

        # Generate predictions for confusion matrix
        predictions = self.model.predict(self.test_generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = self.test_generator.classes

        # Calculate per-class metrics
        from sklearn.metrics import classification_report, confusion_matrix

        class_names = list(self.train_generator.class_indices.keys())
        report = classification_report(y_true, y_pred, target_names=class_names)
        print("\nClassification Report:")
        print(report)

        # Save metrics
        metrics = {
            'test_loss': float(results[0]),
            'test_accuracy': float(results[1]),
            'test_precision': float(results[2]),
            'test_recall': float(results[3]),
            'classification_report': report
        }

        with open(f'data/models/metrics_{self.model_name}.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        return results

    def plot_history(self, history):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Accuracy plot
        ax1.plot(history.history['accuracy'], label='Train')
        ax1.plot(history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)

        # Loss plot
        ax2.plot(history.history['loss'], label='Train')
        ax2.plot(history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('docs/images/training_history.png', dpi=150)
        plt.close()

if __name__ == "__main__":
    # Train EfficientNet model
    trainer = WasteClassifierTrainer(model_name="efficientnet")
    trainer.prepare_data()
    trainer.build_model()
    trainer.train()
    trainer.evaluate()

    # Train MobileNet model for edge devices
    trainer_mobile = WasteClassifierTrainer(model_name="mobilenet")
    trainer_mobile.prepare_data()
    trainer_mobile.build_model()
    trainer_mobile.train()
    trainer_mobile.evaluate()
