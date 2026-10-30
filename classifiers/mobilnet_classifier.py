"""
MobileNetV2 Vehicle Classification Module

Classifies vehicles into Cars, Buses, and Trucks using TensorFlow/Keras MobileNetV2.
This is a separately trained model that provides the final classification.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers


class MobileNetV2Classifier:
    """
    Vehicle classifier using MobileNetV2 transfer learning.
    
    Note: In a real internship scenario, this would be a model trained on a
    custom dataset of car/bus/truck images. Here we provide a skeleton that
    can be loaded with a trained model or initialized for training.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the MobileNetV2 classifier.
        
        Args:
            model_path: Path to trained model .h5 or .keras file. 
                       If None, initializes a base MobileNetV2 architecture.
        """
        self.model_path = model_path
        self.model = None
        self.class_names = ['Car', 'Bus', 'Truck']
        
        if model_path:
            self.load_model(model_path)
        else:
            self._build_base_model()
    
    def _build_base_model(self):
        """Build a base MobileNetV2 model for transfer learning."""
        print("Building base MobileNetV2 model (no pretrained weights)...")
        
        # Load MobileNetV2 without top classification layer,
        # with ImageNet weights for feature extraction
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Freeze the base model
        base_model.trainable = False
        
        # Build the classification head
        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Base MobileNetV2 model built successfully")
    
    def load_model(self, model_path):
        """
        Load a trained model from file.
        
        Args:
            model_path: Path to saved model file
        """
        print(f"Loading MobileNetV2 model from: {model_path}")
        try:
            self.model = tf.keras.models.load_model(model_path)
            # Try to recover class names if possible
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to base model architecture...")
            self._build_base_model()
    
    def predict(self, image_array):
        """
        Classify a single image.
        
        Args:
            image_array: numpy array of the image (H, W, 3)
            
        Returns:
            dict with keys: 'class_name', 'confidence', 'probabilities'
                - class_name: predicted class (Car, Bus, or Truck)
                - confidence: prediction confidence (0-1)
                - probabilities: dict of class -> probability
        """
        if self.model is None:
            raise ValueError("Model not loaded. Provide a model_path or ensure base model is built.")
        
        # Preprocess the image
        processed = self._preprocess_image(image_array)
        
        # Make prediction
        predictions = self.model.predict(processed, verbose=0)[0]
        
        # Get top prediction
        class_idx = int(np.argmax(predictions))
        confidence = float(predictions[class_idx])
        class_name = self.class_names[class_idx]
        
        # Build probabilities dict
        probabilities = {
            self.class_names[i]: float(predictions[i])
            for i in range(len(self.class_names))
        }
        
        return {
            'class_name': class_name,
            'confidence': confidence,
            'probabilities': probabilities,
            'source': 'MobileNetV2'
        }
    
    def _preprocess_image(self, image_array):
        """
        Preprocess image for MobileNetV2 prediction.
        
        Args:
            image_array: numpy array of the image (H, W, 3)
            
        Returns:
            preprocessed array ready for model input
        """
        # Resize to MobileNetV2 input size
        resized = cv2.resize(image_array, (224, 224))
        
        # Apply MobileNetV2 preprocessing (mean subtraction)
        # MobileNetV2 preprocesses with mean [104, 117, 123] in BGR order
        # or equivalently: x -= [103.939, 116.779, 123.68] in BGR
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        processed = preprocess_input(resized.astype(np.float32))
        
        # Add batch dimension
        processed = np.expand_dims(processed, axis=0)
        
        return processed
    
    def summary(self):
        """Print model summary."""
        if self.model:
            self.model.summary()
        else:
            print("No model loaded")


# Alias for convenience - will be replaced after MobileNetV2 training
trained_mobilnet_classifier = MobileNetV2Classifier