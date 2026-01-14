# 🧠 Patch Classifier Module
# Individual patch classification using trained models

import torch
import torch.nn as nn
from torchvision import models, transforms
from typing import Tuple, List, Optional, Dict
import numpy as np
from PIL import Image


class ResNet50Classifier(nn.Module):
    """
    ResNet50 model for binary tumor/normal classification.
    Compatible with your trained model in models/best_resnet50_model.pth
    """
    
    def __init__(self, num_classes: int = 1, pretrained: bool = False, freeze_backbone: bool = False):
        super(ResNet50Classifier, self).__init__()
        
        # Load ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)
    
    def get_features(self, x):
        """Extract features before classification layer"""
        # Get features from backbone (before final FC)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        return x  # (B, 2048, 7, 7) for ResNet50


class PatchClassifier:
    """
    Wrapper for patch-level classification with your trained model.
    Handles preprocessing, inference, and confidence scoring.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        class_names: List[str] = ['Normal', 'Tumor'],
        threshold: float = 0.5
    ):
        self.device = torch.device(device)
        self.class_names = class_names
        self.threshold = threshold
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = ResNet50Classifier(num_classes=1, pretrained=False)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        print("✅ Model loaded successfully!")
        
        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((96, 96)),  # Camelyon16 patch size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def preprocess_patch(self, patch: np.ndarray) -> torch.Tensor:
        """Convert numpy patch to model input tensor"""
        if isinstance(patch, np.ndarray):
            patch = Image.fromarray(patch)
        
        tensor = self.transform(patch).unsqueeze(0)
        return tensor.to(self.device)
    
    def predict_patch(
        self,
        patch: np.ndarray,
        return_features: bool = False
    ) -> Dict:
        """
        Classify a single patch.
        
        Args:
            patch: (H, W, 3) numpy array
            return_features: Whether to return feature embeddings
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        input_tensor = self.preprocess_patch(patch)
        
        # Forward pass
        with torch.no_grad():
            # Get prediction
            output = self.model(input_tensor).squeeze()
            probability = torch.sigmoid(output).item()
            
            # Get features if requested
            features = None
            if return_features:
                features = self.model.get_features(input_tensor)
                features = features.cpu().numpy()
        
        # Interpret results
        confidence = probability * 100
        predicted_class = 1 if probability > self.threshold else 0
        predicted_name = self.class_names[predicted_class]
        
        return {
            'class_id': predicted_class,
            'class_name': predicted_name,
            'tumor_probability': probability,
            'normal_probability': 1 - probability,
            'confidence': confidence,
            'features': features
        }
    
    def predict_batch(
        self,
        patches: List[np.ndarray],
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Classify multiple patches efficiently.
        
        Args:
            patches: List of patches as numpy arrays
            batch_size: Batch size for inference
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i in range(0, len(patches), batch_size):
            batch_patches = patches[i:i+batch_size]
            
            # Preprocess batch
            batch_tensors = []
            for patch in batch_patches:
                if isinstance(patch, np.ndarray):
                    patch = Image.fromarray(patch)
                tensor = self.transform(patch)
                batch_tensors.append(tensor)
            
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(batch_tensor).squeeze()
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                probabilities = torch.sigmoid(outputs)
            
            # Process results
            for prob in probabilities:
                prob_value = prob.item()
                confidence = prob_value * 100
                predicted_class = 1 if prob_value > self.threshold else 0
                
                results.append({
                    'class_id': predicted_class,
                    'class_name': self.class_names[predicted_class],
                    'tumor_probability': prob_value,
                    'normal_probability': 1 - prob_value,
                    'confidence': confidence
                })
        
        return results


class EnsembleClassifier:
    """
    Ensemble of multiple models for robust predictions.
    Combines predictions using voting or averaging.
    """
    
    def __init__(
        self,
        model_paths: List[str],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        ensemble_method: str = 'average'  # 'average' or 'vote'
    ):
        self.device = torch.device(device)
        self.ensemble_method = ensemble_method
        self.models = []
        
        # Load all models
        for model_path in model_paths:
            model = ResNet50Classifier(num_classes=1)
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint)
            model.to(self.device)
            model.eval()
            self.models.append(model)
        
        print(f"✅ Loaded {len(self.models)} models for ensemble")
        
        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def predict(self, patch: np.ndarray) -> Dict:
        """Ensemble prediction on single patch"""
        # Preprocess
        if isinstance(patch, np.ndarray):
            patch = Image.fromarray(patch)
        input_tensor = self.transform(patch).unsqueeze(0).to(self.device)
        
        # Get predictions from all models
        predictions = []
        with torch.no_grad():
            for model in self.models:
                output = model(input_tensor).squeeze()
                prob = torch.sigmoid(output).item()
                predictions.append(prob)
        
        # Ensemble
        if self.ensemble_method == 'average':
            final_prob = np.mean(predictions)
        elif self.ensemble_method == 'vote':
            votes = [1 if p > 0.5 else 0 for p in predictions]
            final_prob = np.mean(votes)
        else:
            final_prob = np.mean(predictions)
        
        predicted_class = 1 if final_prob > 0.5 else 0
        
        return {
            'class_id': predicted_class,
            'class_name': ['Normal', 'Tumor'][predicted_class],
            'tumor_probability': final_prob,
            'confidence': final_prob * 100 if predicted_class == 1 else (1-final_prob) * 100,
            'individual_predictions': predictions,
            'uncertainty': np.std(predictions)  # Higher = more disagreement
        }


class FeatureExtractor:
    """
    Extract deep features from patches for clustering/similarity analysis.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        layer: str = 'layer4'  # Which layer to extract from
    ):
        self.device = torch.device(device)
        self.layer = layer
        
        # Load model
        self.model = ResNet50Classifier(num_classes=1)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract_features(
        self,
        patches: List[np.ndarray],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Extract features from patches.
        
        Returns:
            features: (N, feature_dim) numpy array
        """
        all_features = []
        
        for i in range(0, len(patches), batch_size):
            batch_patches = patches[i:i+batch_size]
            
            # Preprocess
            batch_tensors = []
            for patch in batch_patches:
                if isinstance(patch, np.ndarray):
                    patch = Image.fromarray(patch)
                tensor = self.transform(patch)
                batch_tensors.append(tensor)
            
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model.get_features(batch_tensor)
                # Global average pooling
                features = torch.mean(features, dim=(2, 3))  # (B, 2048)
                all_features.append(features.cpu().numpy())
        
        return np.vstack(all_features)


# ============================================
# UTILITY FUNCTIONS
# ============================================

def calculate_patch_statistics(predictions: List[Dict]) -> Dict:
    """Calculate statistics across multiple patches"""
    tumor_patches = sum(1 for p in predictions if p['class_id'] == 1)
    normal_patches = len(predictions) - tumor_patches
    
    avg_tumor_prob = np.mean([p['tumor_probability'] for p in predictions])
    max_tumor_prob = max([p['tumor_probability'] for p in predictions])
    
    return {
        'total_patches': len(predictions),
        'tumor_patches': tumor_patches,
        'normal_patches': normal_patches,
        'tumor_ratio': tumor_patches / len(predictions) if predictions else 0,
        'avg_tumor_probability': avg_tumor_prob,
        'max_tumor_probability': max_tumor_prob,
        'is_tumor': tumor_patches > (len(predictions) * 0.1)  # >10% tumor patches
    }


if __name__ == "__main__":
    print("🧠 Patch Classifier Module")
    print("=" * 70)
    
    # Test classifier
    import os
    
    model_path = 'models/best_resnet50_model.pth'
    
    if os.path.exists(model_path):
        classifier = PatchClassifier(model_path=model_path)
        
        # Create dummy patch
        dummy_patch = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
        
        # Test prediction
        result = classifier.predict_patch(dummy_patch)
        
        print(f"\n✅ Test Prediction:")
        print(f"   Class: {result['class_name']}")
        print(f"   Confidence: {result['confidence']:.2f}%")
        print(f"   Tumor Probability: {result['tumor_probability']:.4f}")
        
        print("\n🧠 Classifier ready for use!")
    else:
        print(f"ℹ️  Model not found at {model_path}")
        print("   Module ready for use once model is available.")
