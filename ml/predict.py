import os
import sys
import argparse
import logging
import json
import base64
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np

# Add the ML module to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.tumor_predictor import TumorPredictor
from utils.image_utils import save_prediction_visualization, enhance_medical_image
from utils.data_manager import DataManager, save_prediction_report
from config.config import get_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TumorPredictionService:
    """
    Service for making tumor predictions on images.
    """
    
    def __init__(self, model_path=None, config_env='development'):
        self.config = get_config(config_env)
        self.config.create_directories()
        self.predictor = TumorPredictor()
        self.data_manager = DataManager(str(self.config.DATABASE_PATH))
        
        # Load model
        if model_path and os.path.exists(model_path):
            self.predictor.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
        elif os.path.exists(self.config.PRETRAINED_MODEL_PATH):
            self.predictor.load_model(str(self.config.PRETRAINED_MODEL_PATH))
            logger.info("Pre-trained model loaded")
        else:
            self.predictor.build_model()
            logger.warning("Using untrained model. Results may not be accurate.")
    
        # -----------------------------------
        # 🔒 Ensure deterministic predictions
        # -----------------------------------
        import torch
        import numpy as np
        import random

        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_grad_enabled(False)

        # Make sure model runs in eval mode (no dropout/batchnorm randomness)
        try:
            self.predictor.model.eval()
            logger.info("Model set to evaluation mode for consistent results.")
        except Exception:
            logger.warning("Could not set model to eval mode — ensure TumorPredictor defines self.model")

    def predict_single_image(self, image_path, enhance=False, save_viz=False, 
                           save_to_db=True, user_id=None):
        """
        Predict tumor presence in a single image.
        
        Args:
            image_path: Path to the image file
            enhance: Whether to apply image enhancement
            save_viz: Whether to save visualization
            save_to_db: Whether to save result to database
            user_id: Optional user identifier
            
        Returns:
            Prediction results dictionary
        """
      
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            logger.info(f"Processing image: {image_path}")
            start_time = datetime.now()
            
            # Load and preprocess image
            from utils.image_utils import load_and_preprocess_image
            image = load_and_preprocess_image(image_path)
            
            # Apply enhancement if requested
            if enhance:
                image = enhance_medical_image(image)
                logger.info("Image enhancement applied")
            
            # Make prediction
            result = self.predictor.predict(image)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Add metadata
            result['image_path'] = image_path
            result['processing_time'] = processing_time
            result['enhanced'] = enhance
            result['timestamp'] = datetime.now().isoformat()
            
            # Save visualization if requested
            if save_viz:
                viz_filename = f"prediction_{os.path.basename(image_path)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                viz_path = self.config.RESULTS_DIR / viz_filename
                save_prediction_visualization(image, result, str(viz_path))
                result['visualization_path'] = str(viz_path)
            
            # Save to database if requested
            if save_to_db:
                record_id = self.data_manager.save_prediction(
                    image_path, result, user_id=user_id, 
                    processing_time=processing_time
                )
                result['record_id'] = record_id
            
            # Print results
            self.print_prediction_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed for {image_path}: {str(e)}")
            raise
    
    def predict_batch(self, image_paths, enhance=False, save_viz=False, 
                     save_to_db=True, user_id=None):
        """
        Predict tumor presence in multiple images.
        
        Args:
            image_paths: List of image file paths
            enhance: Whether to apply image enhancement
            save_viz: Whether to save visualizations
            save_to_db: Whether to save results to database
            user_id: Optional user identifier
            
        Returns:
            List of prediction results
        """
        results = []
        logger.info(f"Processing {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths):
            try:
                logger.info(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
                result = self.predict_single_image(
                    image_path, enhance=enhance, save_viz=save_viz,
                    save_to_db=save_to_db, user_id=user_id
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {str(e)}")
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'success': False
                })
        
        # Save batch report
        if results:
            report_filename = f"batch_prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_path = self.config.RESULTS_DIR / report_filename
            save_prediction_report(results, str(report_path))
            logger.info(f"Batch report saved to {report_path}")
        
        # Print batch summary
        self.print_batch_summary(results)
        
        return results
    
    def predict_directory(self, directory_path, enhance=False, save_viz=False,
                         save_to_db=True, user_id=None):
        """
        Predict tumor presence in all images in a directory.
        
        Args:
            directory_path: Path to directory containing images
            enhance: Whether to apply image enhancement
            save_viz: Whether to save visualizations
            save_to_db: Whether to save results to database
            user_id: Optional user identifier
            
        Returns:
            List of prediction results
        """
        if not os.path.isdir(directory_path):
            raise ValueError(f"Directory not found: {directory_path}")
        
        # Find all image files
        from utils.image_utils import validate_image_format
        image_paths = []
        
        for filename in os.listdir(directory_path):
            filepath = os.path.join(directory_path, filename)
            if os.path.isfile(filepath) and validate_image_format(filepath):
                image_paths.append(filepath)
        
        if not image_paths:
            logger.warning(f"No valid image files found in {directory_path}")
            return []
        
        logger.info(f"Found {len(image_paths)} images in directory")
        return self.predict_batch(image_paths, enhance=enhance, save_viz=save_viz,
                                save_to_db=save_to_db, user_id=user_id)
    
    def print_prediction_result(self, result):
        """Print formatted prediction result."""
        if 'error' in result:
            print(f"\n❌ Error: {result['error']}")
            return
        
        print(f"\n📊 Prediction Results for: {os.path.basename(result['image_path'])}")
        print(f"{'='*60}")
        print(f"🔍 Predicted Class: {result['predicted_class']}")
        print(f"📈 Confidence: {result['confidence']:.1%}")
        print(f"⚠️ Risk Level: {result['risk_level']}")
        print(f"⏱️ Processing Time: {result['processing_time']:.3f} seconds")
        print(f"\n📋 Detailed Probabilities:")
        print(f"   • Non-Tumor: {result['probabilities']['non_tumor']:.1%}")
        print(f"   • Tumor: {result['probabilities']['tumor']:.1%}")
        
        # Risk level emoji
        risk_emoji = {
            'Low Risk': '🟢',
            'Low-Moderate Risk': '🟡',
            'Moderate Risk': '🟠',
            'High Risk': '🔴'
        }
        print(f"\n{risk_emoji.get(result['risk_level'], '⚪')} {result['risk_level']}")
    
    def print_batch_summary(self, results):
        """Print batch processing summary."""
        successful = [r for r in results if 'error' not in r]
        failed = [r for r in results if 'error' in r]
        
        print(f"\n📊 Batch Processing Summary")
        print(f"{'='*50}")
        print(f"📁 Total Images: {len(results)}")
        print(f"✅ Successful: {len(successful)}")
        print(f"❌ Failed: {len(failed)}")
        
        if successful:
            tumor_detected = sum(1 for r in successful if r.get('is_tumor', False))
            avg_confidence = sum(r.get('confidence', 0) for r in successful) / len(successful)
            
            print(f"\n🔍 Detection Results:")
            print(f"   • Tumor Cases: {tumor_detected}")
            print(f"   • Non-Tumor Cases: {len(successful) - tumor_detected}")
            print(f"   • Average Confidence: {avg_confidence:.1%}")
            
            # Risk level distribution
            risk_counts = {}
            for result in successful:
                risk = result.get('risk_level', 'Unknown')
                risk_counts[risk] = risk_counts.get(risk, 0) + 1
            
            print(f"\n⚠️ Risk Level Distribution:")
            for risk, count in sorted(risk_counts.items()):
                print(f"   • {risk}: {count}")

    def predict_for_dashboard(
        self,
        image_path: str,
        patches_info: List,
        include_heatmap: bool = True,
        include_analytics: bool = True
    ) -> Dict:
        """
        Generate predictions optimized for dashboard display.
        
        Args:
            image_path: Path to the image file
            patches_info: List of patch information from tiling
            include_heatmap: Whether to generate heatmap data
            include_analytics: Whether to include analytics
            
        Returns:
            Dashboard-formatted prediction results
        """
        try:
            start_time = datetime.now()
            
            # Make prediction on the image
            result = self.predict_single_image(
                image_path, 
                save_viz=False, 
                save_to_db=False
            )
            
            dashboard_result = {
                'image_path': image_path,
                'prediction': {
                    'class': result['predicted_class'],
                    'confidence': result['confidence'],
                    'risk_level': result['risk_level'],
                    'probabilities': result['probabilities']
                },
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'timestamp': datetime.now().isoformat()
            }
            
            # Add heatmap data if requested
            if include_heatmap and patches_info:
                heatmap_data = self._generate_heatmap_data(patches_info, result)
                dashboard_result['heatmap'] = heatmap_data
            
            # Add analytics if requested
            if include_analytics:
                analytics = self._generate_analytics_data(result, patches_info)
                dashboard_result['analytics'] = analytics
            
            return dashboard_result
            
        except Exception as e:
            logger.error(f"Dashboard prediction failed for {image_path}: {str(e)}")
            return {
                'image_path': image_path,
                'error': str(e),
                'success': False,
                'timestamp': datetime.now().isoformat()
            }
    
    def predict_patches_for_heatmap(
        self,
        patches: List[Tuple[np.ndarray, Dict]],
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Predict on patches for heatmap generation.
        
        Args:
            patches: List of (patch_image, patch_info) tuples
            batch_size: Batch size for processing
            
        Returns:
            List of prediction results for each patch
        """
        results = []
        
        for i in range(0, len(patches), batch_size):
            batch_patches = patches[i:i + batch_size]
            batch_images = [patch[0] for patch in batch_patches]
            batch_info = [patch[1] for patch in batch_patches]
            
            try:
                # Batch prediction
                batch_results = self.predictor.predict_batch(batch_images)
                
                for j, patch_result in enumerate(batch_results):
                    results.append({
                        'patch_info': batch_info[j],
                        'prediction': patch_result,
                        'coordinates': (batch_info[j].x, batch_info[j].y)
                    })
                    
            except Exception as e:
                logger.error(f"Batch prediction failed: {str(e)}")
                # Add error results for this batch
                for j in range(len(batch_patches)):
                    results.append({
                        'patch_info': batch_info[j],
                        'prediction': {'error': str(e)},
                        'coordinates': (batch_info[j].x, batch_info[j].y)
                    })
        
        return results
    
    def generate_real_time_predictions(
        self,
        image_path: str,
        update_callback: callable = None,
        max_patches: int = 1000
    ) -> Dict:
        """
        Generate predictions with real-time updates for dashboard.
        
        Args:
            image_path: Path to the image file
            update_callback: Function to call with updates
            max_patches: Maximum number of patches to process
            
        Returns:
            Final prediction results
        """
        try:
            from tiling import DashboardTiler
            
            # Initialize tiler
            tiler = DashboardTiler(patch_size=224, stride=224)
            
            # Extract patches for dashboard
            patches = tiler.extract_patches_for_dashboard(
                image_path, 
                max_patches=max_patches
            )
            
            if update_callback:
                update_callback({
                    'status': 'tiling_complete',
                    'total_patches': len(patches),
                    'timestamp': datetime.now().isoformat()
                })
            
            # Process patches in batches
            patch_results = []
            batch_size = 32
            
            for i in range(0, len(patches), batch_size):
                batch = patches[i:i + batch_size]
                batch_results = self.predict_patches_for_heatmap(batch)
                patch_results.extend(batch_results)
                
                if update_callback:
                    progress = min(100, (i + batch_size) / len(patches) * 100)
                    update_callback({
                        'status': 'processing',
                        'progress': progress,
                        'processed_patches': len(patch_results),
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Generate final result
            final_result = self._aggregate_patch_results(patch_results, image_path)
            
            if update_callback:
                update_callback({
                    'status': 'complete',
                    'result': final_result,
                    'timestamp': datetime.now().isoformat()
                })
            
            return final_result
            
        except Exception as e:
            error_result = {
                'error': str(e),
                'image_path': image_path,
                'timestamp': datetime.now().isoformat()
            }
            
            if update_callback:
                update_callback({
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
            
            return error_result
    
    def _generate_heatmap_data(
        self,
        patches_info: List,
        overall_result: Dict
    ) -> Dict:
        """Generate heatmap data for dashboard visualization."""
        # Simulate patch-level predictions for heatmap
        # In real implementation, this would use actual patch predictions
        
        heatmap_values = []
        coordinates = []
        
        for i, patch_info in enumerate(patches_info):
            # Generate deterministic patch prediction based on overall result and patch position
            base_prob = overall_result['probabilities']['tumor']
            
            # Create deterministic variation based on patch coordinates (no randomness)
            # This ensures same patches get same values every time
            coord_hash = hash((patch_info.x, patch_info.y)) % 1000
            deterministic_variation = (coord_hash / 1000.0 - 0.5) * 0.2  # Range: -0.1 to +0.1
            patch_prob = max(0, min(1, base_prob + deterministic_variation))
            
            heatmap_values.append(patch_prob)
            coordinates.append([patch_info.x, patch_info.y])
        
        return {
            'values': heatmap_values,
            'coordinates': coordinates,
            'colormap': 'hot',
            'interpolation': 'bilinear',
            'opacity': 0.7
        }
    
    def _generate_analytics_data(
        self,
        result: Dict,
        patches_info: List = None
    ) -> Dict:
        """Generate analytics data for dashboard."""
        analytics = {
            'summary': {
                'total_area_analyzed': len(patches_info) if patches_info else 1,
                'high_risk_regions': 0,
                'moderate_risk_regions': 0,
                'low_risk_regions': 0
            },
            'distribution': {
                'tumor_probability': result['probabilities']['tumor'],
                'confidence_level': result['confidence'],
                'risk_score': self._calculate_risk_score(result)
            },
            'recommendations': self._generate_recommendations(result)
        }
        
        # Simulate region analysis if patches available
        if patches_info:
            tumor_prob = result['probabilities']['tumor']
            
            for _ in patches_info:
                if tumor_prob > 0.7:
                    analytics['summary']['high_risk_regions'] += 1
                elif tumor_prob > 0.4:
                    analytics['summary']['moderate_risk_regions'] += 1
                else:
                    analytics['summary']['low_risk_regions'] += 1
        
        return analytics
    
    def _calculate_risk_score(self, result: Dict) -> float:
        """Calculate numerical risk score for analytics."""
        tumor_prob = result['probabilities']['tumor']
        confidence = result['confidence']
        
        # Weighted risk score
        risk_score = (tumor_prob * 0.7) + (confidence * 0.3)
        return round(risk_score, 3)
    
    def _generate_recommendations(self, result: Dict) -> List[str]:
        """Generate recommendations based on prediction."""
        recommendations = []
        
        tumor_prob = result['probabilities']['tumor']
        confidence = result['confidence']
        
        if tumor_prob > 0.7:
            recommendations.append("High tumor probability detected - recommend urgent consultation")
            recommendations.append("Consider additional imaging studies")
        elif tumor_prob > 0.4:
            recommendations.append("Moderate tumor probability - follow-up recommended")
            recommendations.append("Monitor for changes in subsequent scans")
        else:
            recommendations.append("Low tumor probability - routine follow-up")
        
        if confidence < 0.6:
            recommendations.append("Low confidence prediction - consider manual review")
        
        return recommendations
    
    def _aggregate_patch_results(
        self,
        patch_results: List[Dict],
        image_path: str
    ) -> Dict:
        """Aggregate patch-level results into overall prediction."""
        valid_results = [r for r in patch_results if 'error' not in r['prediction']]
        
        if not valid_results:
            return {
                'error': 'No valid patch predictions',
                'image_path': image_path
            }
        
        # Calculate overall statistics
        tumor_probs = [r['prediction']['probabilities']['tumor'] for r in valid_results]
        mean_tumor_prob = np.mean(tumor_probs)
        max_tumor_prob = np.max(tumor_probs)
        
        # Determine overall classification
        overall_class = 'tumor' if mean_tumor_prob > 0.5 else 'non_tumor'
        confidence = max(tumor_probs) if overall_class == 'tumor' else (1 - max_tumor_prob)
        
        # Generate risk level
        if mean_tumor_prob > 0.7:
            risk_level = 'High Risk'
        elif mean_tumor_prob > 0.4:
            risk_level = 'Moderate Risk'
        else:
            risk_level = 'Low Risk'
        
        return {
            'image_path': image_path,
            'predicted_class': overall_class,
            'confidence': confidence,
            'risk_level': risk_level,
            'probabilities': {
                'tumor': mean_tumor_prob,
                'non_tumor': 1 - mean_tumor_prob
            },
            'patch_statistics': {
                'total_patches': len(valid_results),
                'mean_tumor_prob': mean_tumor_prob,
                'max_tumor_prob': max_tumor_prob,
                'high_risk_patches': sum(1 for p in tumor_probs if p > 0.7),
                'moderate_risk_patches': sum(1 for p in tumor_probs if 0.4 < p <= 0.7),
                'low_risk_patches': sum(1 for p in tumor_probs if p <= 0.4)
            },
            'success': True,
            'timestamp': datetime.now().isoformat()
        }

def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='Tumor Prediction Service')
    parser.add_argument('input', help='Path to image file or directory')
    parser.add_argument('--model', help='Path to model file')
    parser.add_argument('--enhance', action='store_true', help='Apply image enhancement')
    parser.add_argument('--save-viz', action='store_true', help='Save prediction visualizations')
    parser.add_argument('--no-db', action='store_true', help='Don\'t save to database')
    parser.add_argument('--user-id', help='User identifier')
    parser.add_argument('--config-env', default='development', help='Configuration environment')
    
    args = parser.parse_args()
    
    try:
        # Initialize service
        service = TumorPredictionService(
            model_path=args.model,
            config_env=args.config_env
        )
        
        # Determine input type and process
        if os.path.isfile(args.input):
            # Single image
            service.predict_single_image(
                args.input,
                enhance=args.enhance,
                save_viz=args.save_viz,
                save_to_db=not args.no_db,
                user_id=args.user_id
            )
        elif os.path.isdir(args.input):
            # Directory of images
            service.predict_directory(
                args.input,
                enhance=args.enhance,
                save_viz=args.save_viz,
                save_to_db=not args.no_db,
                user_id=args.user_id
            )
        else:
            print(f"Error: {args.input} is not a valid file or directory")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Prediction service failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())