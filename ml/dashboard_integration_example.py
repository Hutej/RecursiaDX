#!/usr/bin/env python3
"""
Integration example showing how to use the new matplotlib heatmap features
in your RecursiaDx AI analytics dashboard.
"""

import os
import sys
import numpy as np
from PIL import Image

# Add the ML module to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def simulate_pipeline_processing():
    """Simulate the HistopathologyPipeline processing results."""
    # Simulate patch predictions across an image
    np.random.seed(42)  # For reproducible results
    
    # Generate random patch coordinates and predictions
    num_patches = 150
    image_size = (2048, 2048)
    
    predictions = []
    for i in range(num_patches):
        # Random patch location
        x = np.random.randint(0, image_size[0] - 224)
        y = np.random.randint(0, image_size[1] - 224)
        
        # Simulate tumor probability (higher in certain regions)
        center_x, center_y = image_size[0] // 2, image_size[1] // 2
        distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Create tumor hotspots
        hotspot_centers = [(600, 800), (1400, 1200), (300, 1600)]
        tumor_prob = 0.1  # Base probability
        
        for hx, hy in hotspot_centers:
            distance = np.sqrt((x - hx)**2 + (y - hy)**2)
            if distance < 300:  # Within hotspot radius
                tumor_prob += 0.7 * np.exp(-distance / 100)
        
        # Add some randomness
        tumor_prob += np.random.normal(0, 0.1)
        tumor_prob = np.clip(tumor_prob, 0, 1)
        
        # Simulate confidence (inversely related to uncertainty)
        confidence = 0.6 + 0.4 * (1 - abs(tumor_prob - 0.5) * 2)
        confidence += np.random.normal(0, 0.05)
        confidence = np.clip(confidence, 0.3, 1.0)
        
        predictions.append({
            'x': x,
            'y': y,
            'tumor_prob': tumor_prob,
            'confidence': confidence,
            'predicted_class': 'tumor' if tumor_prob > 0.5 else 'normal'
        })
    
    return {
        'predictions': predictions,
        'image_size': image_size,
        'processing_time': 45.2,
        'total_patches': num_patches
    }

def demonstrate_dashboard_integration():
    """Demonstrate how to integrate matplotlib heatmaps in your dashboard."""
    print("🎯 Dashboard Integration Example")
    print("=" * 50)
    
    # Create mock pipeline class for demonstration
    class MockHistopathologyPipeline:
        def __init__(self):
            self.verbose = True
        
        def process_image(self, image_path, **kwargs):
            return simulate_pipeline_processing()
        
        def _create_heatmap_array(self, results, heatmap_type):
            """Create heatmap array from processing results."""
            predictions = results.get('predictions', [])
            
            # Extract coordinates and values
            coords = []
            values = []
            
            for pred in predictions:
                coords.append([pred['x'], pred['y']])
                
                if heatmap_type == 'tumor_probability':
                    values.append(pred.get('tumor_prob', 0))
                elif heatmap_type == 'confidence':
                    values.append(pred.get('confidence', 0))
                elif heatmap_type == 'risk_score':
                    tumor_prob = pred.get('tumor_prob', 0)
                    confidence = pred.get('confidence', 0.5)
                    risk_score = (tumor_prob * 0.8) + (confidence * 0.2)
                    values.append(risk_score)
                else:
                    values.append(pred.get('confidence', 0))
            
            coords = np.array(coords)
            values = np.array(values)
            
            # Create grid
            img_size = results.get('image_size', (2048, 2048))
            grid_size = (128, 128)  # Dashboard resolution
            
            # Simple grid assignment (without scipy interpolation)
            heatmap = np.zeros(grid_size)
            counts = np.zeros(grid_size)
            
            for coord, value in zip(coords, values):
                x_idx = int((coord[0] / img_size[0]) * grid_size[1])
                y_idx = int((coord[1] / img_size[1]) * grid_size[0])
                x_idx = max(0, min(x_idx, grid_size[1] - 1))
                y_idx = max(0, min(y_idx, grid_size[0] - 1))
                heatmap[y_idx, x_idx] += value
                counts[y_idx, x_idx] += 1
            
            # Average values where multiple patches exist
            mask = counts > 0
            heatmap[mask] /= counts[mask]
            
            return heatmap
        
        def _get_colorbar_label(self, heatmap_type):
            labels = {
                'tumor_probability': 'Tumor Probability',
                'confidence': 'Prediction Confidence',
                'risk_score': 'Risk Score'
            }
            return labels.get(heatmap_type, 'Value')
        
        def _extract_analytics_data(self, results):
            predictions = results.get('predictions', [])
            
            # Calculate statistics
            tumor_probs = [p['tumor_prob'] for p in predictions]
            confidences = [p['confidence'] for p in predictions]
            
            high_risk = sum(1 for p in tumor_probs if p > 0.7)
            moderate_risk = sum(1 for p in tumor_probs if 0.3 < p <= 0.7)
            low_risk = sum(1 for p in tumor_probs if p <= 0.3)
            
            return {
                'total_patches': len(predictions),
                'mean_tumor_probability': np.mean(tumor_probs),
                'mean_confidence': np.mean(confidences),
                'risk_distribution': {
                    'high_risk': high_risk,
                    'moderate_risk': moderate_risk,
                    'low_risk': low_risk
                },
                'max_tumor_probability': max(tumor_probs),
                'processing_time': results.get('processing_time', 0)
            }
        
        def generate_matplotlib_heatmap(self, image_path, output_path=None, 
                                      heatmap_type='tumor_probability', 
                                      colormap='hot', figsize=(12, 8),
                                      show_colorbar=True, show_overlay=False,
                                      overlay_alpha=0.6, dpi=150, return_fig=False):
            """Generate matplotlib-style heatmap (simplified version)."""
            import matplotlib.pyplot as plt
            
            try:
                # Process image
                results = self.process_image(image_path)
                
                # Generate heatmap data
                heatmap_array = self._create_heatmap_array(results, heatmap_type)
                
                # Create matplotlib figure
                fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
                
                # Show heatmap
                im = ax.imshow(heatmap_array, cmap=colormap, interpolation='bilinear')
                
                # Customize plot
                ax.set_title(f'{heatmap_type.replace("_", " ").title()} Heatmap', 
                            fontsize=16, fontweight='bold')
                ax.axis('off')
                
                # Add colorbar
                if show_colorbar:
                    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
                    cbar.set_label(self._get_colorbar_label(heatmap_type), 
                                  fontsize=12, fontweight='bold')
                    cbar.ax.tick_params(labelsize=10)
                
                # Adjust layout
                plt.tight_layout()
                
                # Save if output path provided
                if output_path:
                    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                               facecolor='white', edgecolor='none')
                    print(f"✅ Heatmap saved: {output_path}")
                
                # Prepare return data
                result_data = {
                    'success': True,
                    'heatmap_info': {
                        'type': heatmap_type,
                        'colormap': colormap,
                        'shape': heatmap_array.shape,
                        'min_value': float(np.min(heatmap_array)),
                        'max_value': float(np.max(heatmap_array)),
                        'mean_value': float(np.mean(heatmap_array))
                    },
                    'analytics': self._extract_analytics_data(results)
                }
                
                if output_path:
                    result_data['output_path'] = output_path
                
                if return_fig:
                    result_data['figure'] = fig
                else:
                    plt.close(fig)
                
                return result_data
                
            except Exception as e:
                return {'success': False, 'error': str(e)}
    
    # Initialize mock pipeline
    pipeline = MockHistopathologyPipeline()
    
    # Create output directory
    output_dir = "dashboard_integration_example"
    os.makedirs(output_dir, exist_ok=True)
    
    print("📊 Generating dashboard-ready heatmaps...")
    
    # 1. Generate different heatmap types for dashboard
    heatmap_types = ['tumor_probability', 'confidence', 'risk_score']
    colormaps = ['hot', 'viridis', 'plasma']
    
    results = {}
    
    for i, (htype, cmap) in enumerate(zip(heatmap_types, colormaps)):
        print(f"   {i+1}. {htype.replace('_', ' ').title()} ({cmap})")
        
        output_path = os.path.join(output_dir, f"dashboard_{htype}_{cmap}.png")
        
        result = pipeline.generate_matplotlib_heatmap(
            image_path="mock_wsi_image.svs",  # Mock path
            output_path=output_path,
            heatmap_type=htype,
            colormap=cmap,
            figsize=(10, 8),
            show_colorbar=True,
            dpi=150
        )
        
        if result['success']:
            results[htype] = result
            print(f"      ✅ Generated: {os.path.basename(output_path)}")
            print(f"         - Min: {result['heatmap_info']['min_value']:.3f}")
            print(f"         - Max: {result['heatmap_info']['max_value']:.3f}")
            print(f"         - Mean: {result['heatmap_info']['mean_value']:.3f}")
        else:
            print(f"      ❌ Failed: {result['error']}")
    
    # 2. Generate summary analytics
    print(f"\n📈 Analytics Summary:")
    if 'tumor_probability' in results:
        analytics = results['tumor_probability']['analytics']
        print(f"   • Total patches analyzed: {analytics['total_patches']}")
        print(f"   • Mean tumor probability: {analytics['mean_tumor_probability']:.3f}")
        print(f"   • Mean confidence: {analytics['mean_confidence']:.3f}")
        print(f"   • Max tumor probability: {analytics['max_tumor_probability']:.3f}")
        print(f"   • Processing time: {analytics['processing_time']:.1f}s")
        
        risk_dist = analytics['risk_distribution']
        print(f"   • Risk distribution:")
        print(f"     - High risk regions: {risk_dist['high_risk']}")
        print(f"     - Moderate risk regions: {risk_dist['moderate_risk']}")
        print(f"     - Low risk regions: {risk_dist['low_risk']}")
    
    print(f"\n✅ Dashboard integration example completed!")
    print(f"📂 Check outputs in: {os.path.abspath(output_dir)}")
    
    # 3. Show integration code example
    print(f"\n💡 Dashboard Integration Code:")
    print("""
# In your dashboard backend (Flask/FastAPI):

from pipeline import HistopathologyPipeline
import base64
from io import BytesIO

@app.route('/api/generate-heatmap', methods=['POST'])
def generate_heatmap():
    image_path = request.json['image_path']
    heatmap_type = request.json.get('heatmap_type', 'tumor_probability')
    colormap = request.json.get('colormap', 'hot')
    
    pipeline = HistopathologyPipeline()
    
    # Generate heatmap
    result = pipeline.generate_matplotlib_heatmap(
        image_path=image_path,
        heatmap_type=heatmap_type,
        colormap=colormap,
        return_fig=True,
        show_colorbar=True
    )
    
    if result['success']:
        # Convert to base64 for web display
        fig = result['figure']
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return {
            'success': True,
            'heatmap_image': f"data:image/png;base64,{image_base64}",
            'analytics': result['analytics'],
            'heatmap_info': result['heatmap_info']
        }
    else:
        return {'success': False, 'error': result['error']}

# In your frontend (React/Vue):
const generateHeatmap = async (imagePath, type = 'tumor_probability') => {
    const response = await fetch('/api/generate-heatmap', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            image_path: imagePath,
            heatmap_type: type,
            colormap: 'hot'
        })
    });
    
    const data = await response.json();
    
    if (data.success) {
        // Display heatmap in image element
        document.getElementById('heatmap').src = data.heatmap_image;
        
        // Update analytics display
        updateAnalytics(data.analytics);
    }
};
    """)

def main():
    """Main demonstration function."""
    try:
        demonstrate_dashboard_integration()
        return 0
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())