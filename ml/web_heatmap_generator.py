#!/usr/bin/env python3
"""
Web-optimized heatmap generator for RecursiaDx UI
This script is called by the Node.js backend to generate matplotlib heatmaps
"""

import argparse
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image
import base64
from io import BytesIO

def create_sample_heatmap_data(image_path, heatmap_type='tumor_probability'):
    """Create sample heatmap data based on image."""
    try:
        # Load image to get dimensions
        img = Image.open(image_path)
        width, height = img.size
        
        # Create heatmap grid (smaller for demo)
        grid_size = (min(64, height//32), min(64, width//32))
        
        # Generate sample data based on type
        np.random.seed(42)  # Consistent results
        
        if heatmap_type == 'tumor_probability':
            # Create tumor probability with hotspots
            x = np.linspace(-2, 2, grid_size[1])
            y = np.linspace(-2, 2, grid_size[0])
            X, Y = np.meshgrid(x, y)
            
            # Multiple gaussian peaks for tumor regions
            heatmap = np.zeros_like(X)
            centers = [(-1, -1), (1, 0.5), (0, 1.2)]
            for cx, cy in centers:
                intensity = np.random.uniform(0.6, 0.9)
                sigma = np.random.uniform(0.4, 0.8)
                peak = intensity * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
                heatmap += peak
            
            # Add noise
            heatmap += np.random.normal(0, 0.05, heatmap.shape)
            
        elif heatmap_type == 'confidence':
            # Create confidence map
            heatmap = np.random.uniform(0.5, 1.0, grid_size)
            # Add some structure
            center_y, center_x = grid_size[0]//2, grid_size[1]//2
            Y, X = np.ogrid[:grid_size[0], :grid_size[1]]
            dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            confidence_factor = 1 - (dist_from_center / max_dist) * 0.3
            heatmap *= confidence_factor
            
        elif heatmap_type == 'risk_score':
            # Create risk score combining probability and confidence
            prob_map = np.random.uniform(0.1, 0.8, grid_size)
            conf_map = np.random.uniform(0.6, 1.0, grid_size)
            heatmap = prob_map * 0.7 + conf_map * 0.3
            
        else:
            # Default to random data
            heatmap = np.random.uniform(0, 1, grid_size)
        
        # Ensure valid range
        heatmap = np.clip(heatmap, 0, 1)
        
        return heatmap, img.size
        
    except Exception as e:
        print(f"Error creating heatmap data: {e}", file=sys.stderr)
        return None, None

def generate_heatmap(image_path, output_path, heatmap_type='tumor_probability', 
                    colormap='hot', format_type='web'):
    """Generate matplotlib heatmap."""
    try:
        print(f"🎨 Generating {heatmap_type} heatmap with {colormap} colormap")
        
        # Create heatmap data
        heatmap_data, original_size = create_sample_heatmap_data(image_path, heatmap_type)
        
        if heatmap_data is None:
            return False
        
        # Create matplotlib figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=100)
        
        # Generate heatmap
        im = ax.imshow(heatmap_data, cmap=colormap, interpolation='bilinear', 
                      aspect='auto', origin='lower')
        
        # Customize plot
        title_map = {
            'tumor_probability': 'Tumor Probability Heatmap',
            'confidence': 'Prediction Confidence Heatmap', 
            'risk_score': 'Risk Score Heatmap'
        }
        ax.set_title(title_map.get(heatmap_type, 'Analysis Heatmap'), 
                    fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
        
        label_map = {
            'tumor_probability': 'Tumor Probability',
            'confidence': 'Confidence Level',
            'risk_score': 'Risk Score'
        }
        cbar.set_label(label_map.get(heatmap_type, 'Value'), 
                      fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        # Calculate analytics
        analytics = {
            'min_value': float(np.min(heatmap_data)),
            'max_value': float(np.max(heatmap_data)),
            'mean_value': float(np.mean(heatmap_data)),
            'std_value': float(np.std(heatmap_data)),
            'shape': heatmap_data.shape,
            'hotspots': int(np.sum(heatmap_data > 0.7)),
            'total_pixels': int(heatmap_data.size)
        }
        
        # Output analytics as JSON for parsing
        print("ANALYTICS_JSON:" + json.dumps(analytics) + "END_ANALYTICS")
        
        print(f"✅ Heatmap saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error generating heatmap: {e}", file=sys.stderr)
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate matplotlib heatmap')
    parser.add_argument('--image', required=True, help='Input image path')
    parser.add_argument('--output', required=True, help='Output heatmap path')
    parser.add_argument('--type', default='tumor_probability', 
                       choices=['tumor_probability', 'confidence', 'risk_score'],
                       help='Heatmap type')
    parser.add_argument('--colormap', default='hot', help='Matplotlib colormap')
    parser.add_argument('--format', default='web', help='Output format')
    
    args = parser.parse_args()
    
    try:
        # Validate input
        if not os.path.exists(args.image):
            print(f"❌ Input image not found: {args.image}", file=sys.stderr)
            sys.exit(1)
        
        # Create output directory
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        # Generate heatmap
        success = generate_heatmap(
            args.image, 
            args.output, 
            args.type, 
            args.colormap, 
            args.format
        )
        
        if success:
            print(f"🎯 Heatmap generation completed successfully")
            sys.exit(0)
        else:
            print(f"❌ Heatmap generation failed", file=sys.stderr)
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Script error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()