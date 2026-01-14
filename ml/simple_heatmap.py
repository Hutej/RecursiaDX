#!/usr/bin/env python3
"""
Simple Heatmap Generator - Direct Implementation
Creates actual matplotlib heatmaps that you can see immediately
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os

def create_sample_data():
    """Create sample tumor probability data."""
    # Create a 64x64 grid representing an image analysis
    size = (64, 64)
    heatmap = np.zeros(size)
    
    # Add tumor hotspots with realistic patterns
    hotspots = [
        {'center': (15, 20), 'intensity': 0.9, 'radius': 8},
        {'center': (45, 35), 'intensity': 0.8, 'radius': 6},
        {'center': (30, 50), 'intensity': 0.7, 'radius': 5},
        {'center': (10, 45), 'intensity': 0.6, 'radius': 4},
    ]
    
    # Create coordinate grids
    y_coords, x_coords = np.ogrid[:size[0], :size[1]]
    
    # Add each hotspot
    for spot in hotspots:
        center_y, center_x = spot['center']
        intensity = spot['intensity']
        radius = spot['radius']
        
        # Calculate distance from center
        distance = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        # Create gaussian-like distribution
        mask = distance <= radius * 2
        gaussian = intensity * np.exp(-(distance**2) / (2 * (radius/2)**2))
        heatmap[mask] += gaussian[mask]
    
    # Add some background noise
    background_noise = np.random.normal(0, 0.05, size)
    heatmap += background_noise
    
    # Ensure values are in [0, 1] range
    heatmap = np.clip(heatmap, 0, 1)
    
    return heatmap

def create_tissue_background():
    """Create a tissue-like background image."""
    size = (512, 512)
    
    # Create base tissue color
    img = Image.new('RGB', size, color=(230, 200, 180))  # Light pink tissue color
    draw = ImageDraw.Draw(img)
    
    # Add tissue texture
    for _ in range(100):
        x = np.random.randint(0, size[0])
        y = np.random.randint(0, size[1])
        radius = np.random.randint(2, 8)
        
        # Vary the tissue color slightly
        color = (
            np.random.randint(200, 240),
            np.random.randint(170, 210),
            np.random.randint(160, 200)
        )
        
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
    
    return np.array(img)

def generate_heatmap(heatmap_type="tumor_probability", colormap="hot", save_path=None):
    """Generate a matplotlib heatmap visualization."""
    
    print(f"🎨 Generating {heatmap_type} heatmap with {colormap} colormap...")
    
    # Create sample data
    heatmap_data = create_sample_data()
    tissue_img = create_tissue_background()
    
    # Resize tissue image to match heatmap
    tissue_resized = np.array(Image.fromarray(tissue_img).resize(heatmap_data.shape[::-1]))
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{heatmap_type.replace("_", " ").title()} Analysis', fontsize=16, fontweight='bold')
    
    # 1. Heatmap only
    im1 = axes[0].imshow(heatmap_data, cmap=colormap, interpolation='bilinear')
    axes[0].set_title('Heatmap Only', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.8, aspect=20)
    cbar1.set_label('Tumor Probability', fontsize=12, fontweight='bold')
    cbar1.ax.tick_params(labelsize=10)
    
    # 2. Tissue background
    axes[1].imshow(tissue_resized)
    axes[1].set_title('Original Tissue', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # 3. Overlay heatmap on tissue
    axes[2].imshow(tissue_resized, alpha=1.0)
    im3 = axes[2].imshow(heatmap_data, cmap=colormap, alpha=0.6, interpolation='bilinear')
    axes[2].set_title('Heatmap Overlay', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # Add colorbar for overlay
    cbar3 = plt.colorbar(im3, ax=axes[2], shrink=0.8, aspect=20)
    cbar3.set_label('Tumor Probability', fontsize=12, fontweight='bold')
    cbar3.ax.tick_params(labelsize=10)
    
    # Add statistics text
    mean_val = np.mean(heatmap_data)
    max_val = np.max(heatmap_data)
    std_val = np.std(heatmap_data)
    
    stats_text = f'Statistics:\nMean: {mean_val:.3f}\nMax: {max_val:.3f}\nStd: {std_val:.3f}'
    axes[0].text(0.02, 0.98, stats_text, transform=axes[0].transAxes, 
                fontsize=10, fontweight='bold', color='white',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                verticalalignment='top')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✅ Heatmap saved: {save_path}")
    
    # Show the plot
    plt.show()
    
    return {
        'heatmap_data': heatmap_data,
        'tissue_image': tissue_resized,
        'statistics': {
            'mean': mean_val,
            'max': max_val,
            'std': std_val,
            'total_pixels': heatmap_data.size,
            'high_risk_pixels': np.sum(heatmap_data > 0.7),
            'moderate_risk_pixels': np.sum((heatmap_data > 0.3) & (heatmap_data <= 0.7)),
            'low_risk_pixels': np.sum(heatmap_data <= 0.3)
        }
    }

def generate_multiple_colormaps():
    """Generate heatmaps with different colormaps."""
    print("🎨 Generating multiple colormap examples...")
    
    # Create sample data once
    heatmap_data = create_sample_data()
    
    # Different colormaps
    colormaps = ['hot', 'viridis', 'plasma', 'inferno', 'jet', 'coolwarm']
    
    # Create output directory
    output_dir = "visible_heatmaps"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, cmap in enumerate(colormaps):
        print(f"   {i+1}. Generating {cmap} heatmap...")
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Generate heatmap
        im = ax.imshow(heatmap_data, cmap=cmap, interpolation='bilinear')
        ax.set_title(f'Tumor Probability Heatmap - {cmap.title()}', 
                    fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label('Tumor Probability', fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)
        
        # Add statistics
        mean_val = np.mean(heatmap_data)
        max_val = np.max(heatmap_data)
        
        stats_text = f'Mean: {mean_val:.3f}\nMax: {max_val:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               fontsize=12, fontweight='bold', color='white',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
               verticalalignment='top')
        
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(output_dir, f"heatmap_{cmap}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"      ✅ Saved: {output_path}")
    
    print(f"\n✅ All heatmaps saved to: {os.path.abspath(output_dir)}")

def generate_contour_heatmap():
    """Generate contour-style heatmap."""
    print("🎨 Generating contour-style heatmap...")
    
    heatmap_data = create_sample_data()
    tissue_img = create_tissue_background()
    tissue_resized = np.array(Image.fromarray(tissue_img).resize(heatmap_data.shape[::-1]))
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Contour-Style Heatmap Analysis', fontsize=16, fontweight='bold')
    
    # 1. Filled contours
    contour_filled = axes[0].contourf(heatmap_data, levels=20, cmap='hot', alpha=0.8)
    axes[0].set_title('Filled Contours', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Add colorbar
    cbar1 = plt.colorbar(contour_filled, ax=axes[0], shrink=0.8)
    cbar1.set_label('Tumor Probability', fontsize=12, fontweight='bold')
    
    # 2. Contours on tissue
    axes[1].imshow(tissue_resized, alpha=0.7)
    contour_lines = axes[1].contour(heatmap_data, levels=10, colors='red', alpha=0.8, linewidths=2)
    axes[1].clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
    axes[1].set_title('Contour Lines on Tissue', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Save
    output_path = "visible_heatmaps/contour_heatmap.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"✅ Contour heatmap saved: {output_path}")

def main():
    """Main function to demonstrate heatmap generation."""
    print("🎯 Simple Heatmap Generator - Direct Implementation")
    print("=" * 60)
    
    try:
        # 1. Generate basic heatmap
        print("\n1. Generating basic heatmap...")
        result = generate_heatmap(
            heatmap_type="tumor_probability",
            colormap="hot",
            save_path="visible_heatmaps/basic_heatmap.png"
        )
        
        print(f"\n📊 Analysis Results:")
        stats = result['statistics']
        print(f"   • Mean tumor probability: {stats['mean']:.3f}")
        print(f"   • Maximum probability: {stats['max']:.3f}")
        print(f"   • Standard deviation: {stats['std']:.3f}")
        print(f"   • High risk pixels: {stats['high_risk_pixels']}")
        print(f"   • Moderate risk pixels: {stats['moderate_risk_pixels']}")
        print(f"   • Low risk pixels: {stats['low_risk_pixels']}")
        
        # 2. Generate multiple colormaps
        print("\n2. Generating multiple colormap examples...")
        generate_multiple_colormaps()
        
        # 3. Generate contour heatmap
        print("\n3. Generating contour-style heatmap...")
        generate_contour_heatmap()
        
        print("\n✅ All heatmaps generated successfully!")
        print(f"📂 Check the 'visible_heatmaps' directory for all outputs")
        
        # Show usage example
        print("\n💡 Usage in your dashboard:")
        print("""
# Import and use directly:
from simple_heatmap import generate_heatmap

# Generate heatmap
result = generate_heatmap(
    heatmap_type="tumor_probability",
    colormap="hot",
    save_path="output.png"
)

# Access data
heatmap_array = result['heatmap_data']
statistics = result['statistics']
        """)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())