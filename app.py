#!/usr/bin/env python3
"""
TRANSPARENT OBJECT DETECTOR - SINGLE FILE APP
Just run: python app.py
Put your image as 'input.jpg' or 'input.png' in this folder
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage import morphology, measure, transform
from skimage.feature import local_binary_pattern
import os
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# AUTO-DETECT INPUT IMAGE - Just put any image named 'input.*' in the folder
# ==============================================================================
def find_input_image():
    """Find input image automatically"""
    possible_names = ['input.jpg', 'input.png', 'input.jpeg', 'test.jpg', 'test.png']
    for name in possible_names:
        if os.path.exists(name):
            return name
    return None

class GlassDetector:
    def __init__(self):
        self.name = "Advanced Glass Detector"
        
    def detect(self, image):
        """Complete detection pipeline"""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 1. Edge Analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = cv2.blur(edges.astype(np.float32), (15, 15))
        
        # 2. Gradient Analysis
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # 3. Texture Analysis
        lbp = local_binary_pattern(gray, 8, 1, method='uniform')
        texture_var = ndimage.generic_filter(lbp, np.var, size=15)
        
        # 4. Gabor Filters
        gabor_response = self.apply_gabor(gray)
        
        # 5. Phase Congruency
        phase = self.phase_congruency(gray)
        
        # 6. Combine all features
        combined = (
            edge_density/255 * 0.2 +
            grad_mag/grad_mag.max() * 0.2 +
            texture_var/texture_var.max() * 0.2 +
            gabor_response/255 * 0.2 +
            phase/255 * 0.2
        )
        
        # 7. Create mask
        combined = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, mask = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 8. Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 9. Find objects
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objects = []
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                confidence = min(area / 10000, 1.0) * 0.7 + 0.3
                objects.append({
                    'id': i,
                    'bbox': (x, y, w, h),
                    'area': area,
                    'confidence': confidence,
                    'contour': cnt
                })
        
        # 10. Generate depth map
        depth = self.estimate_depth(mask, grad_mag)
        
        return {
            'mask': mask,
            'edges': edges,
            'gradient': grad_mag,
            'texture': texture_var,
            'gabor': gabor_response,
            'phase': phase,
            'combined': combined,
            'depth': depth,
            'objects': objects
        }
    
    def apply_gabor(self, image):
        """Apply Gabor filters"""
        responses = []
        for theta in np.arange(0, np.pi, np.pi/4):
            kernel = cv2.getGaborKernel((21, 21), 4.0, theta, 10.0, 0.5, 0)
            filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
            responses.append(filtered)
        return np.mean(responses, axis=0).astype(np.uint8)
    
    def phase_congruency(self, image):
        """Simplified phase congruency"""
        scales = 3
        pc = np.zeros_like(image, dtype=np.float32)
        for s in range(scales):
            sigma = 2 ** s
            blur = cv2.GaussianBlur(image, (0, 0), sigma)
            dog = image.astype(np.float32) - blur.astype(np.float32)
            pc += np.abs(dog)
        return cv2.normalize(pc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    def estimate_depth(self, mask, gradient):
        """Estimate depth from gradients"""
        depth = gradient * (mask > 0)
        depth = cv2.GaussianBlur(depth, (15, 15), 5)
        return cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def create_visualization(image, results):
    """Create matplotlib visualization with all results"""
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('🔍 TRANSPARENT OBJECT DETECTION - COMPLETE ANALYSIS', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Create grid
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.2)
    
    # Original image with detections
    ax1 = fig.add_subplot(gs[0, :2])
    img_display = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax1.imshow(img_display)
    ax1.set_title('📸 Input Image with Detections', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Draw bounding boxes
    for obj in results['objects']:
        x, y, w, h = obj['bbox']
        conf = obj['confidence']
        color = 'lime' if conf > 0.7 else 'yellow'
        rect = Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
        ax1.add_patch(rect)
        ax1.text(x, y-5, f'{conf:.0%}', color=color, fontsize=10, 
                fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor='black', alpha=0.7))
    
    # Detection mask
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(results['mask'], cmap='hot')
    ax2.set_title('🎯 Detection Mask', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Depth map
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.imshow(results['depth'], cmap='jet')
    ax3.set_title('📏 Depth Estimation', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # Edge detection
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(results['edges'], cmap='gray')
    ax4.set_title('🔲 Edge Detection', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # Gradient magnitude
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(results['gradient'], cmap='viridis')
    ax5.set_title('📊 Gradient Analysis', fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    # Texture variance
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(results['texture'], cmap='copper')
    ax6.set_title('🔍 Texture Variance', fontsize=12, fontweight='bold')
    ax6.axis('off')
    
    # Gabor response
    ax7 = fig.add_subplot(gs[1, 3])
    ax7.imshow(results['gabor'], cmap='magma')
    ax7.set_title('🌊 Gabor Filters', fontsize=12, fontweight='bold')
    ax7.axis('off')
    
    # Phase congruency
    ax8 = fig.add_subplot(gs[2, 0])
    ax8.imshow(results['phase'], cmap='hsv')
    ax8.set_title('🎭 Phase Congruency', fontsize=12, fontweight='bold')
    ax8.axis('off')
    
    # Combined features
    ax9 = fig.add_subplot(gs[2, 1])
    ax9.imshow(results['combined'], cmap='plasma')
    ax9.set_title('🔀 Combined Features', fontsize=12, fontweight='bold')
    ax9.axis('off')
    
    # Statistics panel
    ax10 = fig.add_subplot(gs[2, 2:])
    ax10.axis('off')
    
    # Add statistics text
    stats_text = f"""
📊 DETECTION STATISTICS
{'='*40}
🎯 Objects Detected: {len(results['objects'])}
📐 Total Glass Area: {sum(obj['area'] for obj in results['objects']):,} pixels
🏆 Best Confidence: {max(obj['confidence'] for obj in results['objects']) if results['objects'] else 0:.1%}

📦 DETECTED OBJECTS:
{'='*40}"""
    
    for i, obj in enumerate(results['objects'][:5], 1):  # Show top 5
        stats_text += f"""
Object #{i}:
  • Confidence: {obj['confidence']:.1%}
  • Area: {obj['area']:,} px
  • Location: ({obj['bbox'][0]}, {obj['bbox'][1]})
  • Size: {obj['bbox'][2]}×{obj['bbox'][3]} px"""
    
    if not results['objects']:
        stats_text += "\n\n⚠️ No transparent objects detected"
    
    ax10.text(0.05, 0.95, stats_text, transform=ax10.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
             color='white', family='monospace')
    
    # Add timestamp
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    fig.text(0.99, 0.01, f'Generated: {timestamp}', 
            ha='right', fontsize=9, alpha=0.7)
    
    plt.tight_layout()
    return fig

def main():
    """Main application"""
    print("\n" + "="*70)
    print("🔬 TRANSPARENT OBJECT DETECTOR")
    print("="*70)
    print("Advanced Glass Detection with Physics-Based Analysis")
    print("="*70)
    
    # Find input image
    input_path = find_input_image()
    
    if not input_path:
        print("\n❌ No input image found!")
        print("\n📝 Instructions:")
        print("1. Place your image in this folder as:")
        print("   • input.jpg")
        print("   • input.png")
        print("   • test.jpg")
        print("   • test.png")
        print("\n2. Run: python app.py")
        
        # Create demo image
        print("\n🎨 Creating demo image for you...")
        demo_img = create_demo_image()
        cv2.imwrite('input.png', demo_img)
        print("✅ Demo image created as 'input.png'")
        print("🔄 Running detection on demo image...\n")
        input_path = 'input.png'
    
    # Load image
    print(f"📸 Loading: {input_path}")
    image = cv2.imread(input_path)
    
    if image is None:
        print("❌ Failed to load image!")
        return
    
    print(f"✅ Image loaded: {image.shape[1]}×{image.shape[0]} pixels")
    
    # Run detection
    print("\n🔬 Running Analysis...")
    print("├── Edge detection...")
    print("├── Gradient analysis...")
    print("├── Texture analysis...")
    print("├── Gabor filtering...")
    print("├── Phase congruency...")
    print("├── Feature fusion...")
    print("├── Object extraction...")
    print("└── Depth estimation...")
    
    detector = GlassDetector()
    start_time = time.time()
    
    results = detector.detect(image)
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Processing time: {elapsed:.2f} seconds")
    
    # Display results
    print(f"\n✅ Detection Complete!")
    print(f"🎯 Found {len(results['objects'])} transparent objects")
    
    if results['objects']:
        print("\n📦 Detected Objects:")
        for i, obj in enumerate(results['objects'], 1):
            print(f"  Object #{i}: Confidence={obj['confidence']:.1%}, Area={obj['area']:,}px")
    
    # Create and show visualization
    print("\n🎨 Creating visualization...")
    fig = create_visualization(image, results)
    
    # Save results
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = f"{output_dir}/detection_{timestamp}.png"
    
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black')
    print(f"💾 Saved to: {output_path}")
    
    # Save individual components
    cv2.imwrite(f"{output_dir}/mask_{timestamp}.png", results['mask'])
    cv2.imwrite(f"{output_dir}/depth_{timestamp}.png", results['depth'])
    
    # Show plot
    print("\n📊 Displaying results...")
    plt.style.use('dark_background')
    plt.show()
    
    print("\n" + "="*70)
    print("✨ All Done! Check 'results' folder for saved outputs")
    print("="*70)

def create_demo_image():
    """Create synthetic glass demo image"""
    h, w = 600, 800
    img = np.ones((h, w, 3), dtype=np.uint8) * 230
    
    # Grid background
    for i in range(0, w, 30):
        cv2.line(img, (i, 0), (i, h), (200, 200, 200), 1)
    for i in range(0, h, 30):
        cv2.line(img, (0, i), (w, i), (200, 200, 200), 1)
    
    # Objects
    cv2.rectangle(img, (100, 150), (300, 350), (120, 100, 80), -1)
    cv2.circle(img, (600, 300), 80, (80, 100, 120), -1)
    
    # Glass effect
    center_x, center_y = 400, 300
    for y in range(max(0, center_y-100), min(h, center_y+100)):
        for x in range(max(0, center_x-150), min(w, center_x+150)):
            dist = np.sqrt((x-center_x)**2/1.5 + (y-center_y)**2)
            if dist < 100:
                offset = int(10 * np.sin(dist * 0.1))
                src_x = np.clip(x + offset, 0, w-1)
                src_y = np.clip(y + offset, 0, h-1)
                img[y, x] = img[src_y, src_x] * 0.9
    
    # Highlight
    cv2.ellipse(img, (380, 280), (40, 25), -30, 0, 360, (255, 255, 255), -1)
    
    return img

if __name__ == "__main__":
    main()