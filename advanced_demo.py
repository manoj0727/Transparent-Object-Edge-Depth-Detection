#!/usr/bin/env python3
"""
Advanced Transparent Object Detection System
Automatically detects and analyzes transparent objects with state-of-the-art techniques
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from advanced_transparent_detector import AdvancedTransparentObjectDetector
import os
import time
from pathlib import Path

class TransparentObjectAnalyzer:
    def __init__(self):
        print("🚀 Initializing Advanced Transparent Object Detection System...")
        self.detector = AdvancedTransparentObjectDetector(enable_ml=True, enable_caching=True)
        print("✅ System initialized with ML and caching enabled")
        
    def analyze_image(self, image_path: str, save_results: bool = True):
        """
        Comprehensive analysis of transparent objects in an image
        """
        print(f"\n{'='*60}")
        print(f"📸 Analyzing: {image_path}")
        print(f"{'='*60}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error: Cannot load image from {image_path}")
            return None
        
        print(f"📊 Image dimensions: {image.shape[1]}x{image.shape[0]} pixels")
        
        # Start detection
        print("\n🔍 Running advanced detection pipeline...")
        start_time = time.time()
        
        # Perform advanced detection
        results = self.detector.detect_advanced(image)
        
        detection_time = time.time() - start_time
        print(f"⏱️  Detection completed in {detection_time:.2f} seconds")
        
        # Analyze results
        self._print_analysis(results)
        
        # Generate visualizations
        if save_results:
            output_path = self._save_results(image, results, image_path)
            print(f"\n💾 Results saved to: {output_path}")
        
        return results
    
    def _print_analysis(self, results):
        """Print detailed analysis of detection results"""
        objects = results['objects']
        
        print(f"\n📋 DETECTION SUMMARY")
        print(f"{'─'*40}")
        print(f"🎯 Transparent objects detected: {len(objects)}")
        
        if objects:
            # Overall statistics
            avg_confidence = np.mean([obj['confidence'] for obj in objects])
            total_area = sum([obj['area'] for obj in objects])
            
            print(f"📈 Average confidence: {avg_confidence:.1%}")
            print(f"📐 Total transparent area: {total_area:,} pixels")
            
            # Physics properties
            if 'physics_properties' in results:
                props = results['physics_properties']
                print(f"\n🔬 PHYSICS PROPERTIES")
                print(f"{'─'*40}")
                print(f"💎 Refractive index: {props.get('refractive_index', 'N/A')}")
                print(f"📏 Estimated thickness: {props.get('estimated_thickness', 0):.2f}mm")
                print(f"🌟 Transparency level: {props.get('transparency_level', 0):.1%}")
            
            # Individual object analysis
            print(f"\n🔎 INDIVIDUAL OBJECTS")
            print(f"{'─'*40}")
            
            for obj in objects:
                print(f"\n📦 Object #{obj['id']} - {obj['type']}")
                print(f"   • Confidence: {obj['confidence']:.1%}")
                print(f"   • Area: {obj['area']:,} pixels")
                print(f"   • Mean depth: {obj['mean_depth']:.2f}m")
                print(f"   • Location: ({obj['centroid'][1]:.0f}, {obj['centroid'][0]:.0f})")
                print(f"   • Shape: Eccentricity={obj['eccentricity']:.2f}, Solidity={obj['solidity']:.2f}")
                
                # Classify quality
                if obj['confidence'] > 0.8:
                    quality = "🟢 High confidence detection"
                elif obj['confidence'] > 0.6:
                    quality = "🟡 Medium confidence detection"
                else:
                    quality = "🔴 Low confidence detection"
                print(f"   • Quality: {quality}")
        else:
            print("\n⚠️  No transparent objects detected in this image")
            print("💡 Tips for better detection:")
            print("   • Ensure good lighting conditions")
            print("   • Use images with textured backgrounds")
            print("   • Transparent objects should have visible distortions")
    
    def _save_results(self, image, results, original_path):
        """Generate and save comprehensive visualization"""
        # Create output directory
        output_dir = Path("output_advanced")
        output_dir.mkdir(exist_ok=True)
        
        # Generate base filename
        base_name = Path(original_path).stem
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Create comprehensive visualization
        vis = self._create_advanced_visualization(image, results)
        
        # Save main visualization
        output_path = output_dir / f"{base_name}_analysis_{timestamp}.png"
        cv2.imwrite(str(output_path), vis)
        
        # Save individual components
        components = {
            'mask': results['mask'],
            'depth': (cv2.normalize(results['depth_map'], None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8),
            'confidence': results['confidence_map']
        }
        
        for name, component in components.items():
            comp_path = output_dir / f"{base_name}_{name}_{timestamp}.png"
            cv2.imwrite(str(comp_path), component)
        
        return output_path
    
    def _create_advanced_visualization(self, image, results):
        """Create advanced multi-panel visualization"""
        h, w = image.shape[:2]
        
        # Create 3x3 grid visualization
        grid_h, grid_w = h * 3, w * 3
        visualization = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        # Panel 1: Original image with detections
        panel1 = self._draw_detections(image.copy(), results)
        
        # Panel 2: Depth map
        depth_colored = cv2.applyColorMap(
            cv2.normalize(results['depth_map'], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        
        # Panel 3: Confidence heatmap
        confidence_colored = cv2.applyColorMap(results['confidence_map'], cv2.COLORMAP_HOT)
        
        # Panel 4: Edge disruptions
        edge_map = results['detection_maps'].get('edge_disruptions', np.zeros((h, w), dtype=np.uint8))
        edge_colored = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)
        
        # Panel 5: Refraction map
        refraction_map = results['detection_maps'].get('refraction', np.zeros((h, w), dtype=np.uint8))
        refraction_colored = cv2.applyColorMap(refraction_map, cv2.COLORMAP_OCEAN)
        
        # Panel 6: Transparency map
        transparency_map = results['detection_maps'].get('transparency', np.zeros((h, w), dtype=np.uint8))
        transparency_colored = cv2.applyColorMap(transparency_map, cv2.COLORMAP_PINK)
        
        # Panel 7: Binary mask
        mask_colored = cv2.cvtColor(results['mask'], cv2.COLOR_GRAY2BGR)
        
        # Panel 8: Specular highlights
        specular_map = results['detection_maps'].get('specular', np.zeros((h, w), dtype=np.uint8))
        specular_colored = cv2.cvtColor(specular_map, cv2.COLOR_GRAY2BGR)
        
        # Panel 9: Combined overlay
        overlay = self._create_overlay(image, results)
        
        # Arrange panels in grid
        panels = [
            (panel1, "Detection Results"),
            (depth_colored, "Depth Map"),
            (confidence_colored, "Confidence Heatmap"),
            (edge_colored, "Edge Disruptions"),
            (refraction_colored, "Refraction Analysis"),
            (transparency_colored, "Transparency Patterns"),
            (mask_colored, "Binary Mask"),
            (specular_colored, "Specular Highlights"),
            (overlay, "Combined Analysis")
        ]
        
        for idx, (panel, title) in enumerate(panels):
            row = idx // 3
            col = idx % 3
            y_start = row * h
            x_start = col * w
            
            # Place panel
            visualization[y_start:y_start+h, x_start:x_start+w] = panel
            
            # Add title
            cv2.putText(visualization, title, (x_start + 10, y_start + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Add overall title
        title = f"Advanced Transparent Object Analysis - {len(results['objects'])} Objects Detected"
        cv2.putText(visualization, title, (grid_w//2 - 400, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)
        
        return visualization
    
    def _draw_detections(self, image, results):
        """Draw detection results on image"""
        objects = results['objects']
        
        for obj in objects:
            x, y, w, h = obj['bbox']
            confidence = obj['confidence']
            obj_type = obj['type']
            
            # Color based on confidence
            if confidence > 0.8:
                color = (0, 255, 0)  # Green
            elif confidence > 0.6:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 165, 255)  # Orange
            
            # Draw bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            # Draw filled background for text
            label = f"{obj_type} ({confidence:.0%})"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x, y - 25), (x + label_size[0], y), color, -1)
            
            # Draw text
            cv2.putText(image, label, (x, y - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Draw centroid
            cx, cy = int(obj['centroid'][1]), int(obj['centroid'][0])
            cv2.circle(image, (cx, cy), 5, color, -1)
            
            # Draw depth info
            if obj['mean_depth'] > 0:
                depth_label = f"Depth: {obj['mean_depth']:.1f}m"
                cv2.putText(image, depth_label, (x, y + h + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
        return image
    
    def _create_overlay(self, image, results):
        """Create combined overlay visualization"""
        overlay = image.copy()
        
        # Add transparent mask overlay
        mask_colored = np.zeros_like(image)
        mask_colored[:, :, 2] = results['mask']  # Red channel
        overlay = cv2.addWeighted(overlay, 0.6, mask_colored, 0.4, 0)
        
        # Add depth contours
        depth_norm = cv2.normalize(results['depth_map'], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        contours, _ = cv2.findContours(depth_norm, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)
        
        return overlay
    
    def batch_process(self, image_paths):
        """Process multiple images"""
        print(f"\n{'='*60}")
        print(f"🔄 BATCH PROCESSING - {len(image_paths)} images")
        print(f"{'='*60}")
        
        results_all = []
        for i, path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}] Processing {path}")
            results = self.analyze_image(path)
            results_all.append(results)
        
        # Summary statistics
        print(f"\n{'='*60}")
        print(f"📊 BATCH PROCESSING SUMMARY")
        print(f"{'='*60}")
        
        total_objects = sum(len(r['objects']) for r in results_all if r)
        avg_objects = total_objects / len(results_all) if results_all else 0
        
        print(f"✅ Total images processed: {len(image_paths)}")
        print(f"🎯 Total objects detected: {total_objects}")
        print(f"📈 Average objects per image: {avg_objects:.1f}")
        
        return results_all


def main():
    """Main demo function"""
    print("\n" + "="*60)
    print("🌟 ADVANCED TRANSPARENT OBJECT DETECTION SYSTEM 🌟")
    print("="*60)
    print("\nThis system uses state-of-the-art computer vision techniques:")
    print("• Multi-scale feature extraction")
    print("• Physics-based refraction modeling")
    print("• Machine learning confidence scoring")
    print("• Adaptive threshold learning")
    print("• Parallel processing pipeline")
    
    # Initialize analyzer
    analyzer = TransparentObjectAnalyzer()
    
    # Check if test image exists
    test_image_path = "/Users/manojkumawat/ipcv/Transparent-Object-Edge-Depth-Detection/test.png"
    
    if os.path.exists(test_image_path):
        print(f"\n✅ Found test image: {test_image_path}")
        analyzer.analyze_image(test_image_path)
    else:
        print(f"\n⚠️  Test image not found at: {test_image_path}")
        print("Creating synthetic test images...")
        
        # Create synthetic test images
        create_test_images()
        
        # Process synthetic images
        test_images = [
            "sample_images/synthetic_glass.png",
            "sample_images/synthetic_window.png"
        ]
        
        for img_path in test_images:
            if os.path.exists(img_path):
                analyzer.analyze_image(img_path)
    
    print("\n" + "="*60)
    print("✨ Analysis Complete! Check output_advanced/ folder for results")
    print("="*60)


def create_test_images():
    """Create synthetic test images with transparent objects"""
    os.makedirs("sample_images", exist_ok=True)
    
    # Create synthetic glass object
    h, w = 600, 800
    image = np.ones((h, w, 3), dtype=np.uint8) * 200
    
    # Add background pattern
    for i in range(0, w, 20):
        cv2.line(image, (i, 0), (i, h), (150, 150, 150), 1)
    for i in range(0, h, 20):
        cv2.line(image, (0, i), (w, i), (150, 150, 150), 1)
    
    # Simulate glass distortion
    center = (w//2, h//2)
    radius = 150
    
    for y in range(h):
        for x in range(w):
            dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            if dist < radius:
                # Create refraction effect
                offset = int(10 * np.sin(dist * 0.05))
                src_x = np.clip(x + offset, 0, w-1)
                src_y = np.clip(y + offset, 0, h-1)
                image[y, x] = image[src_y, src_x] * 0.9
    
    cv2.imwrite("sample_images/synthetic_glass.png", image)
    
    # Create synthetic window
    image2 = np.ones((h, w, 3), dtype=np.uint8) * 255
    cv2.rectangle(image2, (200, 150), (600, 450), (230, 230, 230), -1)
    cv2.rectangle(image2, (210, 160), (590, 440), (240, 240, 240), -1)
    
    # Add reflection effect
    for i in range(160, 440, 10):
        cv2.line(image2, (210, i), (590, i), (250, 250, 250), 1)
    
    cv2.imwrite("sample_images/synthetic_window.png", image2)
    
    print("✅ Synthetic test images created")


if __name__ == "__main__":
    main()