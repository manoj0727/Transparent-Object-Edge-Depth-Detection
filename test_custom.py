#!/usr/bin/env python3
"""
Test the transparent object detector with your own images
"""

import cv2
import sys
from transparent_detector import TransparentObjectDetector

def process_image(image_path):
    print(f"Processing: {image_path}")
    
    detector = TransparentObjectDetector()
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Detect transparent objects
    results = detector.detect_transparent_objects(image)
    
    # Visualize results
    visualization = detector.visualize_results(image, results)
    
    # Save output
    output_path = image_path.replace('.', '_detected.')
    cv2.imwrite(output_path, visualization)
    
    print(f"Results saved to: {output_path}")
    print(f"Found {len(results['objects'])} transparent objects")
    
    for obj in results['objects']:
        print(f"  - Object {obj['id']}: Confidence={obj['confidence']:.2%}, Area={obj['area']:.0f}px")
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_custom.py <image_path>")
        print("Example: python test_custom.py my_glass_image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    process_image(image_path)