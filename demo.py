import cv2
import numpy as np
import matplotlib.pyplot as plt
from transparent_detector import TransparentObjectDetector
import os

def create_synthetic_test_image():
    height, width = 600, 800
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    for i in range(0, width, 20):
        cv2.line(image, (i, 0), (i, height), (200, 200, 200), 1)
    for i in range(0, height, 20):
        cv2.line(image, (0, i), (width, i), (200, 200, 200), 1)
    
    glass_region = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(glass_region, (400, 300), (150, 100), 0, 0, 360, 255, -1)
    
    for y in range(height):
        for x in range(width):
            if glass_region[y, x] > 0:
                offset_x = int(5 * np.sin(x * 0.1))
                offset_y = int(5 * np.cos(y * 0.1))
                
                src_x = np.clip(x + offset_x, 0, width - 1)
                src_y = np.clip(y + offset_y, 0, height - 1)
                
                image[y, x] = image[src_y, src_x] * 0.9
    
    cv2.rectangle(image, (250, 200), (550, 400), (180, 180, 180), 2)
    
    return image, glass_region

def test_on_synthetic_image():
    print("Testing on synthetic image with simulated glass object...")
    
    detector = TransparentObjectDetector()
    
    test_image, ground_truth = create_synthetic_test_image()
    background = np.ones_like(test_image) * 255
    for i in range(0, test_image.shape[1], 20):
        cv2.line(background, (i, 0), (i, test_image.shape[0]), (200, 200, 200), 1)
    for i in range(0, test_image.shape[0], 20):
        cv2.line(background, (0, i), (test_image.shape[1], i), (200, 200, 200), 1)
    
    results = detector.detect_transparent_objects(test_image, background=background)
    
    visualization = detector.visualize_results(test_image, results)
    
    if not os.path.exists('output'):
        os.makedirs('output')
    
    cv2.imwrite('output/synthetic_test_result.png', visualization)
    cv2.imwrite('output/synthetic_test_input.png', test_image)
    cv2.imwrite('output/synthetic_mask.png', results['mask'])
    cv2.imwrite('output/synthetic_depth.png', 
                (cv2.normalize(results['depth_map'], None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8))
    
    print(f"Detected {len(results['objects'])} transparent objects")
    for obj in results['objects']:
        print(f"  Object {obj['id']}: Confidence={obj['confidence']:.2f}, Area={obj['area']:.0f}px")
    
    print("\nResults saved to output/ directory")
    
    return results

def test_on_real_image(image_path: str, background_path: str = None):
    print(f"\nTesting on real image: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return None
    
    detector = TransparentObjectDetector()
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    background = None
    if background_path and os.path.exists(background_path):
        background = cv2.imread(background_path)
        print(f"Using background image: {background_path}")
    
    results = detector.detect_transparent_objects(image, background=background)
    
    visualization = detector.visualize_results(image, results)
    
    output_name = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(f'output/{output_name}_result.png', visualization)
    cv2.imwrite(f'output/{output_name}_mask.png', results['mask'])
    cv2.imwrite(f'output/{output_name}_depth.png',
                (cv2.normalize(results['depth_map'], None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8))
    
    print(f"Detected {len(results['objects'])} transparent objects")
    for obj in results['objects']:
        print(f"  Object {obj['id']}: Confidence={obj['confidence']:.2f}, Area={obj['area']:.0f}px, Mean Depth={obj['mean_depth']:.2f}m")
    
    return results

def create_sample_images():
    print("Creating sample test images...")
    
    if not os.path.exists('sample_images'):
        os.makedirs('sample_images')
    
    height, width = 480, 640
    
    scene1 = np.ones((height, width, 3), dtype=np.uint8) * 255
    cv2.rectangle(scene1, (100, 100), (300, 300), (100, 100, 100), -1)
    cv2.circle(scene1, (500, 200), 80, (150, 150, 150), -1)
    
    glass_overlay = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(glass_overlay, (320, 240), (100, 150), 30, 0, 360, 255, -1)
    
    for y in range(height):
        for x in range(width):
            if glass_overlay[y, x] > 0:
                scene1[y, x] = (scene1[y, x] * 0.8).astype(np.uint8)
    
    cv2.imwrite('sample_images/scene_with_glass.png', scene1)
    
    scene2 = np.ones((height, width, 3), dtype=np.uint8) * 200
    for i in range(10, width, 30):
        cv2.line(scene2, (i, 0), (i, height), (150, 150, 150), 2)
    
    cv2.rectangle(scene2, (200, 150), (440, 330), (255, 255, 255), -1)
    cv2.rectangle(scene2, (210, 160), (430, 320), (240, 240, 240), -1)
    
    cv2.imwrite('sample_images/window_scene.png', scene2)
    
    print("Sample images created in sample_images/ directory")

def main():
    print("=" * 60)
    print("Transparent Object Edge & Depth Detection Demo")
    print("=" * 60)
    
    results_synthetic = test_on_synthetic_image()
    
    create_sample_images()
    
    test_on_real_image('sample_images/scene_with_glass.png')
    test_on_real_image('sample_images/window_scene.png')
    
    print("\n" + "=" * 60)
    print("Demo completed! Check the output/ directory for results.")
    print("=" * 60)

if __name__ == "__main__":
    main()