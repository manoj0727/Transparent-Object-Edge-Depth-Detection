import cv2
from transparent_detector import TransparentObjectDetector

# CHANGE THIS to your image path
IMAGE_PATH = "/Users/manojkumawat/ipcv/Transparent-Object-Edge-Depth-Detection/test.png"  # <-- PUT YOUR IMAGE PATH HERE

# Create detector
detector = TransparentObjectDetector()

# Load and process image
print(f"Processing: {IMAGE_PATH}")
image = cv2.imread(IMAGE_PATH)

if image is None:
    print(f"Error: Could not find image at {IMAGE_PATH}")
    print("Please update IMAGE_PATH in this script to point to your image")
else:
    # Detect transparent objects
    results = detector.detect_transparent_objects(image)
    
    # Save visualization
    visualization = detector.visualize_results(image, results)
    output_path = "output/my_test_result.png"
    cv2.imwrite(output_path, visualization)
    
    print(f"\n✅ Detection complete!")
    print(f"📊 Found {len(results['objects'])} transparent objects")
    print(f"💾 Result saved to: {output_path}")
    print(f"\nTo view: open {output_path}")