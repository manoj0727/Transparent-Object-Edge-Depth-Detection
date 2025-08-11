import cv2
import numpy as np
import os

def create_sample_results():
    """Create sample images showing detection results"""
    os.makedirs("readme_images", exist_ok=True)
    
    # Create a sample input image with glass effect
    h, w = 400, 600
    img = np.ones((h, w, 3), dtype=np.uint8) * 240
    
    # Add grid pattern
    for i in range(0, w, 20):
        cv2.line(img, (i, 0), (i, h), (200, 200, 200), 1)
    for i in range(0, h, 20):
        cv2.line(img, (0, i), (w, i), (200, 200, 200), 1)
    
    # Simulate glass region with distortion
    cv2.ellipse(img, (300, 200), (120, 80), 0, 0, 360, (220, 220, 220), -1)
    
    # Add some objects
    cv2.rectangle(img, (100, 150), (200, 250), (100, 100, 100), -1)
    cv2.circle(img, (450, 200), 40, (150, 150, 150), -1)
    
    cv2.imwrite("readme_images/input_sample.png", img)
    
    # Create detection result visualization
    result = img.copy()
    
    # Draw detection overlay
    overlay = result.copy()
    cv2.ellipse(overlay, (300, 200), (120, 80), 0, 0, 360, (0, 0, 255), -1)
    result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)
    
    # Add bounding box
    cv2.rectangle(result, (180, 120), (420, 280), (0, 255, 0), 2)
    cv2.putText(result, "Glass Object (85%)", (185, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imwrite("readme_images/detection_result.png", result)
    
    # Create multi-panel visualization
    panel_size = (200, 150)
    vis = np.zeros((panel_size[1] * 2, panel_size[0] * 4, 3), dtype=np.uint8)
    
    # Resize images for panels
    img_small = cv2.resize(img, panel_size)
    result_small = cv2.resize(result, panel_size)
    
    # Create different analysis maps
    edge_map = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 50, 150)
    edge_colored = cv2.cvtColor(cv2.resize(edge_map, panel_size), cv2.COLOR_GRAY2BGR)
    
    depth_map = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(depth_map, (300, 200), (120, 80), 0, 0, 360, 200, -1)
    depth_colored = cv2.applyColorMap(cv2.resize(depth_map, panel_size), cv2.COLORMAP_JET)
    
    refraction_map = np.random.randint(0, 100, (h, w), dtype=np.uint8)
    cv2.ellipse(refraction_map, (300, 200), (120, 80), 0, 0, 360, 255, -1)
    refraction_colored = cv2.applyColorMap(cv2.resize(refraction_map, panel_size), cv2.COLORMAP_OCEAN)
    
    confidence_map = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(confidence_map, (300, 200), (120, 80), 0, 0, 360, 255, -1)
    confidence_colored = cv2.applyColorMap(cv2.resize(confidence_map, panel_size), cv2.COLORMAP_HOT)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (300, 200), (120, 80), 0, 0, 360, 255, -1)
    mask_colored = cv2.cvtColor(cv2.resize(mask, panel_size), cv2.COLOR_GRAY2BGR)
    
    transparency_map = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(transparency_map, (300, 200), (100, 60), 0, 0, 360, 200, -1)
    transparency_colored = cv2.applyColorMap(cv2.resize(transparency_map, panel_size), cv2.COLORMAP_PINK)
    
    # Place panels
    panels = [
        (img_small, "Input"),
        (result_small, "Detection"),
        (edge_colored, "Edges"),
        (depth_colored, "Depth"),
        (refraction_colored, "Refraction"),
        (confidence_colored, "Confidence"),
        (mask_colored, "Mask"),
        (transparency_colored, "Transparency")
    ]
    
    for idx, (panel, title) in enumerate(panels):
        row = idx // 4
        col = idx % 4
        y = row * panel_size[1]
        x = col * panel_size[0]
        vis[y:y+panel_size[1], x:x+panel_size[0]] = panel
        
        # Add title
        cv2.putText(vis, title, (x + 5, y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imwrite("readme_images/analysis_grid.png", vis)
    
    # Create workflow diagram
    workflow = np.ones((200, 800, 3), dtype=np.uint8) * 255
    
    steps = [
        "Input Image",
        "Preprocessing",
        "Multi-Scale Analysis",
        "Detection Fusion",
        "Result"
    ]
    
    for i, step in enumerate(steps):
        x = 80 + i * 140
        y = 100
        
        # Draw box
        cv2.rectangle(workflow, (x-50, y-20), (x+50, y+20), (100, 100, 100), 2)
        cv2.putText(workflow, step, (x-45, y+5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Draw arrow
        if i < len(steps) - 1:
            cv2.arrowedLine(workflow, (x+50, y), (x+90, y), (150, 150, 150), 2)
    
    cv2.imwrite("readme_images/workflow.png", workflow)
    
    print("✅ Sample images created in readme_images/")

if __name__ == "__main__":
    create_sample_results()