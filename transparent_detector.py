import cv2
import numpy as np
from scipy import ndimage
from skimage import filters, morphology, measure
from typing import Tuple, Optional, Dict, Any
import warnings

class TransparentObjectDetector:
    def __init__(self, use_polarization: bool = False):
        self.use_polarization = use_polarization
        self.background_model = None
        self.calibration_params = None
        
    def preprocess_image(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        if len(image.shape) == 2:
            gray = image
            hsv = None
            lab = None
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        
        if lab is not None:
            l_channel = lab[:, :, 0]
            enhanced_l = clahe.apply(l_channel)
            lab[:, :, 0] = enhanced_l
            enhanced_lab = lab
        else:
            enhanced_lab = None
            
        return {
            'gray': gray,
            'enhanced_gray': enhanced_gray,
            'hsv': hsv,
            'lab': enhanced_lab,
            'original': image
        }
    
    def detect_edge_disruptions(self, image: np.ndarray, 
                               background: Optional[np.ndarray] = None) -> np.ndarray:
        preprocessed = self.preprocess_image(image)
        enhanced = preprocessed['enhanced_gray']
        
        sobel_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_direction = np.arctan2(sobel_y, sobel_x)
        
        if background is not None:
            bg_preprocessed = self.preprocess_image(background)
            bg_enhanced = bg_preprocessed['enhanced_gray']
            
            bg_sobel_x = cv2.Sobel(bg_enhanced, cv2.CV_64F, 1, 0, ksize=3)
            bg_sobel_y = cv2.Sobel(bg_enhanced, cv2.CV_64F, 0, 1, ksize=3)
            bg_gradient_magnitude = np.sqrt(bg_sobel_x**2 + bg_sobel_y**2)
            bg_gradient_direction = np.arctan2(bg_sobel_y, bg_sobel_x)
            
            magnitude_diff = np.abs(gradient_magnitude - bg_gradient_magnitude)
            direction_diff = np.abs(np.sin(gradient_direction - bg_gradient_direction))
            
            disruption_map = magnitude_diff * 0.5 + direction_diff * 0.5
        else:
            kernel_size = 15
            local_mean = cv2.blur(gradient_magnitude, (kernel_size, kernel_size))
            local_std = cv2.blur((gradient_magnitude - local_mean)**2, 
                               (kernel_size, kernel_size))**0.5
            
            disruption_map = np.where(local_std > 0, 
                                     np.abs(gradient_magnitude - local_mean) / local_std, 
                                     0)
        
        disruption_map = cv2.normalize(disruption_map, None, 0, 255, cv2.NORM_MINMAX)
        return disruption_map.astype(np.uint8)
    
    def detect_refraction_regions(self, image: np.ndarray, 
                                 background: Optional[np.ndarray] = None) -> np.ndarray:
        if background is not None:
            flow = cv2.calcOpticalFlowFarneback(
                cv2.cvtColor(background, cv2.COLOR_BGR2GRAY) if len(background.shape) == 3 else background,
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
            
            threshold = np.percentile(magnitude[magnitude > 0], 75)
            refraction_mask = magnitude > threshold
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            gabor_kernels = []
            for theta in np.arange(0, np.pi, np.pi / 4):
                for frequency in [0.1, 0.2, 0.3]:
                    kernel = cv2.getGaborKernel((21, 21), 4.0, theta, 10.0, frequency, 0)
                    gabor_kernels.append(kernel)
            
            responses = []
            for kernel in gabor_kernels:
                filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
                responses.append(filtered)
            
            response_variance = np.var(np.stack(responses), axis=0)
            
            threshold = np.percentile(response_variance, 85)
            refraction_mask = response_variance > threshold
        
        refraction_mask = morphology.binary_closing(refraction_mask, morphology.disk(5))
        refraction_mask = morphology.binary_opening(refraction_mask, morphology.disk(3))
        
        return refraction_mask.astype(np.uint8) * 255
    
    def estimate_depth_from_refraction(self, image: np.ndarray, 
                                      refraction_mask: np.ndarray,
                                      baseline: float = 0.1) -> np.ndarray:
        h, w = image.shape[:2]
        depth_map = np.zeros((h, w), dtype=np.float32)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        edges = cv2.Canny(gray, 50, 150)
        edges_masked = cv2.bitwise_and(edges, refraction_mask)
        
        contours, _ = cv2.findContours(edges_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) < 100:
                continue
                
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            
            roi = cv2.bitwise_and(gray, gray, mask=mask)
            
            gradient_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
            
            curvature = np.abs(gradient_x) + np.abs(gradient_y)
            curvature_normalized = cv2.normalize(curvature, None, 0, 1, cv2.NORM_MINMAX)
            
            n_glass = 1.52
            n_air = 1.0
            critical_angle = np.arcsin(n_air / n_glass)
            
            estimated_depth = baseline * (1.0 / (curvature_normalized + 0.001))
            estimated_depth = np.clip(estimated_depth, 0, 10)
            
            depth_map = np.where(mask > 0, estimated_depth, depth_map)
        
        if depth_map.dtype != np.uint8:
            depth_map_uint8 = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_map_uint8 = cv2.medianBlur(depth_map_uint8, 5)
            depth_map = depth_map_uint8.astype(np.float32) * (depth_map.max() / 255.0) if depth_map.max() > 0 else depth_map_uint8.astype(np.float32)
        else:
            depth_map = cv2.medianBlur(depth_map, 5)
        
        return depth_map
    
    def detect_transparent_objects(self, image: np.ndarray, 
                                  background: Optional[np.ndarray] = None,
                                  stereo_pair: Optional[np.ndarray] = None) -> Dict[str, Any]:
        edge_disruptions = self.detect_edge_disruptions(image, background)
        
        refraction_regions = self.detect_refraction_regions(image, background)
        
        combined_mask = cv2.addWeighted(edge_disruptions, 0.5, refraction_regions, 0.5, 0)
        _, binary_mask = cv2.threshold(combined_mask, 127, 255, cv2.THRESH_BINARY)
        
        binary_mask = morphology.binary_closing(binary_mask > 0, morphology.disk(7))
        binary_mask = morphology.binary_opening(binary_mask, morphology.disk(5))
        binary_mask = (binary_mask * 255).astype(np.uint8)
        
        if stereo_pair is not None:
            depth_map = self.compute_stereo_depth(image, stereo_pair, binary_mask)
        else:
            depth_map = self.estimate_depth_from_refraction(image, binary_mask)
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) < 500:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            mask_roi = binary_mask[y:y+h, x:x+w]
            depth_roi = depth_map[y:y+h, x:x+w]
            
            objects.append({
                'id': i,
                'bbox': (x, y, w, h),
                'contour': contour,
                'area': cv2.contourArea(contour),
                'mean_depth': np.mean(depth_roi[mask_roi > 0]) if np.any(mask_roi > 0) else 0,
                'confidence': self.calculate_confidence(edge_disruptions[y:y+h, x:x+w], 
                                                       refraction_regions[y:y+h, x:x+w])
            })
        
        return {
            'objects': objects,
            'mask': binary_mask,
            'edge_disruptions': edge_disruptions,
            'refraction_regions': refraction_regions,
            'depth_map': depth_map,
            'combined_detection': combined_mask
        }
    
    def compute_stereo_depth(self, left_image: np.ndarray, 
                            right_image: np.ndarray,
                            mask: np.ndarray) -> np.ndarray:
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY) if len(left_image.shape) == 3 else left_image
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY) if len(right_image.shape) == 3 else right_image
        
        stereo = cv2.StereoBM_create(numDisparities=96, blockSize=15)
        disparity = stereo.compute(left_gray, right_gray)
        
        disparity = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        
        disparity_masked = cv2.bitwise_and(disparity.astype(np.uint8), mask)
        
        focal_length = 700
        baseline = 0.1
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            depth = np.where(disparity_masked > 0, 
                           (focal_length * baseline) / (disparity_masked + 1e-6),
                           0)
        
        return depth.astype(np.float32)
    
    def calculate_confidence(self, edge_map: np.ndarray, 
                           refraction_map: np.ndarray) -> float:
        edge_score = np.mean(edge_map) / 255.0
        refraction_score = np.mean(refraction_map) / 255.0
        
        combined_score = (edge_score + refraction_score) / 2.0
        
        confidence = min(combined_score * 1.5, 1.0)
        
        return confidence
    
    def visualize_results(self, image: np.ndarray, 
                         detection_results: Dict[str, Any]) -> np.ndarray:
        vis_image = image.copy()
        
        mask = detection_results['mask']
        depth_map = detection_results['depth_map']
        objects = detection_results['objects']
        
        colored_mask = np.zeros_like(image)
        colored_mask[:, :, 2] = mask
        vis_image = cv2.addWeighted(vis_image, 0.7, colored_mask, 0.3, 0)
        
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        
        for obj in objects:
            x, y, w, h = obj['bbox']
            confidence = obj['confidence']
            
            color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
            
            label = f"Glass #{obj['id']} ({confidence:.2f})"
            cv2.putText(vis_image, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if obj['mean_depth'] > 0:
                depth_label = f"Depth: {obj['mean_depth']:.2f}m"
                cv2.putText(vis_image, depth_label, (x, y + h + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        edge_vis = cv2.cvtColor(detection_results['edge_disruptions'], cv2.COLOR_GRAY2BGR)
        refraction_vis = cv2.cvtColor(detection_results['refraction_regions'], cv2.COLOR_GRAY2BGR)
        
        h, w = vis_image.shape[:2]
        combined_vis = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        combined_vis[0:h, 0:w] = vis_image
        combined_vis[0:h, w:w*2] = depth_colored
        combined_vis[h:h*2, 0:w] = edge_vis
        combined_vis[h:h*2, w:w*2] = refraction_vis
        
        cv2.putText(combined_vis, "Detection Result", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined_vis, "Depth Map", (w + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined_vis, "Edge Disruptions", (10, h + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined_vis, "Refraction Regions", (w + 10, h + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return combined_vis