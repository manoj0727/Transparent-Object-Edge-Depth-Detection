import cv2
import numpy as np
from scipy import ndimage, signal, stats
from scipy.spatial import distance
from skimage import filters, morphology, measure, feature, segmentation, transform
from skimage.feature import local_binary_pattern, hog
from typing import Tuple, Optional, Dict, Any, List
import warnings
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

class AdvancedTransparentObjectDetector:
    def __init__(self, enable_ml: bool = True, enable_caching: bool = True):
        self.enable_ml = enable_ml
        self.enable_caching = enable_caching
        self.cache = {} if enable_caching else None
        self.detection_history = []
        self.adaptive_thresholds = {
            'edge_disruption': 0.3,
            'refraction': 0.4,
            'transparency': 0.5,
            'depth_variation': 0.35
        }
        self.feature_extractors = self._initialize_feature_extractors()
        self.physics_model = PhysicsBasedRefractionModel()
        self.ml_scorer = MLConfidenceScorer() if enable_ml else None
        
    def _initialize_feature_extractors(self):
        return {
            'gabor': GaborFeatureExtractor(),
            'lbp': LBPFeatureExtractor(),
            'hog': HOGFeatureExtractor(),
            'wavelet': WaveletFeatureExtractor(),
            'fractal': FractalDimensionExtractor()
        }
    
    def detect_advanced(self, image: np.ndarray, 
                        background: Optional[np.ndarray] = None,
                        stereo_pair: Optional[np.ndarray] = None,
                        polarized_pair: Optional[np.ndarray] = None) -> Dict[str, Any]:
        
        cache_key = self._generate_cache_key(image) if self.enable_caching else None
        if cache_key and cache_key in self.cache:
            return self.cache[cache_key]
        
        preprocessed = self._advanced_preprocessing(image)
        
        multi_scale_features = self._extract_multiscale_features(preprocessed)
        
        detection_maps = self._parallel_detection_pipeline(
            preprocessed, background, multi_scale_features
        )
        
        if polarized_pair is not None:
            polarization_features = self._analyze_polarization(image, polarized_pair)
            detection_maps['polarization'] = polarization_features
        
        fusion_map = self._advanced_fusion(detection_maps)
        
        refined_mask = self._ml_based_refinement(fusion_map, image) if self.enable_ml else fusion_map
        
        depth_map = self._advanced_depth_estimation(
            image, refined_mask, stereo_pair, detection_maps
        )
        
        objects = self._segment_and_classify_objects(refined_mask, depth_map, image)
        
        self._update_adaptive_thresholds(objects)
        
        result = {
            'objects': objects,
            'mask': refined_mask,
            'depth_map': depth_map,
            'detection_maps': detection_maps,
            'confidence_map': self._generate_confidence_map(detection_maps),
            'features': multi_scale_features,
            'physics_properties': self._estimate_physics_properties(objects, depth_map)
        }
        
        if self.enable_caching and cache_key:
            self.cache[cache_key] = result
        
        self.detection_history.append({
            'timestamp': np.datetime64('now'),
            'num_objects': len(objects),
            'avg_confidence': np.mean([obj['confidence'] for obj in objects]) if objects else 0
        })
        
        return result
    
    def _advanced_preprocessing(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        result = {}
        
        if len(image.shape) == 2:
            gray = image
            result['rgb'] = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            result['rgb'] = image
        
        result['gray'] = gray
        result['hsv'] = cv2.cvtColor(result['rgb'], cv2.COLOR_BGR2HSV)
        result['lab'] = cv2.cvtColor(result['rgb'], cv2.COLOR_BGR2LAB)
        result['yuv'] = cv2.cvtColor(result['rgb'], cv2.COLOR_BGR2YUV)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        result['enhanced_gray'] = clahe.apply(gray)
        
        result['gradient_x'] = cv2.Sobel(result['enhanced_gray'], cv2.CV_64F, 1, 0, ksize=3)
        result['gradient_y'] = cv2.Sobel(result['enhanced_gray'], cv2.CV_64F, 0, 1, ksize=3)
        result['gradient_magnitude'] = np.sqrt(result['gradient_x']**2 + result['gradient_y']**2)
        result['gradient_direction'] = np.arctan2(result['gradient_y'], result['gradient_x'])
        
        result['laplacian'] = cv2.Laplacian(result['enhanced_gray'], cv2.CV_64F)
        
        result['canny_multi'] = {
            'low': cv2.Canny(result['enhanced_gray'], 30, 100),
            'mid': cv2.Canny(result['enhanced_gray'], 50, 150),
            'high': cv2.Canny(result['enhanced_gray'], 100, 200)
        }
        
        return result
    
    def _extract_multiscale_features(self, preprocessed: Dict) -> Dict[str, np.ndarray]:
        features = {}
        gray = preprocessed['enhanced_gray']
        
        pyramid_levels = 4
        pyramid = tuple(transform.pyramid_gaussian(gray, max_layer=pyramid_levels-1, 
                                                   downscale=2, channel_axis=None))
        
        for i, layer in enumerate(pyramid):
            scale_features = {}
            
            for name, extractor in self.feature_extractors.items():
                scale_features[name] = extractor.extract(layer)
            
            features[f'scale_{i}'] = scale_features
        
        features['global_stats'] = self._compute_global_statistics(preprocessed)
        
        return features
    
    def _parallel_detection_pipeline(self, preprocessed: Dict, 
                                    background: Optional[np.ndarray],
                                    features: Dict) -> Dict[str, np.ndarray]:
        detection_maps = {}
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._detect_edge_disruptions_advanced, 
                              preprocessed, background): 'edge_disruptions',
                executor.submit(self._detect_refraction_advanced, 
                              preprocessed, background, features): 'refraction',
                executor.submit(self._detect_transparency_patterns, 
                              preprocessed, features): 'transparency',
                executor.submit(self._detect_specular_highlights, 
                              preprocessed): 'specular'
            }
            
            for future in as_completed(futures):
                map_name = futures[future]
                try:
                    detection_maps[map_name] = future.result()
                except Exception as e:
                    print(f"Error in {map_name}: {e}")
                    detection_maps[map_name] = np.zeros_like(preprocessed['gray'])
        
        return detection_maps
    
    def _detect_edge_disruptions_advanced(self, preprocessed: Dict, 
                                         background: Optional[np.ndarray]) -> np.ndarray:
        grad_mag = preprocessed['gradient_magnitude']
        grad_dir = preprocessed['gradient_direction']
        
        coherence = self._compute_gradient_coherence(grad_mag, grad_dir)
        
        structure_tensor = self._compute_structure_tensor(preprocessed['enhanced_gray'])
        eigenvalues = self._structure_tensor_eigenvalues(structure_tensor)
        anisotropy = (eigenvalues[0] - eigenvalues[1]) / (eigenvalues[0] + eigenvalues[1] + 1e-10)
        
        if background is not None:
            bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY) if len(background.shape) == 3 else background
            bg_edges = cv2.Canny(bg_gray, 50, 150)
            current_edges = preprocessed['canny_multi']['mid']
            
            edge_diff = cv2.absdiff(current_edges, bg_edges)
            
            flow = cv2.calcOpticalFlowFarneback(bg_gray, preprocessed['gray'], 
                                               None, 0.5, 3, 15, 3, 7, 1.5, 0)
            flow_magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
            
            disruption = (1 - coherence) * 0.3 + anisotropy * 0.3 + \
                        edge_diff/255.0 * 0.2 + flow_magnitude/flow_magnitude.max() * 0.2
        else:
            local_variance = ndimage.generic_filter(grad_mag, np.var, size=15)
            normalized_variance = local_variance / (local_variance.max() + 1e-10)
            
            disruption = (1 - coherence) * 0.4 + anisotropy * 0.3 + normalized_variance * 0.3
        
        disruption = cv2.GaussianBlur(disruption.astype(np.float32), (5, 5), 1.0)
        disruption = cv2.normalize(disruption, None, 0, 255, cv2.NORM_MINMAX)
        
        return disruption.astype(np.uint8)
    
    def _detect_refraction_advanced(self, preprocessed: Dict, 
                                   background: Optional[np.ndarray],
                                   features: Dict) -> np.ndarray:
        gabor_responses = []
        for scale_key in features:
            if scale_key.startswith('scale_'):
                gabor_responses.append(features[scale_key].get('gabor', np.zeros_like(preprocessed['gray'])))
        
        gabor_variance = np.var(np.stack(gabor_responses), axis=0) if gabor_responses else np.zeros_like(preprocessed['gray'])
        
        phase_congruency = self._compute_phase_congruency(preprocessed['enhanced_gray'])
        
        laplacian = preprocessed['laplacian']
        laplacian_energy = np.abs(laplacian)
        
        if background is not None:
            bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY) if len(background.shape) == 3 else background
            
            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(bg_gray, None)
            kp2, des2 = orb.detectAndCompute(preprocessed['gray'], None)
            
            if des1 is not None and des2 is not None:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                
                distortion_map = np.zeros_like(preprocessed['gray'], dtype=np.float32)
                for match in matches:
                    pt1 = kp1[match.queryIdx].pt
                    pt2 = kp2[match.trainIdx].pt
                    displacement = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
                    cv2.circle(distortion_map, (int(pt2[0]), int(pt2[1])), 
                             int(10 + displacement), displacement/10, -1)
            else:
                distortion_map = np.zeros_like(preprocessed['gray'], dtype=np.float32)
            
            refraction = gabor_variance * 0.3 + phase_congruency * 0.3 + \
                        laplacian_energy/laplacian_energy.max() * 0.2 + \
                        distortion_map/distortion_map.max() * 0.2 if distortion_map.max() > 0 else 0
        else:
            refraction = gabor_variance * 0.4 + phase_congruency * 0.4 + \
                        laplacian_energy/laplacian_energy.max() * 0.2
        
        refraction = cv2.normalize(refraction, None, 0, 255, cv2.NORM_MINMAX)
        return refraction.astype(np.uint8)
    
    def _detect_transparency_patterns(self, preprocessed: Dict, features: Dict) -> np.ndarray:
        hsv = preprocessed['hsv']
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        
        low_saturation = saturation < np.percentile(saturation, 30)
        
        texture_energy = np.zeros_like(preprocessed['gray'], dtype=np.float32)
        for scale_key in features:
            if scale_key.startswith('scale_'):
                lbp = features[scale_key].get('lbp', np.zeros_like(preprocessed['gray']))
                if lbp.shape == texture_energy.shape:
                    texture_energy += lbp
        
        texture_energy /= len([k for k in features if k.startswith('scale_')])
        
        intensity_variance = ndimage.generic_filter(preprocessed['gray'], np.var, size=11)
        
        lab = preprocessed['lab']
        l_channel = lab[:, :, 0]
        luminance_gradient = np.gradient(l_channel)[0]**2 + np.gradient(l_channel)[1]**2
        
        transparency = low_saturation.astype(np.float32) * 0.3 + \
                       (1 - texture_energy/texture_energy.max()) * 0.3 + \
                       (intensity_variance/intensity_variance.max()) * 0.2 + \
                       (luminance_gradient/luminance_gradient.max()) * 0.2
        
        transparency = cv2.normalize(transparency, None, 0, 255, cv2.NORM_MINMAX)
        return transparency.astype(np.uint8)
    
    def _detect_specular_highlights(self, preprocessed: Dict) -> np.ndarray:
        gray = preprocessed['gray']
        
        threshold = np.percentile(gray, 95)
        bright_regions = gray > threshold
        
        gradient_mag = preprocessed['gradient_magnitude']
        high_gradient = gradient_mag > np.percentile(gradient_mag, 90)
        
        specular = bright_regions.astype(np.float32) * high_gradient.astype(np.float32)
        
        specular = morphology.binary_dilation(specular > 0, morphology.disk(3))
        
        return (specular * 255).astype(np.uint8)
    
    def _advanced_fusion(self, detection_maps: Dict[str, np.ndarray]) -> np.ndarray:
        weights = {
            'edge_disruptions': 0.3,
            'refraction': 0.35,
            'transparency': 0.25,
            'specular': 0.1,
            'polarization': 0.4
        }
        
        h, w = next(iter(detection_maps.values())).shape
        fusion = np.zeros((h, w), dtype=np.float32)
        total_weight = 0
        
        for map_name, detection_map in detection_maps.items():
            weight = weights.get(map_name, 0.2)
            if detection_map.shape == fusion.shape:
                fusion += detection_map.astype(np.float32) * weight
                total_weight += weight
        
        fusion /= total_weight if total_weight > 0 else 1
        
        fusion = cv2.bilateralFilter(fusion.astype(np.float32), 9, 75, 75)
        
        adaptive_threshold = np.percentile(fusion, 70)
        binary = fusion > adaptive_threshold
        
        binary = morphology.binary_closing(binary, morphology.disk(5))
        binary = morphology.binary_opening(binary, morphology.disk(3))
        
        min_area = 500
        labeled = measure.label(binary)
        for region in measure.regionprops(labeled):
            if region.area < min_area:
                binary[labeled == region.label] = 0
        
        return (binary * 255).astype(np.uint8)
    
    def _ml_based_refinement(self, initial_mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        features = self._extract_region_features(initial_mask, image)
        
        confidence_scores = self.ml_scorer.predict(features) if self.ml_scorer else np.ones(len(features))
        
        refined_mask = initial_mask.copy()
        labeled = measure.label(initial_mask > 0)
        
        for i, region in enumerate(measure.regionprops(labeled)):
            if i < len(confidence_scores) and confidence_scores[i] < 0.3:
                refined_mask[labeled == region.label] = 0
        
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, 
                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        
        return refined_mask
    
    def _advanced_depth_estimation(self, image: np.ndarray, mask: np.ndarray,
                                  stereo_pair: Optional[np.ndarray],
                                  detection_maps: Dict) -> np.ndarray:
        h, w = mask.shape
        depth_map = np.zeros((h, w), dtype=np.float32)
        
        if stereo_pair is not None:
            depth_map = self._compute_stereo_depth_advanced(image, stereo_pair, mask)
        else:
            refraction_map = detection_maps.get('refraction', np.zeros_like(mask))
            depth_map = self.physics_model.estimate_depth_from_refraction(
                image, mask, refraction_map
            )
        
        depth_map = self._apply_depth_refinement(depth_map, mask, image)
        
        return depth_map
    
    def _segment_and_classify_objects(self, mask: np.ndarray, depth_map: np.ndarray,
                                     image: np.ndarray) -> List[Dict]:
        objects = []
        
        labeled = measure.label(mask > 0)
        regions = measure.regionprops(labeled)
        
        for i, region in enumerate(regions):
            bbox = region.bbox
            contour = self._region_to_contour(region, mask.shape)
            
            mask_roi = mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            depth_roi = depth_map[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            image_roi = image[bbox[0]:bbox[2], bbox[1]:bbox[3]] if len(image.shape) == 3 else image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            
            features = self._extract_object_features(mask_roi, depth_roi, image_roi, region)
            
            obj_type = self._classify_transparent_object(features)
            confidence = self._calculate_advanced_confidence(features)
            
            objects.append({
                'id': i,
                'type': obj_type,
                'bbox': (bbox[1], bbox[0], bbox[3]-bbox[1], bbox[2]-bbox[0]),
                'contour': contour,
                'area': region.area,
                'perimeter': region.perimeter,
                'mean_depth': np.mean(depth_roi[mask_roi > 0]) if np.any(mask_roi > 0) else 0,
                'depth_variance': np.var(depth_roi[mask_roi > 0]) if np.any(mask_roi > 0) else 0,
                'confidence': confidence,
                'features': features,
                'centroid': region.centroid,
                'orientation': region.orientation,
                'eccentricity': region.eccentricity,
                'solidity': region.solidity
            })
        
        return objects
    
    def _generate_confidence_map(self, detection_maps: Dict[str, np.ndarray]) -> np.ndarray:
        maps = list(detection_maps.values())
        if not maps:
            return np.zeros((100, 100), dtype=np.uint8)
        
        confidence = np.mean(np.stack([m.astype(np.float32) for m in maps]), axis=0)
        confidence = cv2.normalize(confidence, None, 0, 255, cv2.NORM_MINMAX)
        return confidence.astype(np.uint8)
    
    def _estimate_physics_properties(self, objects: List[Dict], depth_map: np.ndarray) -> Dict:
        properties = {}
        
        if objects:
            properties['refractive_index'] = 1.52
            properties['estimated_thickness'] = np.mean([obj['depth_variance'] for obj in objects])
            properties['transparency_level'] = np.mean([obj['confidence'] for obj in objects])
            properties['total_refracted_area'] = sum([obj['area'] for obj in objects])
        
        return properties
    
    def _compute_gradient_coherence(self, magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
        window_size = 7
        coherence = np.zeros_like(magnitude)
        
        for i in range(window_size//2, magnitude.shape[0] - window_size//2):
            for j in range(window_size//2, magnitude.shape[1] - window_size//2):
                window_dir = direction[i-window_size//2:i+window_size//2+1,
                                     j-window_size//2:j+window_size//2+1]
                window_mag = magnitude[i-window_size//2:i+window_size//2+1,
                                      j-window_size//2:j+window_size//2+1]
                
                mean_dir = np.arctan2(np.sum(np.sin(window_dir) * window_mag),
                                     np.sum(np.cos(window_dir) * window_mag))
                
                coherence[i, j] = np.mean(np.cos(window_dir - mean_dir) * window_mag) / (np.mean(window_mag) + 1e-10)
        
        return coherence
    
    def _compute_structure_tensor(self, image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        Ixx = cv2.GaussianBlur(Ix * Ix, (0, 0), sigma)
        Ixy = cv2.GaussianBlur(Ix * Iy, (0, 0), sigma)
        Iyy = cv2.GaussianBlur(Iy * Iy, (0, 0), sigma)
        
        return np.stack([Ixx, Ixy, Iyy], axis=-1)
    
    def _structure_tensor_eigenvalues(self, tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Ixx = tensor[:, :, 0]
        Ixy = tensor[:, :, 1]
        Iyy = tensor[:, :, 2]
        
        trace = Ixx + Iyy
        det = Ixx * Iyy - Ixy * Ixy
        
        gap = np.sqrt(np.maximum(0, trace**2 - 4*det))
        
        lambda1 = (trace + gap) / 2
        lambda2 = (trace - gap) / 2
        
        return lambda1, lambda2
    
    def _compute_phase_congruency(self, image: np.ndarray) -> np.ndarray:
        scales = 4
        orientations = 6
        
        pc = np.zeros_like(image, dtype=np.float32)
        
        for scale in range(scales):
            sigma = 2 ** scale
            for orientation in range(orientations):
                theta = orientation * np.pi / orientations
                
                kernel_real = cv2.getGaborKernel((21, 21), sigma, theta, 10.0, 0.5, 0, cv2.CV_32F)
                kernel_imag = cv2.getGaborKernel((21, 21), sigma, theta, 10.0, 0.5, np.pi/2, cv2.CV_32F)
                
                response_real = cv2.filter2D(image, cv2.CV_32F, kernel_real)
                response_imag = cv2.filter2D(image, cv2.CV_32F, kernel_imag)
                
                amplitude = np.sqrt(response_real**2 + response_imag**2)
                pc += amplitude
        
        pc /= (scales * orientations)
        return pc
    
    def _extract_region_features(self, mask: np.ndarray, image: np.ndarray) -> List[np.ndarray]:
        features = []
        labeled = measure.label(mask > 0)
        
        for region in measure.regionprops(labeled):
            region_features = [
                region.area,
                region.perimeter,
                region.eccentricity,
                region.solidity,
                region.extent,
                region.major_axis_length,
                region.minor_axis_length
            ]
            features.append(np.array(region_features))
        
        return features
    
    def _extract_object_features(self, mask_roi: np.ndarray, depth_roi: np.ndarray,
                                image_roi: np.ndarray, region) -> Dict:
        features = {
            'geometric': {
                'area': region.area,
                'perimeter': region.perimeter,
                'eccentricity': region.eccentricity,
                'solidity': region.solidity,
                'extent': region.extent
            },
            'depth': {
                'mean': np.mean(depth_roi[mask_roi > 0]) if np.any(mask_roi > 0) else 0,
                'std': np.std(depth_roi[mask_roi > 0]) if np.any(mask_roi > 0) else 0,
                'min': np.min(depth_roi[mask_roi > 0]) if np.any(mask_roi > 0) else 0,
                'max': np.max(depth_roi[mask_roi > 0]) if np.any(mask_roi > 0) else 0
            },
            'intensity': {
                'mean': np.mean(image_roi[mask_roi > 0]) if np.any(mask_roi > 0) else 0,
                'std': np.std(image_roi[mask_roi > 0]) if np.any(mask_roi > 0) else 0
            }
        }
        return features
    
    def _classify_transparent_object(self, features: Dict) -> str:
        eccentricity = features['geometric']['eccentricity']
        solidity = features['geometric']['solidity']
        area = features['geometric']['area']
        
        if eccentricity < 0.3 and solidity > 0.9:
            return 'circular_glass'
        elif eccentricity > 0.8:
            return 'elongated_glass'
        elif area > 10000:
            return 'window_pane'
        elif solidity < 0.7:
            return 'irregular_glass'
        else:
            return 'generic_transparent'
    
    def _calculate_advanced_confidence(self, features: Dict) -> float:
        geometric_score = min(features['geometric']['solidity'], 1.0)
        
        depth_consistency = 1.0 - min(features['depth']['std'] / (features['depth']['mean'] + 1e-10), 1.0)
        
        intensity_contrast = min(features['intensity']['std'] / 128.0, 1.0)
        
        confidence = geometric_score * 0.4 + depth_consistency * 0.3 + intensity_contrast * 0.3
        
        return min(max(confidence, 0.0), 1.0)
    
    def _region_to_contour(self, region, shape: Tuple[int, int]) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        coords = region.coords
        mask[coords[:, 0], coords[:, 1]] = 255
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return contours[0] if contours else np.array([])
    
    def _compute_stereo_depth_advanced(self, left: np.ndarray, right: np.ndarray,
                                      mask: np.ndarray) -> np.ndarray:
        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) if len(left.shape) == 3 else left
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) if len(right.shape) == 3 else right
        
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,
            blockSize=11,
            P1=8 * 3 * 11**2,
            P2=32 * 3 * 11**2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=100,
            speckleRange=32
        )
        
        disparity = stereo.compute(left_gray, right_gray)
        
        disparity = cv2.medianBlur(disparity.astype(np.float32), 5)
        
        focal_length = 700
        baseline = 0.1
        depth = (focal_length * baseline) / (disparity + 1e-6)
        depth[disparity <= 0] = 0
        
        depth = np.where(mask > 0, depth, 0)
        
        return depth.astype(np.float32)
    
    def _apply_depth_refinement(self, depth_map: np.ndarray, mask: np.ndarray,
                               image: np.ndarray) -> np.ndarray:
        depth_refined = depth_map.copy()
        
        depth_refined = cv2.bilateralFilter(depth_refined.astype(np.float32), 9, 75, 75)
        
        kernel = np.ones((5, 5), np.uint8)
        mask_dilated = cv2.dilate(mask, kernel, iterations=1)
        depth_refined = np.where(mask_dilated > 0, depth_refined, 0)
        
        labeled = measure.label(mask > 0)
        for region in measure.regionprops(labeled):
            bbox = region.bbox
            depth_roi = depth_refined[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            mask_roi = mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            
            if np.any(mask_roi > 0):
                mean_depth = np.mean(depth_roi[mask_roi > 0])
                std_depth = np.std(depth_roi[mask_roi > 0])
                
                outliers = np.abs(depth_roi - mean_depth) > 2 * std_depth
                depth_roi[outliers & (mask_roi > 0)] = mean_depth
        
        return depth_refined
    
    def _generate_cache_key(self, image: np.ndarray) -> str:
        return f"{image.shape}_{np.mean(image):.2f}_{np.std(image):.2f}"
    
    def _update_adaptive_thresholds(self, objects: List[Dict]):
        if not objects:
            return
        
        avg_confidence = np.mean([obj['confidence'] for obj in objects])
        
        if avg_confidence > 0.7:
            for key in self.adaptive_thresholds:
                self.adaptive_thresholds[key] *= 0.95
        elif avg_confidence < 0.4:
            for key in self.adaptive_thresholds:
                self.adaptive_thresholds[key] *= 1.05
        
        for key in self.adaptive_thresholds:
            self.adaptive_thresholds[key] = np.clip(self.adaptive_thresholds[key], 0.2, 0.8)
    
    def _compute_global_statistics(self, preprocessed: Dict) -> Dict:
        stats = {}
        gray = preprocessed['gray']
        
        stats['mean'] = np.mean(gray)
        stats['std'] = np.std(gray)
        stats['entropy'] = -np.sum(gray * np.log2(gray + 1e-10)) / gray.size
        stats['skewness'] = np.mean(((gray - stats['mean']) / stats['std']) ** 3)
        stats['kurtosis'] = np.mean(((gray - stats['mean']) / stats['std']) ** 4) - 3
        
        return stats
    
    def _analyze_polarization(self, normal: np.ndarray, polarized: np.ndarray) -> np.ndarray:
        normal_gray = cv2.cvtColor(normal, cv2.COLOR_BGR2GRAY) if len(normal.shape) == 3 else normal
        polarized_gray = cv2.cvtColor(polarized, cv2.COLOR_BGR2GRAY) if len(polarized.shape) == 3 else polarized
        
        difference = cv2.absdiff(normal_gray, polarized_gray)
        
        difference = cv2.GaussianBlur(difference, (5, 5), 1.0)
        
        threshold = np.percentile(difference, 85)
        polarization_map = (difference > threshold).astype(np.uint8) * 255
        
        return polarization_map


class PhysicsBasedRefractionModel:
    def __init__(self):
        self.n_glass = 1.52
        self.n_air = 1.0
        self.n_water = 1.33
        
    def estimate_depth_from_refraction(self, image: np.ndarray, mask: np.ndarray,
                                      refraction_map: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        depth_map = np.zeros((h, w), dtype=np.float32)
        
        gradient_x = cv2.Sobel(refraction_map, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(refraction_map, cv2.CV_64F, 0, 1, ksize=3)
        
        curvature = np.sqrt(gradient_x**2 + gradient_y**2)
        
        critical_angle = np.arcsin(self.n_air / self.n_glass)
        
        incidence_angle = np.arctan(curvature / (np.max(curvature) + 1e-10)) * critical_angle
        
        refraction_angle = np.arcsin((self.n_air / self.n_glass) * np.sin(incidence_angle))
        
        deviation = incidence_angle - refraction_angle
        
        baseline = 0.1
        depth = baseline / (np.tan(deviation) + 1e-10)
        
        depth = np.clip(depth, 0, 10)
        
        depth = np.where(mask > 0, depth, 0)
        
        return depth


class MLConfidenceScorer:
    def __init__(self):
        self.weights = np.random.randn(7) * 0.1
        self.bias = 0.5
        
    def predict(self, features_list: List[np.ndarray]) -> np.ndarray:
        scores = []
        for features in features_list:
            score = np.dot(self.weights, features) + self.bias
            score = 1 / (1 + np.exp(-score))
            scores.append(score)
        return np.array(scores)


class GaborFeatureExtractor:
    def extract(self, image: np.ndarray) -> np.ndarray:
        kernels = []
        for theta in np.arange(0, np.pi, np.pi / 4):
            kernel = cv2.getGaborKernel((21, 21), 4.0, theta, 10.0, 0.5, 0)
            kernels.append(kernel)
        
        responses = []
        for kernel in kernels:
            filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
            responses.append(filtered)
        
        return np.mean(np.stack(responses), axis=0)


class LBPFeatureExtractor:
    def extract(self, image: np.ndarray) -> np.ndarray:
        radius = 1
        n_points = 8
        
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        lbp = local_binary_pattern(image, n_points, radius, method='uniform')
        
        return lbp.astype(np.float32) / lbp.max() if lbp.max() > 0 else lbp.astype(np.float32)


class HOGFeatureExtractor:
    def extract(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        orientations = 9
        pixels_per_cell = (8, 8)
        cells_per_block = (2, 2)
        
        hog_features, hog_image = hog(image, orientations=orientations,
                                     pixels_per_cell=pixels_per_cell,
                                     cells_per_block=cells_per_block,
                                     visualize=True)
        
        return hog_image


class WaveletFeatureExtractor:
    def extract(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        coeffs = cv2.pyrDown(image)
        coeffs = cv2.pyrUp(coeffs, dstsize=(image.shape[1], image.shape[0]))
        
        detail = cv2.absdiff(image, coeffs)
        
        return detail.astype(np.float32)


class FractalDimensionExtractor:
    def extract(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        threshold = np.mean(image)
        binary = image > threshold
        
        pixels = np.array(np.where(binary)).T
        
        if len(pixels) == 0:
            return np.zeros_like(image, dtype=np.float32)
        
        scales = np.logspace(0.01, 1, num=10, base=2)
        counts = []
        
        for scale in scales:
            scaled = (pixels / scale).astype(int)
            unique = np.unique(scaled, axis=0)
            counts.append(len(unique))
        
        coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
        fractal_dim = -coeffs[0]
        
        result = np.ones_like(image, dtype=np.float32) * fractal_dim
        
        return result