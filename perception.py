import numpy as np
import cv2
from matplotlib import pyplot as plt
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Any
import warnings

class CameraCalibration:
    """Handles camera calibration parameters and coordinate transformations."""
    
    def __init__(self, k_matrix: np.ndarray):
        """
        Initialize camera calibration.
        
        Args:
            k_matrix: 3x3 camera intrinsic matrix
        """
        self.k = k_matrix
        self.f = k_matrix[0, 0]  # focal length
        self.c_u = k_matrix[0, 2]  # principal point x
        self.c_v = k_matrix[1, 2]  # principal point y
    
    def pixel_to_camera_coords(self, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert pixel coordinates to camera coordinate frame.
        
        Args:
            depth: (H, W) depth map in meters
            
        Returns:
            Tuple of (x, y) coordinates in camera frame
        """
        H, W = depth.shape
        
        # Create coordinate grids
        u_coords, v_coords = np.meshgrid(np.arange(W), np.arange(H))
        
        # Convert to camera coordinates using vectorized operations
        x = (u_coords - self.c_u) * depth / self.f
        y = (v_coords - self.c_v) * depth / self.f
        
        return x, y
    
    def compute_distances_from_origin(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Compute Euclidean distances from camera origin."""
        return np.sqrt(x**2 + y**2 + z**2)


class PlaneEstimator:
    """Handles ground plane estimation using RANSAC."""
    
    def __init__(self, max_iterations: int = 100, distance_threshold: float = 0.01, 
                 min_inlier_ratio: float = 0.5):
        self.max_iterations = max_iterations
        self.distance_threshold = distance_threshold
        self.min_inlier_ratio = min_inlier_ratio
    
    def fit_plane(self, xyz_data: np.ndarray) -> np.ndarray:
        """
        Fit plane using RANSAC algorithm.
        
        Args:
            xyz_data: (3, N) array of 3D points
            
        Returns:
            Plane parameters [a, b, c, d] for ax + by + cz + d = 0
        """
        min_num_inliers = int(xyz_data.shape[1] * self.min_inlier_ratio)
        best_inliers = 0
        best_plane = None
        best_inlier_indices = None
        
        for _ in range(self.max_iterations):
            # Randomly sample 3 points
            sample_indices = np.random.choice(xyz_data.shape[1], 3, replace=False)
            sample_points = xyz_data[:, sample_indices]
            
            # Compute plane from 3 points
            plane = self._compute_plane_from_points(sample_points)
            
            # Find inliers
            distances = self._distance_to_plane(plane, xyz_data)
            inlier_indices = np.where(distances < self.distance_threshold)[0]
            num_inliers = len(inlier_indices)
            
            # Update best plane if current is better
            if num_inliers > best_inliers:
                best_inliers = num_inliers
                best_inlier_indices = inlier_indices
                
                # Early termination if enough inliers found
                if num_inliers >= min_num_inliers:
                    break
        
        # Recompute plane using all inliers
        if best_inlier_indices is not None:
            best_plane = self._compute_plane_from_points(xyz_data[:, best_inlier_indices])
        
        return best_plane
    
    def _compute_plane_from_points(self, points: np.ndarray) -> np.ndarray:
        """Compute plane coefficients from 3 or more 3D points."""
        if points.shape[1] < 3:
            raise ValueError("Need at least 3 points to fit a plane")
        
        # Use SVD for robust plane fitting
        centroid = np.mean(points, axis=1, keepdims=True)
        centered_points = points - centroid
        
        _, _, vh = np.linalg.svd(centered_points)
        normal = vh[-1, :]  # Last row of V^T is the normal vector
        
        # Plane equation: n · (p - centroid) = 0 => n · p - n · centroid = 0
        d = -np.dot(normal, centroid.flatten())
        
        return np.array([normal[0], normal[1], normal[2], d])
    
    def _distance_to_plane(self, plane: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Compute distance from points to plane."""
        a, b, c, d = plane
        x, y, z = points[0], points[1], points[2]
        return np.abs(a * x + b * y + c * z + d) / np.sqrt(a**2 + b**2 + c**2)


class DrivableSpaceEstimator:
    """Estimates drivable space from semantic segmentation and depth."""
    
    def __init__(self, camera_calib: CameraCalibration, road_class_id: int = 7):
        self.camera_calib = camera_calib
        self.road_class_id = road_class_id
        self.plane_estimator = PlaneEstimator()
    
    def estimate_drivable_space(self, segmentation: np.ndarray, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate drivable space mask and ground plane.
        
        Args:
            segmentation: (H, W) semantic segmentation output
            depth: (H, W) depth map
            
        Returns:
            Tuple of (ground_mask, ground_plane_params)
        """
        # Convert pixel coordinates to camera coordinates
        x, y = self.camera_calib.pixel_to_camera_coords(depth)
        
        # Extract road pixels
        road_mask = (segmentation == self.road_class_id)
        
        # Get 3D coordinates of road pixels
        x_ground = x[road_mask]
        y_ground = y[road_mask] 
        z_ground = depth[road_mask]
        xyz_ground = np.stack([x_ground, y_ground, z_ground])
        
        # Estimate ground plane using RANSAC
        ground_plane = self.plane_estimator.fit_plane(xyz_ground)
        
        # Create ground mask based on distance to plane
        distances = self.plane_estimator._distance_to_plane(ground_plane, np.stack([x, y, depth]))
        ground_mask = distances < 0.1
        
        return ground_mask, ground_plane


class LaneEstimator:
    """Estimates lane boundaries from semantic segmentation."""
    
    def __init__(self, lane_marking_class: int = 6, sidewalk_class: int = 8):
        self.lane_marking_class = lane_marking_class
        self.sidewalk_class = sidewalk_class
        
        # Line detection parameters
        self.canny_low = 100
        self.canny_high = 150
        self.hough_rho = 10
        self.hough_theta = np.pi / 180
        self.hough_threshold = 200
        self.min_line_length = 150
        self.max_line_gap = 50
        
        # Line merging parameters
        self.slope_similarity_threshold = 0.1
        self.intercept_similarity_threshold = 40
        self.min_slope_threshold = 0.3
    
    def estimate_lane_boundaries(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Estimate lane boundaries from semantic segmentation.
        
        Args:
            segmentation: (H, W) semantic segmentation output
            
        Returns:
            Array of merged lane lines in format [x1, y1, x2, y2]
        """
        # Step 1: Extract lane boundary pixels
        lane_proposals = self._extract_lane_proposals(segmentation)
        
        # Step 2: Merge similar lines and filter horizontal ones
        merged_lines = self._merge_and_filter_lines(lane_proposals)
        
        return merged_lines
    
    def _extract_lane_proposals(self, segmentation: np.ndarray) -> np.ndarray:
        """Extract lane line proposals using edge detection and Hough transform."""
        # Create lane boundary mask
        lane_mask = np.zeros(segmentation.shape, dtype=np.uint8)
        lane_mask[segmentation == self.lane_marking_class] = 255
        lane_mask[segmentation == self.sidewalk_class] = 255
        
        # Edge detection
        edges = cv2.Canny(lane_mask, self.canny_low, self.canny_high)
        
        # Line detection
        lines = cv2.HoughLinesP(
            edges, 
            rho=self.hough_rho,
            theta=self.hough_theta,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        if lines is not None:
            return lines.reshape(-1, 4)
        else:
            return np.array([]).reshape(0, 4)
    
    def _merge_and_filter_lines(self, lines: np.ndarray) -> np.ndarray:
        """Merge similar lines and filter out horizontal lines."""
        if len(lines) == 0:
            return lines
        
        # Get slopes and intercepts
        slopes, intercepts = self._get_slope_intercept(lines)
        
        # Filter out horizontal lines
        valid_slope_mask = np.abs(slopes) > self.min_slope_threshold
        
        # Cluster lines by slope and intercept similarity
        clusters = self._cluster_lines(lines, slopes, intercepts, valid_slope_mask)
        
        # Merge lines within each cluster
        merged_lines = []
        for cluster_indices in clusters:
            if len(cluster_indices) > 0:
                cluster_lines = lines[cluster_indices]
                merged_line = np.mean(cluster_lines, axis=0)
                merged_lines.append(merged_line)
        
        return np.array(merged_lines).reshape(-1, 4) if merged_lines else np.array([]).reshape(0, 4)
    
    def _get_slope_intercept(self, lines: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate slope and intercept for each line."""
        x1, y1, x2, y2 = lines[:, 0], lines[:, 1], lines[:, 2], lines[:, 3]
        
        # Handle vertical lines
        dx = x2 - x1
        dy = y2 - y1
        
        slopes = np.where(dx != 0, dy / dx, np.inf)
        intercepts = np.where(dx != 0, y1 - slopes * x1, x1)
        
        return slopes, intercepts
    
    def _cluster_lines(self, lines: np.ndarray, slopes: np.ndarray, intercepts: np.ndarray, 
                      valid_mask: np.ndarray) -> List[np.ndarray]:
        """Cluster lines based on slope and intercept similarity."""
        clusters = []
        used_indices = set()
        
        for i, (slope, intercept) in enumerate(zip(slopes, intercepts)):
            if i in used_indices or not valid_mask[i]:
                continue
                
            # Find similar lines
            slope_similar = np.abs(slopes - slope) < self.slope_similarity_threshold
            intercept_similar = np.abs(intercepts - intercept) < self.intercept_similarity_threshold
            
            cluster_mask = slope_similar & intercept_similar & valid_mask
            cluster_indices = np.where(cluster_mask)[0]
            
            # Add to clusters and mark as used
            clusters.append(cluster_indices)
            used_indices.update(cluster_indices)
        
        return clusters


class ObjectDetectionFilter:
    """Filters object detection results using semantic segmentation."""
    
    def __init__(self, ratio_threshold: float = 0.3):
        self.ratio_threshold = ratio_threshold
        self.class_mapping = {
            'Car': 10,
            'Pedestrian': 4
        }
    
    def filter_detections(self, detections: List, segmentation: np.ndarray) -> List:
        """
        Filter detections based on semantic segmentation overlap.
        
        Args:
            detections: List of detections [class, x_min, y_min, x_max, y_max, score]
            segmentation: (H, W) semantic segmentation output
            
        Returns:
            List of filtered detections
        """
        filtered_detections = []
        
        for detection in detections:
            if self._is_valid_detection(detection, segmentation):
                filtered_detections.append(detection)
        
        return filtered_detections
    
    def _is_valid_detection(self, detection: List, segmentation: np.ndarray) -> bool:
        """Check if detection is valid based on segmentation overlap."""
        class_name, x_min, y_min, x_max, y_max, score = detection
        
        # Convert to integers
        x_min, y_min = int(float(x_min)), int(float(y_min))
        x_max, y_max = int(float(x_max)), int(float(y_max))
        
        # Get expected class index
        if class_name not in self.class_mapping:
            return False
        
        expected_class = self.class_mapping[class_name]
        
        # Extract bounding box region
        box_segmentation = segmentation[y_min:y_max, x_min:x_max]
        
        # Calculate overlap ratio
        correct_pixels = np.sum(box_segmentation == expected_class)
        total_pixels = box_segmentation.size
        
        if total_pixels == 0:
            return False
        
        ratio = correct_pixels / total_pixels
        return ratio > self.ratio_threshold


class DistanceEstimator:
    """Estimates minimum distance to detected objects."""
    
    def __init__(self, camera_calib: CameraCalibration):
        self.camera_calib = camera_calib
    
    def compute_min_distances(self, detections: List, x: np.ndarray, y: np.ndarray, 
                            z: np.ndarray) -> List[float]:
        """
        Compute minimum distance to each detection.
        
        Args:
            detections: List of filtered detections
            x, y, z: Camera coordinate arrays
            
        Returns:
            List of minimum distances for each detection
        """
        min_distances = []
        
        for detection in detections:
            min_dist = self._compute_detection_min_distance(detection, x, y, z)
            min_distances.append(min_dist)
        
        return min_distances
    
    def _compute_detection_min_distance(self, detection: List, x: np.ndarray, 
                                      y: np.ndarray, z: np.ndarray) -> float:
        """Compute minimum distance for a single detection."""
        class_name, x_min, y_min, x_max, y_max, score = detection
        
        # Convert to integers
        x_min, y_min = int(float(x_min)), int(float(y_min))
        x_max, y_max = int(float(x_max)), int(float(y_max))
        
        # Extract bounding box coordinates
        box_x = x[y_min:y_max, x_min:x_max]
        box_y = y[y_min:y_max, x_min:x_max]
        box_z = z[y_min:y_max, x_min:x_max]
        
        # Compute distances
        distances = self.camera_calib.compute_distances_from_origin(box_x, box_y, box_z)
        
        return np.min(distances) if distances.size > 0 else np.inf


class PerceptionSystem:
    """Main perception system that coordinates all components."""
    
    def __init__(self, k_matrix: np.ndarray):
        """
        Initialize the perception system.
        
        Args:
            k_matrix: 3x3 camera intrinsic matrix
        """
        self.camera_calib = CameraCalibration(k_matrix)
        self.drivable_space_estimator = DrivableSpaceEstimator(self.camera_calib)
        self.lane_estimator = LaneEstimator()
        self.detection_filter = ObjectDetectionFilter()
        self.distance_estimator = DistanceEstimator(self.camera_calib)
    
    def process_frame(self, image: np.ndarray, depth: np.ndarray, 
                     segmentation: np.ndarray, detections: List) -> dict:
        """
        Process a complete frame and extract all perception information.
        
        Args:
            image: RGB image
            depth: Depth map
            segmentation: Semantic segmentation output
            detections: Object detection results
            
        Returns:
            Dictionary containing all perception results
        """
        results = {}
        
        # 1. Estimate drivable space
        ground_mask, ground_plane = self.drivable_space_estimator.estimate_drivable_space(
            segmentation, depth
        )
        results['ground_mask'] = ground_mask
        results['ground_plane'] = ground_plane
        
        # 2. Estimate lane boundaries
        lane_lines = self.lane_estimator.estimate_lane_boundaries(segmentation)
        results['lane_lines'] = lane_lines
        
        # 3. Filter detections and compute distances
        filtered_detections = self.detection_filter.filter_detections(detections, segmentation)
        results['filtered_detections'] = filtered_detections
        
        if filtered_detections:
            # Get camera coordinates
            x, y = self.camera_calib.pixel_to_camera_coords(depth)
            
            # Compute minimum distances
            min_distances = self.distance_estimator.compute_min_distances(
                filtered_detections, x, y, depth
            )
            results['min_distances'] = min_distances
        else:
            results['min_distances'] = []
        
        return results
    
    def visualize_results(self, image: np.ndarray, results: dict, 
                         dataset_handler=None) -> None:
        """
        Visualize perception results.
        
        Args:
            image: Original RGB image
            results: Results from process_frame
            dataset_handler: Optional dataset handler for visualization functions
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Ground mask
        axes[0, 1].imshow(results['ground_mask'], cmap='gray')
        axes[0, 1].set_title('Drivable Space')
        axes[0, 1].axis('off')
        
        # Lane lines (if dataset_handler available for visualization)
        if dataset_handler and len(results['lane_lines']) > 0:
            lane_vis = dataset_handler.vis_lanes(results['lane_lines'])
            axes[1, 0].imshow(lane_vis)
        else:
            axes[1, 0].imshow(image)
            # Draw lane lines manually if no dataset_handler
            for line in results['lane_lines']:
                x1, y1, x2, y2 = line.astype(int)
                axes[1, 0].plot([x1, x2], [y1, y2], 'r-', linewidth=2)
        axes[1, 0].set_title('Lane Boundaries')
        axes[1, 0].axis('off')
        
        # Object detections with distances
        if dataset_handler and results['filtered_detections']:
            detection_vis = dataset_handler.vis_object_detection(results['filtered_detections'])
            axes[1, 1].imshow(detection_vis)
            
            # Add distance annotations
            for detection, min_distance in zip(results['filtered_detections'], results['min_distances']):
                bbox = np.array(detection[1:5], dtype=float)
                axes[1, 1].text(bbox[0], bbox[1] - 20, 
                               f'Distance: {min_distance:.2f}m',
                               color='red', fontsize=10, weight='bold')
        else:
            axes[1, 1].imshow(image)
        axes[1, 1].set_title('Object Detection & Distances')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print numerical results
        print("=== Perception Results ===")
        print(f"Ground Plane: {results['ground_plane']}")
        print(f"Number of Lane Lines: {len(results['lane_lines'])}")
        print(f"Number of Valid Detections: {len(results['filtered_detections'])}")
        if results['min_distances']:
            print(f"Minimum Distances: {results['min_distances']}")


# Example usage function
def main():
    """Example of how to use the PerceptionSystem class."""
    
    # Assuming you have a dataset_handler from the original code
    # dataset_handler = DatasetHandler()
    # dataset_handler.set_frame(1)
    
    # Example camera calibration matrix
    k_matrix = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ], dtype=float)
    
    # Initialize perception system
    perception_system = PerceptionSystem(k_matrix)
    
    # Process frame (replace with actual data)
    # image = dataset_handler.image
    # depth = dataset_handler.depth  
    # segmentation = dataset_handler.segmentation
    # detections = dataset_handler.object_detection
    
    # results = perception_system.process_frame(image, depth, segmentation, detections)
    
    # Visualize results
    # perception_system.visualize_results(image, results, dataset_handler)
    
    print("Perception system initialized successfully!")
    print("Use process_frame() method to analyze camera data.")
    return perception_system

# if __name__ == "__main__":
#     main()