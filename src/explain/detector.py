"""Classical microcalcification speck detector, spatial clustering, and high-resolution ROI extraction.

Adheres strictly to AGENTS.md Phase P3b, Rule 9 Geometry, and Decision D-010:
  - Source resolution: Detects and crops from native resolution standardized tissue (via crop_box, flip, scale).
  - High-frequency detection: Morphological white top-hat filter tuned for microcalcification specks (< 0.5 mm).
  - Spatial clustering: DBSCAN aggregation merging candidate specks into diagnostic clusters (not isolated noise).
  - Rule 9 coordinate mapping: Cluster centroids and bounding boxes map bi-directionally between native and 512x512 space.
  - High-resolution ROI extraction: Fixed-window 256x256 crops centered at top K clusters (K <= 16) for Multiple Instance Learning.
  - By-product radiological descriptors: Non-deep-learning cluster descriptors (speck count, area, density, grid cell) for report generation.
  - Combined localization: Generates 512x512 detector density map for multiplicative CAM sharpening:
        Map_final = CAM * (epsilon + Detector_Density).
"""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

from src.explain.grid import BreastGrid


def get_grid_cell_name(grid: BreastGrid | None, point_yx: Tuple[float, float]) -> str:
    """Find anatomical cell name for a (y, x) coordinate in 512x512 space."""
    if grid is None:
        return "Central Middle"
    y, x = point_yx
    for cell in grid.cells:
        y0, x0, y1, x1 = cell.bbox
        if y0 <= y <= y1 and x0 <= x <= x1:
            return cell.cell_name.replace("-", " ").title()
    return "Central Middle"


@dataclass
class SpeckCluster:
    """Diagnostic cluster of microcalcification specks."""
    cluster_id: int
    speck_count: int
    centroid_native: Tuple[float, float]  # (x, y) in native cropped pixels
    centroid_512: Tuple[float, float]    # (x, y) in 512x512 space
    bbox_native: Tuple[int, int, int, int]  # (ymin, xmin, ymax, xmax) native
    bbox_512: Tuple[float, float, float, float]  # (ymin, xmin, ymax, xmax) 512
    area_native: float  # Convex hull or bounding box area in native px^2
    density: float      # specks per unit area
    mean_intensity: float
    grid_cell: str      # Name of 3x3 anatomical grid cell (e.g. "Upper Posterior")
    score: float        # Ranking score = speck_count * mean_intensity
    roi_patch_256: np.ndarray | None = None  # 256x256 native patch


class MicrocalcificationDetector:
    """Classical high-resolution microcalcification detector and cluster extractor."""

    def __init__(
        self,
        kernel_size: int = 7,
        std_multiplier: float = 3.5,
        min_speck_area: int = 2,
        max_speck_area: int = 150,
        cluster_eps_native: float = 80.0,
        cluster_min_specks: int = 3,
        max_clusters_per_image: int = 12,
        patch_size: int = 256,
    ):
        self.kernel_size = kernel_size
        self.std_multiplier = std_multiplier
        self.min_speck_area = min_speck_area
        self.max_speck_area = max_speck_area
        self.cluster_eps_native = cluster_eps_native
        self.cluster_min_specks = cluster_min_specks
        self.max_clusters_per_image = max_clusters_per_image
        self.patch_size = patch_size

    def detect_specks(
        self,
        cropped_tissue: np.ndarray,
        tissue_mask: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect high-frequency microcalcification specks via morphological white top-hat.

        Args:
            cropped_tissue: 2D float32 image in [0, 1] (standardized native crop).
            tissue_mask: Optional binary mask (1=tissue, 0=background).

        Returns:
            centroids: (N, 2) array of (x, y) speck centroids in native coordinates.
            tophat_map: 2D float32 top-hat response map.
        """
        if tissue_mask is None:
            tissue_mask = cropped_tissue > 0.05

        # Morphological white top-hat extracts bright structures smaller than kernel
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
        tophat = cv2.morphologyEx(cropped_tissue, cv2.MORPH_TOPHAT, k)

        tissue_vals = tophat[tissue_mask]
        if len(tissue_vals) == 0:
            return np.empty((0, 2), dtype=np.float32), tophat

        mean_v = float(np.mean(tissue_vals))
        std_v = float(np.std(tissue_vals))
        thresh = mean_v + self.std_multiplier * max(1e-6, std_v)

        speck_bin = ((tophat > thresh) & tissue_mask).astype(np.uint8)

        # Connected component analysis for individual speck filtering
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(speck_bin)

        valid_centroids = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if self.min_speck_area <= area <= self.max_speck_area:
                valid_centroids.append(centroids[i])

        if len(valid_centroids) == 0:
            return np.empty((0, 2), dtype=np.float32), tophat

        return np.array(valid_centroids, dtype=np.float32), tophat

    def cluster_specks(
        self,
        speck_centroids: np.ndarray,
        cropped_tissue: np.ndarray,
        scale: float,
        grid: BreastGrid | None = None,
        extract_patches: bool = True,
    ) -> List[SpeckCluster]:
        """Group detected specks into spatial diagnostic clusters via DBSCAN.

        Args:
            speck_centroids: (N, 2) coordinates in native space.
            cropped_tissue: 2D native tissue array.
            scale: Scale factor mapping native coordinates to 512 space (512 / max(H, W)).
            grid: Optional 3x3 AnatomicalGrid for mapping clusters to breast cells.
            extract_patches: Whether to crop 256x256 native patches around cluster centers.

        Returns:
            List of SpeckCluster objects sorted by diagnostic significance score.
        """
        if len(speck_centroids) < self.cluster_min_specks:
            return []

        clustering = DBSCAN(
            eps=self.cluster_eps_native,
            min_samples=self.cluster_min_specks,
        ).fit(speck_centroids)

        labels = clustering.labels_
        unique_labels = set(labels) - {-1}

        clusters: List[SpeckCluster] = []
        h_nat, w_nat = cropped_tissue.shape

        for cid in unique_labels:
            pts = speck_centroids[labels == cid]
            n_specks = len(pts)

            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))

            # Bounding box in native coordinates
            xmin_n = int(np.floor(np.min(pts[:, 0])))
            xmax_n = int(np.ceil(np.max(pts[:, 0])))
            ymin_n = int(np.floor(np.min(pts[:, 1])))
            ymax_n = int(np.ceil(np.max(pts[:, 1])))

            area_n = max(1.0, float((xmax_n - xmin_n + 1) * (ymax_n - ymin_n + 1)))
            density = float(n_specks / area_n)

            # Map to 512x512 space (Rule 9 Geometry)
            c512_x = cx * scale
            c512_y = cy * scale
            bbox_512 = (
                float(ymin_n * scale),
                float(xmin_n * scale),
                float(ymax_n * scale),
                float(xmax_n * scale),
            )

            # Sample mean pixel intensity of cluster specks
            intensities = []
            for px, py in pts:
                ix = int(np.clip(round(px), 0, w_nat - 1))
                iy = int(np.clip(round(py), 0, h_nat - 1))
                intensities.append(cropped_tissue[iy, ix])
            mean_intensity = float(np.mean(intensities)) if intensities else 0.5

            # Map to 3x3 anatomical grid cell
            grid_cell = get_grid_cell_name(grid, (c512_y, c512_x))

            # Diagnostic significance score
            score = float(n_specks * mean_intensity)

            # Extract 256x256 native patch centered at cluster
            roi_patch = None
            if extract_patches:
                roi_patch = self._crop_fixed_window(cropped_tissue, cx, cy, self.patch_size)

            cluster = SpeckCluster(
                cluster_id=int(cid),
                speck_count=n_specks,
                centroid_native=(cx, cy),
                centroid_512=(c512_x, c512_y),
                bbox_native=(ymin_n, xmin_n, ymax_n, xmax_n),
                bbox_512=bbox_512,
                area_native=area_n,
                density=density,
                mean_intensity=mean_intensity,
                grid_cell=grid_cell,
                score=score,
                roi_patch_256=roi_patch,
            )
            clusters.append(cluster)

        # Sort by score descending and cap at max_clusters_per_image
        clusters.sort(key=lambda c: -c.score)
        return clusters[: self.max_clusters_per_image]

    def _crop_fixed_window(
        self,
        img: np.ndarray,
        center_x: float,
        center_y: float,
        window_size: int,
    ) -> np.ndarray:
        """Crop a fixed window around (center_x, center_y) with boundary zero-padding."""
        h, w = img.shape
        half = window_size // 2

        cx = int(round(center_x))
        cy = int(round(center_y))

        y0 = cy - half
        y1 = y0 + window_size
        x0 = cx - half
        x1 = x0 + window_size

        patch = np.zeros((window_size, window_size), dtype=img.dtype)

        # Compute overlap with source image
        src_y0 = max(0, y0)
        src_y1 = min(h, y1)
        src_x0 = max(0, x0)
        src_x1 = min(w, x1)

        dst_y0 = src_y0 - y0
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        dst_x0 = src_x0 - x0
        dst_x1 = dst_x0 + (src_x1 - src_x0)

        if src_y1 > src_y0 and src_x1 > src_x0:
            patch[dst_y0:dst_y1, dst_x0:dst_x1] = img[src_y0:src_y1, src_x0:src_x1]

        return patch

    def compute_detector_density_map(
        self,
        clusters: List[SpeckCluster],
        target_size: Tuple[int, int] = (512, 512),
        sigma: float = 12.0,
    ) -> np.ndarray:
        """Generate smooth 2D detector density heatmap in 512x512 space.

        Used for multiplicative localization sharpening:
            Map_final = CAM * (epsilon + Detector_Density)
        """
        density_map = np.zeros(target_size, dtype=np.float32)
        if not clusters:
            return density_map

        h, w = target_size
        for c in clusters:
            cx, cy = c.centroid_512
            ix = int(round(cx))
            iy = int(round(cy))
            weight = float(np.log1p(c.speck_count))

            # Add Gaussian splat at cluster centroid
            rad = int(round(3 * sigma))
            y_min = max(0, iy - rad)
            y_max = min(h, iy + rad + 1)
            x_min = max(0, ix - rad)
            x_max = min(w, ix + rad + 1)

            for yy in range(y_min, y_max):
                for xx in range(x_min, x_max):
                    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
                    density_map[yy, xx] += weight * np.exp(-dist_sq / (2.0 * sigma ** 2))

        # Normalize to [0.0, 1.0]
        max_v = float(density_map.max())
        if max_v > 0.0:
            density_map /= max_v

        return density_map

    @staticmethod
    def combine_cam_and_detector(
        cam_heatmap: np.ndarray,
        detector_density: np.ndarray,
        epsilon: float = 0.20,
    ) -> np.ndarray:
        """Multiply coarse Grad-CAM with fine detector density.

        Preserves overall attention where detector finds no specks via epsilon base weight.
        Formula:
            Map_combined = CAM * (epsilon + (1 - epsilon) * Detector_Density)
        """
        modulator = epsilon + (1.0 - epsilon) * detector_density
        product = cam_heatmap * modulator

        p_max = float(product.max())
        if p_max > 0.0:
            product /= p_max
        return product

    @staticmethod
    def generate_roi_descriptors(clusters: List[SpeckCluster]) -> List[str]:
        """Generate non-deep-learning structured sentences describing detected clusters."""
        descriptors = []
        for idx, c in enumerate(clusters):
            desc = (
                f"Cluster {idx + 1}: {c.speck_count} microcalcification specks identified "
                f"in the {c.grid_cell} quadrant (area: {c.area_native:.0f} native px², "
                f"density: {c.density:.3f} specks/px²)."
            )
            descriptors.append(desc)
        return descriptors
