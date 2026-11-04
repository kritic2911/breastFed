"""Unit tests for microcalcification detector, DBSCAN clustering, and ROI extraction."""

import numpy as np
import pytest

from src.explain.detector import MicrocalcificationDetector, SpeckCluster
from src.explain.grid import fit_anatomical_grid


def test_detector_speck_detection_and_clustering():
    detector = MicrocalcificationDetector(
        kernel_size=5,
        std_multiplier=2.0,
        cluster_eps_native=30.0,
        cluster_min_specks=3,
        max_clusters_per_image=5,
        patch_size=64,
    )

    # Create synthetic tissue with 2 simulated microcalcification clusters
    h, w = 400, 300
    tissue = np.zeros((h, w), dtype=np.float32) + 0.20  # Background tissue
    mask = np.ones((h, w), dtype=np.uint8)

    # Cluster 1 around (100, 150): 5 bright specks
    pts_c1 = [(100, 150), (102, 152), (98, 148), (105, 155), (101, 149)]
    for x, y in pts_c1:
        tissue[y, x] = 0.95
        tissue[y + 1, x] = 0.90

    # Cluster 2 around (200, 250): 4 bright specks
    pts_c2 = [(200, 250), (202, 251), (198, 248), (205, 253)]
    for x, y in pts_c2:
        tissue[y, x] = 0.90
        tissue[y, x + 1] = 0.85

    specks, tophat = detector.detect_specks(tissue, mask)
    assert len(specks) >= 7, f"Expected at least 7 specks, got {len(specks)}"

    scale = 512.0 / max(h, w)
    grid = fit_anatomical_grid(np.zeros((512, 512), dtype=np.float32), view="CC")

    clusters = detector.cluster_specks(specks, tissue, scale=scale, grid=grid, extract_patches=True)
    assert len(clusters) >= 2, f"Expected at least 2 clusters, got {len(clusters)}"

    # Check first cluster properties
    c1 = clusters[0]
    assert c1.speck_count >= 3
    assert c1.roi_patch_256.shape == (64, 64)
    assert 0 <= c1.centroid_512[0] <= 512
    assert 0 <= c1.centroid_512[1] <= 512

    # Check descriptors
    descs = detector.generate_roi_descriptors(clusters)
    assert len(descs) == len(clusters)
    assert "microcalcification specks" in descs[0]


def test_detector_density_map_and_multiplication():
    detector = MicrocalcificationDetector()
    clusters = [
        SpeckCluster(
            cluster_id=1,
            speck_count=15,
            centroid_native=(400.0, 600.0),
            centroid_512=(100.0, 150.0),
            bbox_native=(580, 380, 620, 420),
            bbox_512=(145.0, 95.0, 155.0, 105.0),
            area_native=1600.0,
            density=15.0 / 1600.0,
            mean_intensity=0.85,
            grid_cell="Central Middle",
            score=15.0 * 0.85,
        )
    ]

    density_map = detector.compute_detector_density_map(clusters, target_size=(512, 512), sigma=10.0)
    assert density_map.shape == (512, 512)
    assert 0.0 <= density_map.min() <= density_map.max() <= 1.0
    # Peak should be at (150, 100) (y, x)
    assert density_map[150, 100] == pytest.approx(1.0, abs=1e-3)

    # Multiplicative sharpening test
    cam = np.ones((512, 512), dtype=np.float32) * 0.5
    combined = detector.combine_cam_and_detector(cam, density_map, epsilon=0.20)
    assert combined.shape == (512, 512)
    assert combined.max() == pytest.approx(1.0, abs=1e-3)
    # Combined at peak should be higher than combined far away
    assert combined[150, 100] > combined[0, 0]

