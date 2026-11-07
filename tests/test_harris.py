"""Flat, edge, and square checks for the Harris reference.

The numeric cases follow demos/harris.test.js on the website. They use the same
sizes and parameters, including the straight edge and the bright square.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_harris():
    path = ROOT / "05_computer_vision/harris_corner_detection/harris.py"
    spec = importlib.util.spec_from_file_location("harris_corner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["harris_corner"] = module
    spec.loader.exec_module(module)
    return module


def test_gray_weights_are_not_rounded() -> None:
    harris = load_harris()
    rgb = np.zeros((1, 1, 3), dtype=np.uint8)
    rgb[0, 0] = (10, 20, 30)
    gray = harris.to_gray(rgb)
    assert gray.dtype == np.float64
    assert gray[0, 0] == 0.299 * 10 + 0.587 * 20 + 0.114 * 30


def test_flat_image_has_no_corners() -> None:
    harris = load_harris()
    gray = np.full((18, 24), 40.0)
    result = harris.harris(gray, ksize_d=1, sigma_d=1, ksize_i=5, sigma_i=1.2, alpha=0.04, rel_thresh=0.01, nms=3)
    assert np.all(np.abs(result["R"]) < 1e-8)
    assert np.all(result["corners"] == 0)
    assert np.all(result["ixy"] == 0)


def test_vertical_edge_is_negative_and_ix_points_right() -> None:
    harris = load_harris()
    gray = np.full((21, 31), 20.0)
    gray[:, 16:] = 200.0
    result = harris.harris(gray, ksize_d=1, sigma_d=0, ksize_i=5, sigma_i=1, alpha=0.06, rel_thresh=0.05, nms=3)
    assert result["R"][10, 16] < 0
    assert result["ix"][10, 16] > 0


def test_square_corners_are_stronger_than_edges_and_the_interior() -> None:
    harris = load_harris()
    gray = np.zeros((36, 36))
    gray[8:28, 8:28] = 220.0
    result = harris.harris(gray, ksize_d=1, sigma_d=0, ksize_i=5, sigma_i=1.2, alpha=0.04, rel_thresh=0.05, nms=3)
    corner = result["R"][5:12, 5:12].max()
    edge = result["R"][8, 18]
    flat = result["R"][18, 18]
    assert corner > 0
    assert corner > edge * 2
    assert abs(flat) < corner * 0.05
    marked = np.argwhere(result["corners"])
    assert len(marked) >= 4
    assert not np.any((marked[:, 0] > 12) & (marked[:, 0] < 24) & (marked[:, 1] > 12) & (marked[:, 1] < 24))
    assert result["ix"][27, 27] != 0 or result["iy"][27, 27] != 0
    assert float(np.max(result["R"])) > 1.0


def test_defaults_still_mark_the_square_and_reject_a_straight_edge() -> None:
    harris = load_harris()
    assert harris.DEFAULTS == {
        "ksize_d": 3,
        "sigma_d": 1.0,
        "ksize_i": 5,
        "sigma_i": 1.5,
        "alpha": 0.04,
        "rel_thresh": 0.08,
        "nms": 5,
    }
    square = np.zeros((36, 36))
    square[8:28, 8:28] = 220.0
    found = harris.harris(square)
    assert np.count_nonzero(found["corners"]) >= 4

    edge = np.full((21, 31), 20.0)
    edge[:, 16:] = 200.0
    straight = harris.harris(edge)
    assert straight["R"][10, 16] < 0
    assert np.count_nonzero(straight["corners"]) == 0


def test_tie_keeps_the_smaller_row_major_index() -> None:
    harris = load_harris()
    score = np.zeros((3, 3))
    score[1, 1] = 5
    score[1, 2] = 5
    kept = harris.suppress(score, 3, 1)
    assert kept[1, 1] == 255
    assert kept[1, 2] == 0

    border = np.zeros((2, 2))
    border[0, 1] = 3
    assert harris.suppress(border, 3, 1)[0, 1] == 255


def test_normalized_gaussian_preserves_a_constant() -> None:
    harris = load_harris()
    image = np.full((9, 9), 10.4)
    assert np.allclose(harris.gaussian_blur(image, 5, 1.5), 10.4)
    assert np.allclose(harris.gaussian_kernel(5, 0.0), harris.gaussian_kernel(5, 1.0))
