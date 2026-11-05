from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------- k-NN
def test_knn_predicts_the_majority_class_of_the_k_nearest_points() -> None:
    knn = load_module("knn_module", "05_computer_vision/knn/knn_from_scratch.py")
    points = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [1.0, 1.0], [0.9, 1.0]])
    labels = np.array([0, 0, 0, 1, 1])
    model = knn.KNNClassifier(k=3).fit(points, labels)

    assert model.predict(np.array([[0.05, 0.05], [0.95, 0.95]])).tolist() == [0, 1]
    # With k = 5 the three class-0 points outvote the two class-1 points everywhere.
    assert knn.KNNClassifier(k=5).fit(points, labels).predict(np.array([[0.95, 0.95]])).tolist() == [0]


def test_knn_metrics_and_tie_break() -> None:
    knn = load_module("knn_module_metrics", "05_computer_vision/knn/knn_from_scratch.py")
    a, b = np.array([[0.0, 0.0]]), np.array([[3.0, 4.0]])
    assert knn.pairwise_distances(a, b, "euclidean")[0, 0] == pytest.approx(5.0)
    assert knn.pairwise_distances(a, b, "manhattan")[0, 0] == pytest.approx(7.0)

    # k = 2 with one neighbour of each class: the nearest neighbour decides.
    model = knn.KNNClassifier(k=2).fit(np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([0, 1]))
    assert model.predict(np.array([[0.2, 0.0]])).tolist() == [0]
    assert model.predict(np.array([[0.8, 0.0]])).tolist() == [1]


def test_knn_matches_scikit_learn_on_two_classes_with_odd_k() -> None:
    pytest.importorskip("sklearn")
    knn = load_module("knn_module_sklearn", "05_computer_vision/knn/knn_from_scratch.py")
    points, labels = knn.make_moons()
    queries = np.random.default_rng(3).uniform(0, 1, size=(500, 2))
    for metric in ("euclidean", "manhattan"):
        ours = knn.KNNClassifier(5, metric).fit(points, labels).predict(queries)
        reference = knn.predict_sklearn(points, labels, queries, 5, metric)
        assert (ours == reference).all()


# ------------------------------------------------------------------- edge detection
def test_numpy_sobel_matches_opencv() -> None:
    cv2 = pytest.importorskip("cv2")
    edges = load_module("edge_module_sobel", "05_computer_vision/edge_detection/edge_detection.py")
    gray = edges.to_gray(edges.make_sample_image())
    gx, gy = edges.sobel(gray)
    assert np.allclose(gx, cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    assert np.allclose(gy, cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))


@pytest.mark.parametrize("l2", [False, True])
def test_numpy_canny_agrees_with_opencv(l2: bool) -> None:
    cv2 = pytest.importorskip("cv2")
    edges = load_module(f"edge_module_canny_{l2}", "05_computer_vision/edge_detection/edge_detection.py")
    gray = edges.to_gray(edges.make_sample_image())
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    ours = edges.canny_from_scratch(blurred, 50, 150, l2)
    reference = cv2.Canny(blurred, 50, 150, L2gradient=l2)

    assert ours.any() and reference.any()
    assert edges.edge_agreement(ours, reference)["f1"] > 0.99


def test_vertical_edge_is_seen_by_sobel_x_only() -> None:
    edges = load_module("edge_module_direction", "05_computer_vision/edge_detection/edge_detection.py")
    image = np.zeros((20, 20), dtype=np.uint8)
    image[:, 10:] = 255                                   # dark left half, bright right half
    gx, gy = edges.sobel(image)
    assert np.abs(gx[10, 9:11]).min() > 0
    assert np.abs(gy[2:-2, :]).max() == 0
