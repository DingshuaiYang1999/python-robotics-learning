"""k-Nearest Neighbours (k-NN) from scratch, checked against scikit-learn and OpenCV.

What is k-NN?
    k-NN is one of the simplest classifiers. It does no training in the usual sense:
    it just stores all labelled points. To classify a new "query" point it

        1. measures the distance from the query to every stored point,
        2. picks the k closest ones (the "neighbours"),
        3. lets those k neighbours vote; the most common class wins.

    A small k follows the data closely (jagged decision boundary, sensitive to noise).
    A large k averages over many points (smooth boundary, can blur small classes).

Distance metrics
    Euclidean (L2):  sqrt(sum_i (a_i - b_i)^2)   straight-line distance
    Manhattan (L1):  sum_i |a_i - b_i|           "city-block" distance

Tie-breaking rule (also used by the interactive demo on the website)
    If two or more classes get the same number of votes, the class that contains the
    single nearest neighbour among the tied classes wins. Distances that are exactly
    equal are resolved by the lower point index (a stable sort).

Run it
    python knn_from_scratch.py                 # k = 5, Euclidean, writes ../output/knn_comparison.png
    python knn_from_scratch.py --k 9 --metric manhattan
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"


# --------------------------------------------------------------------------------------
# 1. The classifier
# --------------------------------------------------------------------------------------
def pairwise_distances(queries: np.ndarray, points: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Distance matrix of shape (n_queries, n_points) using NumPy broadcasting.

    ``queries[:, None, :] - points[None, :, :]`` creates an array of all coordinate
    differences with shape (n_queries, n_points, n_features) without any Python loop.
    """
    diff = queries[:, None, :] - points[None, :, :]
    if metric == "euclidean":
        return np.sqrt((diff ** 2).sum(axis=2))
    if metric == "manhattan":
        return np.abs(diff).sum(axis=2)
    raise ValueError(f"unknown metric: {metric!r} (use 'euclidean' or 'manhattan')")


class KNNClassifier:
    """Minimal k-NN classifier with the same fit / predict interface as scikit-learn."""

    def __init__(self, k: int = 5, metric: str = "euclidean") -> None:
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k = k
        self.metric = metric
        self.points_ = np.empty((0, 2))
        self.labels_ = np.empty((0,), dtype=int)

    def fit(self, points: np.ndarray, labels: np.ndarray) -> "KNNClassifier":
        """'Training' only stores the data."""
        self.points_ = np.asarray(points, dtype=float)
        self.labels_ = np.asarray(labels, dtype=int)
        return self

    def kneighbors(self, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (distances, indices) of the k nearest stored points for every query."""
        queries = np.atleast_2d(np.asarray(queries, dtype=float))
        dist = pairwise_distances(queries, self.points_, self.metric)
        k = min(self.k, len(self.points_))
        # kind="stable": equal distances keep their original order -> deterministic result.
        order = np.argsort(dist, axis=1, kind="stable")[:, :k]
        return np.take_along_axis(dist, order, axis=1), order

    def predict(self, queries: np.ndarray) -> np.ndarray:
        """Majority vote among the k nearest neighbours (ties: see module docstring)."""
        _, idx = self.kneighbors(queries)
        n_classes = int(self.labels_.max()) + 1
        predictions = np.empty(len(idx), dtype=int)
        for row, neighbour_idx in enumerate(idx):
            neighbour_labels = self.labels_[neighbour_idx]       # nearest first
            votes = np.bincount(neighbour_labels, minlength=n_classes)
            tied = np.flatnonzero(votes == votes.max())
            if len(tied) == 1:
                predictions[row] = tied[0]
            else:
                # first neighbour (= nearest) whose class is among the tied classes
                predictions[row] = next(lab for lab in neighbour_labels if lab in tied)
        return predictions

    def score(self, queries: np.ndarray, labels: np.ndarray) -> float:
        return float((self.predict(queries) == np.asarray(labels)).mean())


# --------------------------------------------------------------------------------------
# 2. Small synthetic datasets (the website demo offers similar presets)
# --------------------------------------------------------------------------------------
def make_blobs(n_per_class: int = 40, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Three Gaussian clusters in the unit square."""
    rng = np.random.default_rng(seed)
    centres = np.array([[0.25, 0.30], [0.75, 0.30], [0.50, 0.78]])
    points = np.vstack([rng.normal(c, 0.09, size=(n_per_class, 2)) for c in centres])
    points = np.clip(points, 0.02, 0.98)              # keep every point inside the plot
    labels = np.repeat(np.arange(len(centres)), n_per_class)
    return points, labels


def make_moons(n_per_class: int = 60, noise: float = 0.05, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two interleaving half circles: a classic non-linear problem."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, np.pi, n_per_class)
    upper = np.c_[np.cos(t), np.sin(t)]
    lower = np.c_[1 - np.cos(t), 0.5 - np.sin(t)]
    points = np.vstack([upper, lower]) + rng.normal(0, noise, size=(2 * n_per_class, 2))
    # rescale into the unit square so that all examples share the same plotting range
    points = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0))
    labels = np.repeat([0, 1], n_per_class)
    return points, labels


def decision_grid(model: KNNClassifier, resolution: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict the class of every pixel of a regular grid -> decision-boundary image."""
    xs = np.linspace(0, 1, resolution)
    gx, gy = np.meshgrid(xs, xs)
    grid = np.c_[gx.ravel(), gy.ravel()]
    return gx, gy, model.predict(grid).reshape(gx.shape)


# --------------------------------------------------------------------------------------
# 3. Cross-checks against well-tested libraries
# --------------------------------------------------------------------------------------
def predict_sklearn(points, labels, queries, k, metric):
    """scikit-learn reference (optional dependency). Returns None if it is not installed."""
    try:
        from sklearn.neighbors import KNeighborsClassifier
    except ImportError:
        return None
    clf = KNeighborsClassifier(n_neighbors=k, metric=metric, algorithm="brute")
    return clf.fit(points, labels).predict(queries)


def predict_opencv(points, labels, queries, k):
    """OpenCV reference: cv2.ml.KNearest (Euclidean distance only, OpenCV 4.x; None if unavailable).

    OpenCV expects float32 samples with one row per sample and int/float32 responses.
    ``findNearest`` returns (retval, results, neighbour_responses, dists).
    """
    try:
        import cv2
        knn = cv2.ml.KNearest_create()
    except (ImportError, AttributeError):          # OpenCV 5 moved the ml module out of the main package
        return None

    knn.train(points.astype(np.float32), cv2.ml.ROW_SAMPLE, labels.astype(np.float32).reshape(-1, 1))
    _, results, _, _ = knn.findNearest(queries.astype(np.float32), k)
    return results.ravel().astype(int)


def compare_implementations(points, labels, k, metric, n_queries=2000, seed=1):
    """Fraction of random query points on which each library agrees with our version."""
    rng = np.random.default_rng(seed)
    queries = rng.uniform(0, 1, size=(n_queries, 2))
    ours = KNNClassifier(k, metric).fit(points, labels).predict(queries)
    report = {}
    skl = predict_sklearn(points, labels, queries, k, metric)
    if skl is not None:
        report["scikit-learn"] = float((skl == ours).mean())
    if metric == "euclidean":                       # cv2.ml.KNearest is Euclidean only
        cv_pred = predict_opencv(points, labels, queries, k)
        if cv_pred is not None:
            report["cv2.ml.KNearest"] = float((cv_pred == ours).mean())
    return report


# --------------------------------------------------------------------------------------
# 4. Figure
# --------------------------------------------------------------------------------------
def save_figure(path: Path, metric: str, ks=(1, 5, 15)) -> None:
    import matplotlib
    matplotlib.use("Agg")                           # draw to a file, no window needed
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    colours = ["#2a6fbb", "#d2823f", "#1b8f7d"]
    cmap = ListedColormap(colours)
    datasets = {"three blobs": make_blobs(), "two moons": make_moons()}
    fig, axes = plt.subplots(len(datasets), len(ks), figsize=(3.4 * len(ks), 3.4 * len(datasets)), sharex=True, sharey=True)
    for r, (name, (pts, lab)) in enumerate(datasets.items()):
        for c, k in enumerate(ks):
            ax = axes[r, c]
            model = KNNClassifier(k, metric).fit(pts, lab)
            gx, gy, pred = decision_grid(model)
            ax.pcolormesh(gx, gy, pred, cmap=cmap, alpha=0.25, shading="auto", vmin=0, vmax=2)
            ax.scatter(pts[:, 0], pts[:, 1], c=lab, cmap=cmap, vmin=0, vmax=2, s=18, edgecolor="white", linewidth=0.5)
            ax.set_title(f"{name}, k = {k}", fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
    fig.suptitle(f"k-NN decision regions ({metric} distance): small k is jagged, large k is smooth", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--metric", choices=["euclidean", "manhattan"], default="euclidean")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "knn_comparison.png")
    args = parser.parse_args()

    points, labels = make_blobs()
    model = KNNClassifier(args.k, args.metric).fit(points, labels)
    print(f"training accuracy (k={args.k}, {args.metric}): {model.score(points, labels):.3f}")
    for name, agreement in compare_implementations(points, labels, args.k, args.metric).items():
        print(f"agreement with {name}: {agreement:.4f}")
    save_figure(args.output, args.metric)
    print(f"figure written to {args.output}")


if __name__ == "__main__":
    main()
