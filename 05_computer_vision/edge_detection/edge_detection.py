"""Sobel x / Sobel y / gradient magnitude and the Canny edge detector.

This script has two goals:
    * show how the OpenCV functions ``cv2.Sobel`` and ``cv2.Canny`` are used, and
    * re-implement the same steps with plain NumPy so that you can see what they do
      (the interactive demo on the website follows this NumPy version step by step).

The idea in one paragraph
    An edge is a place where the brightness changes quickly. The *derivative* of the
    image therefore peaks at edges. For a digital image the derivative is approximated
    by a small convolution kernel. The Sobel kernels do this and smooth a little in the
    perpendicular direction at the same time:

            Sobel x (Gx)            Sobel y (Gy)
         [-1  0  +1]             [-1 -2 -1]
         [-2  0  +2]             [ 0  0  0]
         [-1  0  +1]             [+1 +2 +1]

    * Gx responds to brightness changes from left to right, so it highlights VERTICAL edges.
    * Gy responds to brightness changes from top to bottom, so it highlights HORIZONTAL edges.
    * The gradient magnitude combines both: |G| = sqrt(Gx^2 + Gy^2). This shows edges of
      every orientation, but they are thick and noisy.

Canny turns that blurry magnitude image into thin, connected, black-and-white edges:
    1. Gaussian blur          suppresses noise (otherwise noise creates false edges)
    2. Sobel gradient         magnitude and direction at every pixel
    3. Non-maximum suppression keep only the pixel with the largest magnitude across the edge
    4. Double threshold       strong (> high), weak (between low and high), rejected (< low)
    5. Hysteresis             keep weak pixels only if they connect to a strong pixel

Note for OpenCV beginners:
    * ``cv2.Canny`` does NOT blur the image itself. Call ``cv2.GaussianBlur`` first.
    * OpenCV loads colour images as BGR (not RGB).
    * ``cv2.Canny(img, low, high)`` measures the gradient as |Gx| + |Gy| by default.
      Passing ``L2gradient=True`` uses the more accurate sqrt(Gx^2 + Gy^2).

Run it
    python edge_detection.py                       # synthetic sample, writes ../output/edge_detection.png
    python edge_detection.py --image my_photo.jpg --low 50 --high 150 --sigma 1.4 --ksize 5 --l2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = SOBEL_X.T.copy()


# --------------------------------------------------------------------------------------
# Sample image (generated, so there is no copyright question)
# --------------------------------------------------------------------------------------
def make_sample_image(size: int = 320) -> np.ndarray:
    """A synthetic BGR image with a circle, a rectangle, a triangle, a diagonal line and a
    soft gradient. Using shapes with known edges makes the results easy to judge."""
    img = np.full((size, size, 3), 235, dtype=np.uint8)
    # soft horizontal gradient in the lower part: edges should NOT appear here
    ramp = np.linspace(170, 215, size, dtype=np.uint8)
    img[int(size * 0.72):, :, :] = ramp[None, :, None]
    cv2.rectangle(img, (int(size * 0.08), int(size * 0.10)), (int(size * 0.42), int(size * 0.44)), (60, 60, 60), -1)
    cv2.circle(img, (int(size * 0.70), int(size * 0.28)), int(size * 0.17), (40, 90, 200), -1)
    triangle = np.array([[0.20, 0.90], [0.34, 0.55], [0.48, 0.90]]) * size
    cv2.fillPoly(img, [triangle.astype(np.int32)], (120, 170, 90))
    cv2.line(img, (int(size * 0.55), int(size * 0.92)), (int(size * 0.95), int(size * 0.52)), (30, 30, 30), 3, cv2.LINE_AA)
    return img


def to_gray(bgr: np.ndarray) -> np.ndarray:
    """OpenCV's own conversion: gray = 0.299 R + 0.587 G + 0.114 B."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


# --------------------------------------------------------------------------------------
# Building blocks implemented with NumPy
# --------------------------------------------------------------------------------------
def gaussian_kernel_1d(ksize: int, sigma: float) -> np.ndarray:
    """Normalised 1-D Gaussian. The 2-D Gaussian is separable: blur the rows, then the columns."""
    if ksize % 2 == 0:
        raise ValueError("kernel size must be odd")
    x = np.arange(ksize) - ksize // 2
    kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


def correlate2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Slide ``kernel`` over ``image`` (cross-correlation, like cv2.filter2D).

    The border is handled by mirroring (``BORDER_REFLECT_101``, OpenCV's default):
    the pixels just outside the image are copies of those just inside.
    """
    kh, kw = kernel.shape
    padded = np.pad(image.astype(np.float64), ((kh // 2, kh // 2), (kw // 2, kw // 2)), mode="reflect")
    out = np.zeros(image.shape, dtype=np.float64)
    h, w = image.shape
    for i in range(kh):                  # one shifted copy of the image per kernel entry
        for j in range(kw):
            out += kernel[i, j] * padded[i:i + h, j:j + w]
    return out


def gaussian_blur(gray: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
    """Separable Gaussian blur, rounded back to uint8 like cv2.GaussianBlur."""
    k = gaussian_kernel_1d(ksize, sigma)
    blurred = correlate2d(correlate2d(gray, k[None, :]), k[:, None])
    return np.clip(np.rint(blurred), 0, 255).astype(np.uint8)


def sobel(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (Gx, Gy) as float arrays (signed: dark->bright is positive in Gx)."""
    return correlate2d(gray, SOBEL_X), correlate2d(gray, SOBEL_Y)


def gradient_magnitude(gx: np.ndarray, gy: np.ndarray, l2: bool = False) -> np.ndarray:
    """|Gx| + |Gy| (OpenCV default, cheap) or sqrt(Gx^2 + Gy^2) (L2 gradient, exact)."""
    return np.sqrt(gx ** 2 + gy ** 2) if l2 else np.abs(gx) + np.abs(gy)


def non_maximum_suppression(gx: np.ndarray, gy: np.ndarray, mag: np.ndarray, low: float) -> np.ndarray:
    """Thin the edges: keep a pixel only if it is a local maximum ALONG the gradient direction.

    The gradient direction is rounded to one of four neighbour pairs:
        mostly horizontal gradient  -> compare with left / right pixel   (vertical edge)
        mostly vertical gradient    -> compare with upper / lower pixel  (horizontal edge)
        diagonal (two kinds)        -> compare with the two diagonal neighbours
    The comparison rule (strict '>' on one side, '>=' on the other) is the one OpenCV uses,
    so that a plateau of equal values keeps exactly one pixel.
    Returns a boolean mask of pixels that survive and are above ``low``.
    """
    ax, ay = np.abs(gx), np.abs(gy)
    tan22 = np.tan(np.deg2rad(22.5))
    tan67 = np.tan(np.deg2rad(67.5))
    horizontal = ay < ax * tan22                       # gradient within 22.5 deg of the x axis
    vertical = ay > ax * tan67                         # gradient within 22.5 deg of the y axis
    diagonal = ~horizontal & ~vertical
    same_sign = (gx * gy) >= 0                         # gradient points to lower-right / upper-left

    m = np.pad(mag, 1)                                 # zero border so every pixel has 8 neighbours
    h, w = mag.shape
    left, right = m[1:h + 1, 0:w], m[1:h + 1, 2:w + 2]
    up, down = m[0:h, 1:w + 1], m[2:h + 2, 1:w + 1]
    up_left, up_right = m[0:h, 0:w], m[0:h, 2:w + 2]
    down_left, down_right = m[2:h + 2, 0:w], m[2:h + 2, 2:w + 2]

    keep_h = (mag > left) & (mag >= right)
    keep_v = (mag > up) & (mag >= down)
    keep_d1 = (mag > up_left) & (mag > down_right)     # "\" diagonal
    keep_d2 = (mag > up_right) & (mag > down_left)     # "/" diagonal
    keep = np.where(horizontal, keep_h, np.where(vertical, keep_v, np.where(diagonal & same_sign, keep_d1, keep_d2)))
    return keep & (mag > low)


def hysteresis(candidates: np.ndarray, mag: np.ndarray, high: float) -> np.ndarray:
    """Keep a candidate (weak) pixel only if its 8-connected chain contains a strong pixel.

    ``cv2.connectedComponents`` labels each chain of touching candidate pixels; we then keep
    the chains that contain at least one pixel with magnitude > ``high``.
    """
    n_labels, labels = cv2.connectedComponents(candidates.astype(np.uint8), connectivity=8)
    strong = candidates & (mag > high)
    keep_label = np.zeros(n_labels, dtype=bool)
    keep_label[np.unique(labels[strong])] = True
    keep_label[0] = False                              # label 0 is the background
    return keep_label[labels]


def canny_from_scratch(gray_blurred: np.ndarray, low: float, high: float, l2: bool = False) -> np.ndarray:
    """Canny steps 2-5 on an already blurred uint8 image. Returns a uint8 image of 0 / 255.

    With ``l2=True`` thresholds are compared against sqrt(Gx^2 + Gy^2), otherwise |Gx| + |Gy|.
    """
    if low > high:
        low, high = high, low                          # OpenCV also swaps them
    gx, gy = sobel(gray_blurred)
    mag = gradient_magnitude(gx, gy, l2)
    thin = non_maximum_suppression(gx, gy, mag, low)
    edges = hysteresis(thin, mag, high)
    return edges.astype(np.uint8) * 255


# --------------------------------------------------------------------------------------
# The same pipeline with OpenCV
# --------------------------------------------------------------------------------------
def opencv_pipeline(bgr: np.ndarray, ksize: int, sigma: float, low: float, high: float, l2: bool):
    """Everything with OpenCV functions. Returns a dict of images for plotting."""
    gray = to_gray(bgr)
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
    # Use a 16-bit signed output (CV_16S) so that negative derivatives are not clipped to 0.
    gx = cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=3)    # dx = 1, dy = 0  -> vertical edges
    gy = cv2.Sobel(blurred, cv2.CV_16S, 0, 1, ksize=3)    # dx = 0, dy = 1  -> horizontal edges
    # Display helper: |G| scaled to 0..255 (convertScaleAbs takes the absolute value).
    abs_x, abs_y = cv2.convertScaleAbs(gx), cv2.convertScaleAbs(gy)
    magnitude = cv2.magnitude(gx.astype(np.float32), gy.astype(np.float32))
    magnitude_u8 = np.clip(magnitude / magnitude.max() * 255, 0, 255).astype(np.uint8)
    canny = cv2.Canny(blurred, low, high, L2gradient=l2)
    return {"gray": gray, "blurred": blurred, "sobel_x": abs_x, "sobel_y": abs_y,
            "magnitude": magnitude_u8, "canny": canny}


# --------------------------------------------------------------------------------------
# Agreement between two edge maps
# --------------------------------------------------------------------------------------
def edge_agreement(a: np.ndarray, b: np.ndarray, tolerance: int = 1) -> dict[str, float]:
    """Compare two binary edge maps.

    ``exact``  = fraction of identical pixels (almost always high because most pixels are background).
    ``f1``     = F1 score of the edge pixels, where an edge pixel counts as matched if the other map has
                 an edge within ``tolerance`` pixels (a 1-pixel shift is not a real disagreement).
    """
    a, b = a > 0, b > 0
    kernel = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
    near_b = cv2.dilate(b.astype(np.uint8), kernel) > 0
    near_a = cv2.dilate(a.astype(np.uint8), kernel) > 0
    precision = (a & near_b).sum() / max(a.sum(), 1)
    recall = (b & near_a).sum() / max(b.sum(), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"exact": float((a == b).mean()), "f1": float(f1)}


# --------------------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------------------
def save_figure(path: Path, bgr: np.ndarray, ksize: int, sigma: float, low: float, high: float, l2: bool) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cv = opencv_pipeline(bgr, ksize, sigma, low, high, l2)
    scratch = canny_from_scratch(cv["blurred"], low, high, l2)
    agreement = edge_agreement(scratch, cv["canny"])

    panels = [("Grayscale", cv["gray"]), ("Sobel x  (vertical edges)", cv["sobel_x"]),
              ("Sobel y  (horizontal edges)", cv["sobel_y"]), ("Gradient magnitude", cv["magnitude"]),
              (f"cv2.Canny  ({low:g}, {high:g}{', L2' if l2 else ''})", cv["canny"]),
              (f"NumPy Canny  (F1 vs OpenCV = {agreement['f1']:.3f})", scratch)]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.6))
    for ax, (title, image) in zip(axes.ravel(), panels):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Sobel and Canny edge detection (Gaussian blur: {ksize}x{ksize}, sigma = {sigma:g})", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return agreement


def main() -> None:
    parser = argparse.ArgumentParser(description="Sobel x / y / magnitude and Canny: OpenCV vs NumPy")
    parser.add_argument("--image", type=Path, help="input image (default: generated synthetic shapes)")
    parser.add_argument("--ksize", type=int, default=5, help="Gaussian kernel size (odd)")
    parser.add_argument("--sigma", type=float, default=1.4, help="Gaussian sigma")
    parser.add_argument("--low", type=float, default=50, help="Canny low threshold")
    parser.add_argument("--high", type=float, default=150, help="Canny high threshold")
    parser.add_argument("--l2", action="store_true", help="use the L2 gradient magnitude")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "edge_detection.png")
    args = parser.parse_args()

    if args.image is None:
        bgr = make_sample_image()
    else:
        bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"could not read image: {args.image}")
    agreement = save_figure(args.output, bgr, args.ksize, args.sigma, args.low, args.high, args.l2)
    print(f"NumPy Canny vs cv2.Canny: exact pixel match {agreement['exact']:.4f}, F1 (1 px tolerance) {agreement['f1']:.4f}")
    print(f"figure written to {args.output}")


if __name__ == "__main__":
    main()
