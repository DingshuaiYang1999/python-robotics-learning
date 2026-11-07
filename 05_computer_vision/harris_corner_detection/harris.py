"""Harris corner detector, matching the website demo.

The pipeline is the one in ``demos/harris.js`` on
https://dingshuaiyang1999.github.io/demos/harris.html :

* gray = 0.299 R + 0.587 G + 0.114 B, with no rounding
* optional Gaussian pre-blur, default 3×3, sigma 1, reflect-101, weights sum to 1
* 3×3 Sobel correlation, replicate borders; +x is brighter to the right, +y is down
* products Ix², Iy², Ix Iy, then Gaussian integration, default 5×5, sigma 1.5
* R = A B − C² − α (A + B)², α = 0.04
* keep R strictly above (rel × largest positive R), rel = 0.08, if it is the
  maximum in a 5×5 neighbourhood; an exact tie keeps the smallest row-major index

R is not a probability. Dividing the response by its maximum is display-only on
the website and is not done here.

Run
    python 05_computer_vision/harris_corner_detection/harris.py
    python 05_computer_vision/harris_corner_detection/harris.py --image photo.jpg
"""
from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

import numpy as np

DEFAULTS = {
    "ksize_d": 3,
    "sigma_d": 1.0,
    "ksize_i": 5,
    "sigma_i": 1.5,
    "alpha": 0.04,
    "rel_thresh": 0.08,
    "nms": 5,
}

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """0.299 R + 0.587 G + 0.114 B as float64. No rounding. Alpha is ignored."""
    image = np.asarray(rgb)
    if image.ndim == 2:
        return image.astype(np.float64, copy=False)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[-1] != 3:
        raise ValueError("expected a gray image or an RGB(A) image")
    channels = image.astype(np.float64, copy=False)
    return 0.299 * channels[..., 0] + 0.587 * channels[..., 1] + 0.114 * channels[..., 2]


def gaussian_kernel(ksize: int, sigma: float) -> np.ndarray:
    """Sampled Gaussian. Non-positive sigma uses sigma^2 = 1, as the website does."""
    center = (ksize - 1) / 2.0
    t = np.arange(ksize, dtype=np.float64) - center
    s2 = sigma * sigma if sigma > 0 else 1.0
    kernel = np.exp(-(t * t) / (2.0 * s2))
    return kernel / kernel.sum()


def reflect101(index: np.ndarray, n: int) -> np.ndarray:
    """OpenCV BORDER_REFLECT_101. n <= 1 maps every index to 0."""
    i = np.array(index, dtype=np.int64, copy=True)
    if n <= 1:
        i[...] = 0
        return i
    for _ in range(64):
        negative = i < 0
        if negative.any():
            i[negative] = -i[negative]
        too_big = i >= n
        if too_big.any():
            i[too_big] = 2 * n - 2 - i[too_big]
        if not np.any((i < 0) | (i >= n)):
            return i
    raise RuntimeError("reflect-101 did not converge")


def gaussian_blur(src: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
    """Separable normalized Gaussian, reflect-101, no rounding. ksize <= 1 copies."""
    image = np.asarray(src, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError("gaussian_blur expects a 2-D image")
    if not ksize > 1:
        return image.copy()
    ksize = int(ksize)
    if ksize < 1 or ksize % 2 == 0:
        raise ValueError("Gaussian kernel size must be a positive odd integer")
    kernel = gaussian_kernel(ksize, sigma)
    radius = ksize // 2
    height, width = image.shape
    offsets = np.arange(ksize, dtype=np.int64) - radius
    x_index = reflect101(np.arange(width, dtype=np.int64)[:, None] + offsets[None, :], width)
    tmp = np.einsum("hwk,k->hw", image[:, x_index], kernel)
    y_index = reflect101(np.arange(height, dtype=np.int64)[:, None] + offsets[None, :], height)
    return np.einsum("hkw,k->hw", tmp[y_index, :], kernel)


def sobel(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """3×3 Sobel correlation with replicate borders.

    Ix is positive when the image gets brighter toward the right.
    Iy is positive when it gets brighter downward.
    """
    padded = np.pad(np.asarray(gray, dtype=np.float64), 1, mode="edge")
    upper, mid, lower = padded[:-2], padded[1:-1], padded[2:]
    ix = (upper[:, 2:] + 2.0 * mid[:, 2:] + lower[:, 2:]) - (upper[:, :-2] + 2.0 * mid[:, :-2] + lower[:, :-2])
    iy = (lower[:, :-2] + 2.0 * lower[:, 1:-1] + lower[:, 2:]) - (upper[:, :-2] + 2.0 * upper[:, 1:-1] + upper[:, 2:])
    return ix, iy


def suppress(response: np.ndarray, nms: int, threshold: float) -> np.ndarray:
    """Mark 255 where R is a strict local maximum above the threshold.

    Neighbours outside the image are ignored. An exact tie keeps the smallest
    row-major index, which is the website rule.
    """
    score = np.asarray(response, dtype=np.float64)
    height, width = score.shape
    out = np.zeros((height, width), dtype=np.uint8)
    radius = max(0, int((float(nms) - 1) / 2.0))
    candidates = np.argwhere((score > threshold) & (score > 0.0))
    for y, x in candidates:
        y = int(y)
        x = int(x)
        value = score[y, x]
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        window = score[y0:y1, x0:x1]
        if np.any(window > value):
            continue
        index = y * width + x
        tied = np.argwhere(window == value)
        if any((y0 + int(ty)) * width + (x0 + int(tx)) < index for ty, tx in tied):
            continue
        out[y, x] = 255
    return out


def harris(
    gray: np.ndarray,
    ksize_d: int = DEFAULTS["ksize_d"],
    sigma_d: float = DEFAULTS["sigma_d"],
    ksize_i: int = DEFAULTS["ksize_i"],
    sigma_i: float = DEFAULTS["sigma_i"],
    alpha: float = DEFAULTS["alpha"],
    rel_thresh: float = DEFAULTS["rel_thresh"],
    nms: int = DEFAULTS["nms"],
) -> dict[str, np.ndarray | float]:
    """Harris response and corner mask. ``gray`` is a float luminance image."""
    image = np.asarray(gray, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError("harris expects a 2-D gray image")
    source = gaussian_blur(image, ksize_d, sigma_d) if ksize_d > 1 else image
    ix, iy = sobel(source)
    ixx, iyy, ixy = ix * ix, iy * iy, ix * iy
    structure_xx = gaussian_blur(ixx, ksize_i, sigma_i)
    structure_yy = gaussian_blur(iyy, ksize_i, sigma_i)
    structure_xy = gaussian_blur(ixy, ksize_i, sigma_i)
    trace = structure_xx + structure_yy
    response = structure_xx * structure_yy - structure_xy * structure_xy - alpha * trace * trace
    max_positive = 0.0
    if response.size:
        peak = float(np.max(response))
        if peak > max_positive:
            max_positive = peak
    fraction = rel_thresh if rel_thresh > 0 else 0.0
    threshold = fraction * max_positive
    return {
        "ix": ix,
        "iy": iy,
        "ixx": ixx,
        "iyy": iyy,
        "ixy": ixy,
        "A": structure_xx,
        "B": structure_yy,
        "C": structure_xy,
        "R": response,
        "max_pos": max_positive,
        "threshold": threshold,
        "corners": suppress(response, nms, threshold),
    }


def synthetic_square(size: int = 80, value: int = 220) -> np.ndarray:
    """Bright square on a black background. Used when no local image is given."""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    lo, hi = size // 4, 3 * size // 4
    image[lo:hi, lo:hi] = value
    return image


def corner_overlay(rgb: np.ndarray, corners: np.ndarray, radius: float = 3.5) -> np.ndarray:
    """RGB image with red rings at corner pixels. The rings are only a drawing."""
    image = np.asarray(rgb)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.dtype != np.uint8:
        image = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    out = np.array(image[..., :3], dtype=np.uint8, copy=True)
    color = np.array([226, 61, 43], dtype=np.uint8)
    span = int(np.ceil(radius + 1.0))
    height, width = corners.shape
    ys, xs = np.nonzero(corners)
    for y, x in zip(ys.tolist(), xs.tolist()):
        y0, y1 = max(0, y - span), min(height, y + span + 1)
        x0, x1 = max(0, x - span), min(width, x + span + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        distance2 = (yy - y) ** 2 + (xx - x) ** 2
        ring = (distance2 >= (radius - 1.2) ** 2) & (distance2 <= (radius + 0.8) ** 2)
        out[y0:y1, x0:x1][ring] = color
    return out


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    """Write a PNG (or a binary PPM if the path ends in ``.ppm``). NumPy only."""
    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("save_rgb expects an RGB uint8 image")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width, _ = image.shape
    if path.suffix.lower() == ".ppm":
        header = f"P6\n{width} {height}\n255\n".encode("ascii")
        path.write_bytes(header + np.ascontiguousarray(image).tobytes())
        return
    raw = b"".join(b"\x00" + np.ascontiguousarray(image[y]).tobytes() for y in range(height))
    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += _png_chunk(b"IDAT", zlib.compress(raw, 9))
    png += _png_chunk(b"IEND", b"")
    path.write_bytes(png)


def load_image(path: Path) -> np.ndarray:
    """Load a local RGB uint8 image. Does not download anything.

    PPM is read with the standard library. PNG and JPEG use OpenCV or Pillow
    when one of those is installed (``opencv-python`` is in the requirements).
    """
    suffix = path.suffix.lower()
    if suffix == ".ppm":
        return _read_ppm(path)
    try:
        import cv2
    except ImportError:
        cv2 = None
    if cv2 is not None:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"could not read image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    try:
        from PIL import Image
    except ImportError:
        raise SystemExit(
            f"could not read {path}. Install opencv-python or Pillow, or pass a PPM image."
        ) from None
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _read_ppm(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if not raw.startswith(b"P6"):
        raise SystemExit("only binary PPM (P6) is supported")
    tokens: list[bytes] = []
    i = 2
    while len(tokens) < 3:
        if i >= len(raw):
            raise SystemExit(f"truncated PPM header: {path}")
        if raw[i] in b" \t\r\n":
            i += 1
            continue
        if raw[i] == ord("#"):
            while i < len(raw) and raw[i] != ord("\n"):
                i += 1
            continue
        start = i
        while i < len(raw) and raw[i] not in b" \t\r\n":
            i += 1
        tokens.append(raw[start:i])
    width, height, max_value = (int(part) for part in tokens)
    if max_value != 255:
        raise SystemExit("PPM max value must be 255")
    if i < len(raw) and raw[i] in b" \t\r\n":
        i += 1
    count = width * height * 3
    pixels = np.frombuffer(raw, dtype=np.uint8, count=count, offset=i)
    if pixels.size != count:
        raise SystemExit(f"truncated PPM pixels: {path}")
    return pixels.reshape(height, width, 3).copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Harris corners, matching the website demo")
    parser.add_argument("--image", type=Path, help="local input image (nothing is downloaded)")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "harris_corners.png")
    parser.add_argument("--ksize-d", type=int, default=DEFAULTS["ksize_d"])
    parser.add_argument("--sigma-d", type=float, default=DEFAULTS["sigma_d"])
    parser.add_argument("--ksize-i", type=int, default=DEFAULTS["ksize_i"])
    parser.add_argument("--sigma-i", type=float, default=DEFAULTS["sigma_i"])
    parser.add_argument("--alpha", type=float, default=DEFAULTS["alpha"])
    parser.add_argument("--rel", type=float, default=DEFAULTS["rel_thresh"], help="fraction of the largest positive R")
    parser.add_argument("--nms", type=int, default=DEFAULTS["nms"], help="odd neighbourhood width")
    args = parser.parse_args()

    rgb = synthetic_square() if args.image is None else load_image(args.image)
    gray = to_gray(rgb)
    result = harris(
        gray,
        ksize_d=args.ksize_d,
        sigma_d=args.sigma_d,
        ksize_i=args.ksize_i,
        sigma_i=args.sigma_i,
        alpha=args.alpha,
        rel_thresh=args.rel,
        nms=args.nms,
    )
    corners = result["corners"]
    save_rgb(args.output, corner_overlay(rgb, corners))
    count = int(np.count_nonzero(corners))
    print(f"corners: {count}")
    print(f"largest positive R: {result['max_pos']:.6e}")
    print(f"threshold: {result['threshold']:.6e}  (rel {args.rel:g} times that maximum)")
    print("R is the Harris response, not a probability. Display scaling is not applied.")
    print(f"overlay written to {args.output}")


if __name__ == "__main__":
    main()
