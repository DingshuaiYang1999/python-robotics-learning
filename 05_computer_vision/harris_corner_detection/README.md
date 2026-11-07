# Harris corner detector

NumPy counterpart of the live demo at
<https://dingshuaiyang1999.github.io/demos/harris.html>.
The arithmetic follows `demos/harris.js` (website `main`).

Gray is `0.299 R + 0.587 G + 0.114 B` with no rounding. An optional Gaussian
pre-blur (default 3×3, σ = 1) uses reflect-101 and a kernel that sums to 1;
there is no integer rounding. A width of 1 or less skips that blur. If σ ≤ 0
the website kernel uses σ² = 1. Sobel is 3×3 correlation with replicate
borders: Ix is positive toward the right, Iy positive downward. The products
Ix², Iy² and Ix Iy are smoothed by a second Gaussian (default 5×5, σ = 1.5),
same border and normalization.

`R = AB − C² − α(A+B)²` with α = 0.04. A corner is a pixel whose R is strictly
above `rel` times the largest **positive** R (default rel = 0.08) and is the
maximum in an odd window (default 5). An exact tie keeps the smallest
row-major index. Pixels outside the image are not neighbours. R is not a
probability. The page's max-abs picture scaling is display-only and is not
applied to the threshold.

```bash
python 05_computer_vision/harris_corner_detection/harris.py
python 05_computer_vision/harris_corner_detection/harris.py --image photo.jpg --output corners.png
pytest tests/test_harris.py
```

With no `--image`, the input is a synthetic square. A local photo is read only
to write the overlay you name; it is not stored in this repository. PPM works
with NumPy alone. PNG and JPEG need OpenCV or Pillow (`05_computer_vision/requirements.txt`
includes OpenCV). The default output is `05_computer_vision/output/harris_corners.png`.
