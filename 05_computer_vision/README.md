# 05 Computer Vision

Small, readable examples that accompany the interactive demos on
[dingshuaiyang1999.github.io](https://dingshuaiyang1999.github.io/). Each script explains the idea
in its docstring, implements it from scratch with NumPy, and compares the result with a library.

| Folder | What it shows | Library cross-check |
|---|---|---|
| `knn/knn_from_scratch.py` | k-nearest-neighbour classifier, Euclidean and Manhattan distance, decision regions | scikit-learn, `cv2.ml.KNearest` |
| `neural_network/nn_decision_boundary.py` | Fully connected network with back-propagation and Adam, trained on circle, XOR, Gaussian and spiral data; decision-boundary heat map and loss curve | gradient check against finite differences |
| `edge_detection/edge_detection.py` | Sobel x, Sobel y, gradient magnitude, Canny (Gaussian blur, non-maximum suppression, double threshold, hysteresis) | `cv2.Sobel`, `cv2.Canny` |
| `harris_corner_detection/harris.py` | Harris response `AB − C² − α(A+B)²`, relative threshold, non-maximum suppression | website `demos/harris.js` |

## Run

```bash
pip install -r 05_computer_vision/requirements.txt

python 05_computer_vision/knn/knn_from_scratch.py --k 5 --metric euclidean
python 05_computer_vision/neural_network/nn_decision_boundary.py --data spiral --layers 2 --neurons 8 --activation tanh
python 05_computer_vision/edge_detection/edge_detection.py --low 50 --high 150 --sigma 1.4 --ksize 5
python 05_computer_vision/edge_detection/edge_detection.py --image my_photo.jpg --l2
python 05_computer_vision/harris_corner_detection/harris.py
python 05_computer_vision/harris_corner_detection/harris.py --image photo.jpg --output corners.png
```

Figures are written to `05_computer_vision/output/`:

* `knn_comparison.png`: decision regions for k = 1, 5, 15 on two synthetic data sets.
* `nn_decision_boundary.png`: decision boundary of the trained network with the training points, and the loss curve.
* `edge_detection.png`: grayscale, Sobel x, Sobel y, gradient magnitude, `cv2.Canny` and the NumPy
  re-implementation side by side.

## Notes

* The neural network uses only NumPy. `gradient_check` compares the back-propagation formulas with numerical
  finite differences; the unit tests run it for tanh, sigmoid and ReLU.
* In k-NN, differences between this implementation and the libraries occur only when two classes receive the
  same number of votes; the tie-breaking rule is documented in `knn_from_scratch.py`.
* `cv2.ml` is part of OpenCV 4.x. The requirements file therefore pins `opencv-python-headless<5`.
* The NumPy Canny implementation reproduces `cv2.Canny` pixel for pixel on the generated sample image
  (see `tests/test_computer_vision.py`). `cv2.Canny` does not blur the image itself, so both are applied
  to an image that was blurred with the same Gaussian kernel.
* Run the tests with `pytest tests/test_computer_vision.py`. They skip themselves if OpenCV or
  scikit-learn is missing.
