"""A small neural network from scratch (NumPy only) that learns a 2-D decision boundary.

This is the Python counterpart of the "Neural Network Playground" on
https://dingshuaiyang1999.github.io/demos/neural-network.html. It uses no deep-learning library, so
every step of training is visible.

The problem
    Each data point has two coordinates (x, y) and one of two labels (0 or 1). A single straight line
    cannot separate the labels of the "circle", "xor" and "spiral" data sets. A network with hidden
    layers can, because every hidden layer bends the space a little further.

The model (a fully connected feed-forward network)
    For layer l with weight matrix W_l and bias vector b_l:

        z_l = a_{l-1} @ W_l + b_l          (weighted sum, "pre-activation")
        a_l = f(z_l)                       (activation function, f = tanh / relu / sigmoid)

    a_0 is the input (x, y). The last layer has one neuron with a sigmoid, so its output
    p = P(label = 1 | x, y) lies between 0 and 1.

The loss (binary cross-entropy, plus an optional L2 penalty on the weights)
        L = -mean( y*log(p) + (1-y)*log(1-p) ) + lambda/2 * sum(W^2)

Back-propagation (the chain rule, applied layer by layer from the output to the input)
    With a sigmoid output and cross-entropy loss, the gradient at the output layer simplifies to

        delta_L = (p - y) / N

    and for every earlier layer

        dL/dW_l = a_{l-1}^T @ delta_l (+ lambda * W_l)
        dL/db_l = sum over the batch of delta_l
        delta_{l-1} = (delta_l @ W_l^T) * f'(z_{l-1})

    The function ``gradient_check`` compares these formulas with numerical finite differences.

The optimiser (Adam)
    Plain gradient descent moves every weight by -learning_rate * gradient. Adam additionally keeps a
    running mean m of the gradient and a running mean v of its square, and steps by
    learning_rate * m_hat / (sqrt(v_hat) + eps). This gives a sensible step size for every weight.

Run it
    python nn_decision_boundary.py                              # spiral, 2 hidden layers of 8 tanh neurons
    python nn_decision_boundary.py --data xor --layers 4 --activation relu
    python nn_decision_boundary.py --data circle --layers 3 --epochs 300

The figure (decision boundary, training points and loss curve) is written to ../output/nn_decision_boundary.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"


# --------------------------------------------------------------------------------------
# 1. Data sets (all coordinates lie roughly in [-1, 1] x [-1, 1])
# --------------------------------------------------------------------------------------
def make_dataset(name: str, n: int = 200, noise: float = 0.05, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X, y)``: X has shape (n, 2), y has shape (n,) with labels 0 or 1."""
    rng = np.random.default_rng(seed)
    half = n // 2
    if name == "circle":
        # class 0: a disc around the origin, class 1: a ring around it
        r0 = 0.45 * np.sqrt(rng.random(half))
        r1 = 0.7 + 0.25 * rng.random(n - half)
        a0 = 2 * np.pi * rng.random(half)
        a1 = 2 * np.pi * rng.random(n - half)
        X = np.vstack([np.c_[r0 * np.cos(a0), r0 * np.sin(a0)], np.c_[r1 * np.cos(a1), r1 * np.sin(a1)]])
    elif name == "xor":
        # label = 1 if x and y have the same sign (opposite quadrants share a label)
        X = rng.uniform(-1, 1, size=(n, 2))
        X += 0.1 * np.sign(X)                      # keep a gap around the axes
        y = (X[:, 0] * X[:, 1] > 0).astype(float)
        return np.clip(X, -1, 1) + rng.normal(0, noise, X.shape), y
    elif name == "gaussian":
        X = np.vstack([rng.normal([-0.45, -0.45], 0.25, (half, 2)), rng.normal([0.45, 0.45], 0.25, (n - half, 2))])
    elif name == "spiral":
        # two interleaved spiral arms; the second arm is the first rotated by 180 degrees
        t = np.linspace(0.1, 1.0, half)
        theta = 1.25 * 2 * np.pi * t
        arm = np.c_[t * np.cos(theta), t * np.sin(theta)] * 0.95
        X = np.vstack([arm, -arm])
    else:
        raise ValueError(f"unknown data set: {name!r} (circle, xor, gaussian, spiral)")
    y = np.r_[np.zeros(half), np.ones(n - half)]
    return X + rng.normal(0, noise, X.shape), y


# --------------------------------------------------------------------------------------
# 2. Activation functions: (function, derivative expressed with the *output* a = f(z))
# --------------------------------------------------------------------------------------
ACTIVATIONS = {
    "tanh": (np.tanh, lambda a: 1.0 - a ** 2),
    "relu": (lambda z: np.maximum(z, 0.0), lambda a: (a > 0).astype(float)),
    "sigmoid": (lambda z: 1.0 / (1.0 + np.exp(-z)), lambda a: a * (1.0 - a)),
}


# --------------------------------------------------------------------------------------
# 3. The network
# --------------------------------------------------------------------------------------
class NeuralNetwork:
    """Fully connected network: input(2) -> hidden layers -> 1 sigmoid output."""

    def __init__(self, hidden: list[int], activation: str = "tanh", l2: float = 0.0, seed: int = 0) -> None:
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}")
        self.act, self.act_grad = ACTIVATIONS[activation]
        self.l2 = l2
        sizes = [2, *hidden, 1]
        rng = np.random.default_rng(seed)
        # Initial weights must be random and have a sensible scale, otherwise all neurons in a
        # layer would learn the same thing, or the signals would shrink or explode layer by layer.
        #   He initialisation (variance 2/fan_in) suits ReLU; Xavier (1/fan_in) suits tanh/sigmoid.
        gain = 2.0 if activation == "relu" else 1.0
        self.W = [rng.normal(0.0, np.sqrt(gain / m), (m, n)) for m, n in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(n) for n in sizes[1:]]
        # Adam state (first and second moment estimates for every parameter array)
        self._m = [np.zeros_like(p) for p in self.W + self.b]
        self._v = [np.zeros_like(p) for p in self.W + self.b]
        self._t = 0

    # ---- forward pass ----------------------------------------------------------------
    def forward(self, X: np.ndarray) -> list[np.ndarray]:
        """Return the list of activations [a_0 = X, a_1, ..., p]; they are re-used by back-propagation."""
        activations = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = activations[-1] @ W + b
            last = i == len(self.W) - 1
            activations.append(ACTIVATIONS["sigmoid"][0](z) if last else self.act(z))
        return activations

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)[-1][:, 0]

    # ---- loss ------------------------------------------------------------------------
    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        p = np.clip(self.predict_proba(X), 1e-12, 1 - 1e-12)     # avoid log(0)
        data_term = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        return float(data_term + 0.5 * self.l2 * sum((W ** 2).sum() for W in self.W))

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean((self.predict_proba(X) > 0.5) == (y > 0.5)))

    # ---- back-propagation ------------------------------------------------------------
    def gradients(self, X: np.ndarray, y: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Gradients of the loss with respect to every weight matrix and bias vector."""
        a = self.forward(X)
        n = len(X)
        delta = (a[-1] - y[:, None]) / n                      # output layer: (p - y) / N
        grads_W: list[np.ndarray] = [np.empty(0)] * len(self.W)
        grads_b: list[np.ndarray] = [np.empty(0)] * len(self.W)
        for l in reversed(range(len(self.W))):
            grads_W[l] = a[l].T @ delta + self.l2 * self.W[l]
            grads_b[l] = delta.sum(axis=0)
            if l > 0:                                         # push the error one layer back
                delta = (delta @ self.W[l].T) * self.act_grad(a[l])
        return grads_W, grads_b

    # ---- optimiser ---------------------------------------------------------------------
    def adam_step(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01,
                  beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        gW, gb = self.gradients(X, y)
        params = self.W + self.b
        self._t += 1
        for i, g in enumerate(gW + gb):
            self._m[i] = beta1 * self._m[i] + (1 - beta1) * g            # running mean of the gradient
            self._v[i] = beta2 * self._v[i] + (1 - beta2) * g ** 2       # running mean of its square
            m_hat = self._m[i] / (1 - beta1 ** self._t)                   # bias correction (start-up)
            v_hat = self._v[i] / (1 - beta2 ** self._t)
            params[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)              # in-place update

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 200, lr: float = 0.01,
            batch_size: int = 10, seed: int = 0, verbose: bool = False) -> list[float]:
        """Train with mini-batches. One epoch = one pass over all training points."""
        rng = np.random.default_rng(seed)
        history = []
        for epoch in range(epochs):
            order = rng.permutation(len(X))                   # shuffle so that batches differ every epoch
            for start in range(0, len(X), batch_size):
                idx = order[start:start + batch_size]
                self.adam_step(X[idx], y[idx], lr=lr)
            history.append(self.loss(X, y))
            if verbose and (epoch % 25 == 0 or epoch == epochs - 1):
                print(f"epoch {epoch:4d}  loss {history[-1]:.4f}  accuracy {self.accuracy(X, y):.3f}")
        return history


def gradient_check(net: NeuralNetwork, X: np.ndarray, y: np.ndarray, h: float = 1e-5) -> float:
    """Largest relative difference between back-propagation and finite differences.

    For each parameter w:  dL/dw ~= (L(w + h) - L(w - h)) / (2h).  A correct implementation gives
    a value around 1e-7 or smaller; a bug in back-propagation typically gives 1e-2 or more.
    (ReLU is not differentiable at z = 0; if a pre-activation is exactly 0, which happens when the biases
    are all zero and a whole layer is inactive, the check can report a false mismatch.)
    """
    gW, gb = net.gradients(X, y)
    worst = 0.0
    for params, grads in ((net.W, gW), (net.b, gb)):
        for p, g in zip(params, grads):
            for index in np.ndindex(p.shape):
                old = p[index]
                p[index] = old + h
                up = net.loss(X, y)
                p[index] = old - h
                down = net.loss(X, y)
                p[index] = old
                numeric = (up - down) / (2 * h)
                worst = max(worst, abs(numeric - g[index]) / max(1e-6, abs(numeric) + abs(g[index])))
    return worst


# --------------------------------------------------------------------------------------
# 4. Visualisation
# --------------------------------------------------------------------------------------
def decision_grid(net: NeuralNetwork, resolution: int = 200, extent: float = 1.2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the network on a regular grid; the result is the "heat map" of P(label = 1)."""
    xs = np.linspace(-extent, extent, resolution)
    gx, gy = np.meshgrid(xs, xs)
    p = net.predict_proba(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    return gx, gy, p


def plot(net: NeuralNetwork, X: np.ndarray, y: np.ndarray, history: list[float], path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gx, gy, p = decision_grid(net)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    ax1.contourf(gx, gy, p, levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=0.85)
    ax1.contour(gx, gy, p, levels=[0.5], colors="k", linewidths=1.2)           # the decision boundary p = 0.5
    ax1.scatter(X[y == 0, 0], X[y == 0, 1], c="#2a6fbb", edgecolors="white", label="class 0")
    ax1.scatter(X[y == 1, 0], X[y == 1, 1], c="#d97f35", edgecolors="white", marker="s", label="class 1")
    ax1.set_aspect("equal"); ax1.legend(loc="upper right"); ax1.set_title(title)
    ax2.plot(history); ax2.set_xlabel("epoch"); ax2.set_ylabel("training loss (cross-entropy)")
    ax2.set_title("Loss curve"); ax2.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default="spiral", choices=["circle", "xor", "gaussian", "spiral"])
    parser.add_argument("--layers", type=int, default=2, help="number of hidden layers")
    parser.add_argument("--neurons", type=int, default=8, help="neurons per hidden layer")
    parser.add_argument("--activation", default="tanh", choices=list(ACTIVATIONS))
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "nn_decision_boundary.png")
    args = parser.parse_args()

    X, y = make_dataset(args.data, seed=args.seed)
    net = NeuralNetwork([args.neurons] * args.layers, args.activation, l2=args.l2, seed=args.seed)
    history = net.fit(X, y, epochs=args.epochs, lr=args.lr, seed=args.seed, verbose=True)
    print(f"final training accuracy: {net.accuracy(X, y):.3f}")
    plot(net, X, y, history, args.output,
         f"{args.data}: {args.layers}x{args.neurons} {args.activation}, accuracy {net.accuracy(X, y):.2f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
