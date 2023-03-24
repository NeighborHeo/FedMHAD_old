import numpy as np
import matplotlib.pyplot as plt

def sparsemax(z):
    sorted_z = np.sort(z)[::-1]
    cumulative_sum = np.cumsum(sorted_z)
    k = np.arange(1, len(z) + 1)
    comp = (1 + k * sorted_z) > cumulative_sum
    k_star = k[np.argmax(comp)]
    threshold = (cumulative_sum[k_star - 1] - 1) / k_star
    return np.maximum(z - threshold, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__ == '__main__':
    x = np.linspace(-10, 10, 1000)

    # Sigmoid plot
    sigmoid_y = sigmoid(x)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, sigmoid_y)
    plt.title("Sigmoid Function")
    plt.xlabel("x")
    plt.ylabel("sigmoid(x)")

    # Sparsemax plot
    sparsemax_y = sparsemax(x)

    plt.subplot(1, 2, 2)
    plt.plot(x, sparsemax_y)
    plt.title("Sparsemax Function")
    plt.xlabel("Index")
    plt.ylabel("sparsemax(z)")
    plt.savefig("sparsemax.png", dpi=300)

    plt.tight_layout()
    plt.show()

 