import random
import unittest
import math
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
import numpy as np

random.seed(0)

import warnings
warnings.filterwarnings('ignore')

def rosenbrock(a, b, x, y):
    out = (a - x)**2 + b*(y - x**2)**2
    return out

def rosenbrock_grad(a, b, x, y):
    grad_x = -2*(a - x) - 4*b*(y - x**2)*x
    grad_y = 2*b*(y - x**2)
    return grad_x, grad_y

def rosenbrock_sgd(initial_x, initial_y, a, b, n_epochs, lr, tolerance):
    out_prev = -math.inf
    final_x = initial_x
    final_y = initial_y
    stop_epoch = 0
    for epoch in range(n_epochs):
        out = rosenbrock(a, b, final_x, final_y)
        if abs(out - out_prev) <= tolerance:
            return final_x, final_y, stop_epoch
        out_prev = out
        grad_x, grad_y = rosenbrock_grad(a, b, final_x, final_y)
        final_x = final_x - lr * grad_x
        final_y = final_y - lr * grad_y
        stop_epoch = epoch
        if epoch%500 == 0:
            print(final_x, final_y)
            
    return final_x, final_y, n_epochs

class TestRosenBrock(unittest.TestCase):
    def test_sgd(self):
        final_x, final_y, stop_epoch = rosenbrock_sgd(0, 0, 1, 100, 1, 0.001, 1e-06)
        print(final_x, final_y, stop_epoch)
        self.assertAlmostEqual(final_x, 0.002, places = 4)
        self.assertAlmostEqual(final_y, 0, places = 4)
        self.assertAlmostEqual(stop_epoch, 1, places = 4)

        final_x, final_y, stop_epoch = rosenbrock_sgd(0, 0, 1, 100, 5, 0.001, 1e-06)
        print(final_x, final_y, stop_epoch)
        self.assertAlmostEqual(final_x, 0.009959805751775453, places = 4)
        self.assertAlmostEqual(final_y, 2.091e-05, places = 4)
        self.assertAlmostEqual(stop_epoch, 5, places = 4)
        
        final_x, final_y, stop_epoch = rosenbrock_sgd(0, 0, 1, 100, 100000, 0.001, 1e-06)
        print(final_x, final_y, stop_epoch)
        self.assertAlmostEqual(final_x, 0.965628504058544, places = 4)
        self.assertAlmostEqual(final_y, 0.932297997695398, places = 4)
        self.assertAlmostEqual(stop_epoch,  5570, places = 4)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    print("----------------------------------------------------------------------")
    final_x, final_y, stop_epoch = rosenbrock_sgd(0, 0, 1, 100, 100000, 0.001, 1e-06)
    x0 = np.linspace(-4, 4, 800)
    x1 = np.linspace(-3, 3, 600)
    X, Y = np.meshgrid(x0, x1)
    Z = rosenbrock(1, 100, X, Y)

    levels = np.logspace(-1, 3, 10)
    plt.contourf(X, Y, Z, alpha=0.2, levels=levels)
    plt.contour(X, Y, Z, colors="gray",
                levels=[0.4, 3, 15, 50, 150, 500, 1500, 5000])
    plt.plot(final_x, final_y, 'ro', markersize=10)
    plt.xlim(-4, 4)
    plt.ylim(-3, 3)
    plt.xticks(np.linspace(-4, 4, 9))
    plt.yticks(np.linspace(-3, 3, 7))
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.show()