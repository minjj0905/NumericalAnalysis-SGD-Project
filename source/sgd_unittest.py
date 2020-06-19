import unittest
import random
from sgd_numpy import SGD

class TestSgd(unittest.TestCase):
    def test_predict(self):
        testsgd = SGD()
        y = testsgd.predict(10, 1, 1)
        self.assertEqual(y, 11)

    def test_error(self):
        testsgd = SGD()
        error = testsgd.error(10, 1, 1, 11)
        self.assertEqual(error, 0)

    def test_squared_error(self):
        testsgd = SGD()
        squared_error = testsgd.error(10, 1, 1, 11)
        self.assertEqual(squared_error, 0)

    def test_get_gradient(self):
        testsgd = SGD()
        gradient = testsgd.get_gradient(6, 18, [3, 2])
        self.assertEqual(gradient, [-5.9998999999777425, -35.99640000013338])

    def test_vector_subtract(self):
        testsgd = SGD()
        vector = testsgd.vector_subtract([1,2,3,4], [1,1,1,1])
        self.assertEqual(vector, [0,1,2,3])

    def test_scalar_multiply(self):
        testsgd = SGD()
        vector = testsgd.scalar_multiply(3, [1,2,3])
        self.assertEqual(vector, [3,6,9])

    def test_stochastic_gradient_descent(self):
        testsgd = SGD()
        data_x = list(range(1, 100))
        data_y = list(range(1, 100))
        delta = [random.random(), random.random()]
        alpha, beta = testsgd.stochastic_gradient_descent(data_x, data_y, delta, 0.0001)
        self.assertEqual(round(alpha), 0)
        self.assertEqual(round(beta), 1)

if __name__ == "__main__":
#단위테스트
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    print("----------------------------------------------------------------------")