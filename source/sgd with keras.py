from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

class SGD:
    def __init__(self):
        #학습 파라미터 설정
        self.learning_rate = 0.01
        self.training_steps = 1000
        self.display_step = 50
        #학습에 사용될 W와 b 초기값 생성
        self.W = tf.Variable(np.random.randn(), name="weight")
        self.b = tf.Variable(np.random.randn(), name="bias")
        #Training data 생성
        self.X, self.Y = make_regression(n_samples=100, n_features=1, bias=10.0,
                                        noise=15.0, random_state=2)
        self.Y = np.expand_dims(self.Y, axis=1)
        #X의 차원
        self.n_samples = self.X.shape[0]
        #Stochastic Gradient Descent
        self.optimizer = tf.optimizers.SGD(self.learning_rate)

    def linear_regression(self, x):
        return self.W * x + self.b

    def mean_square(self, y_pred, y_true):
        return tf.reduce_sum(tf.pow(y_pred-y_true, 2)) / (2 * self.n_samples)

    def run_optimization(self):
        with tf.GradientTape() as g:
            pred = self.linear_regression(self.X)
            loss = self.mean_square(pred, self.Y)

        gradients = g.gradient(loss, [self.W, self.b])
        
        self.optimizer.apply_gradients(zip(gradients, [self.W, self.b]))

    def training(self):
        for step in range(1, self.training_steps + 1):
            self.run_optimization()

            if step % self.display_step == 0:
                self.pred = self.linear_regression(self.X)
                self.loss = self.mean_square(self.pred, self.Y)
                print("step:%i, loss:%f, W:%f, b:%f" % (step, self.loss, self.W, self.b))

#단위테스트
import unittest
class TestSgd(unittest.TestCase):
    def test_sgd(self):
        testsgd = SGD()
        testsgd.training()
        W, b = testsgd.W, testsgd.b
        loss = testsgd.loss
        self.assertAlmostEqual(float(loss), 98.185654, places=2)
        self.assertAlmostEqual(float(W), 58.797782, places=2)
        self.assertAlmostEqual(float(b), 10.393133, places=2)

if __name__ == "__main__":
    #단위테스트
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    print("----------------------------------------------------------------------")

    #학습
    sgd = SGD()
    sgd.training()

    #결과 확인
    x = np.arange(-3, 3)
    y = sgd.W*x + sgd.b
    plt.scatter(sgd.X, sgd.Y, label="Original data")
    plt.plot(x, y, linewidth=3, color='red', label='Fitted line')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()  