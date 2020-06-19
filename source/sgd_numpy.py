import numpy as np
import matplotlib.pyplot as plt
import random

class SGD:
    def predict(self, alpha, beta, x_i):
        '''
        해당 데이터로 linear function의 값 예측
        '''
        return beta * x_i + alpha

    def error(self, alpha, beta, x_i, y_i):
        '''
        오차함수
        '''
        return y_i - self.predict(alpha, beta, x_i)

    def squared_error(self, x_i, y_i, delta):
        '''
        제곱 오차 함수
        '''
        alpha, beta = delta
        return self.error(alpha, beta, x_i, y_i)**2
    
    def partial_difference_quotient(self, f, v, i, h):
        '''
        각각 다른 변수들은 고정, 하나의 변수만 h만큼 이동했을때의 변화율 구함
        h는 매우 작은 수

        v와 v에서 i만 아주 약간의 차이가 있는 w 두개의 리스트를
        각각 f(x)에 대입했을때의 차이를 h로 나눈 값 반환
        '''
        w = [v_j + (h if j == i else 0)
            for j, v_j in enumerate(v)]
        return (f(w) - f(v)) / h

    def estimate_gradient(self, f, v, h=0.00001):
        '''
        여러 변수의 각각의 편미분 리스트 반환 
        '''
        return [self.partial_difference_quotient(f, v, i, h)
                for i, _ in enumerate(v)]

    def get_gradient(self, x_i, y_i, coeff):
        '''
        손실함수인 제곱 오차 함수에 대해 편미분을 구함
        '''
        return self.estimate_gradient(lambda coeff:self.squared_error(x_i, y_i,coeff),
                                        coeff, h=1e-4)

    def vector_subtract(self, v, w):
        '''
        배열 차
        '''
        return [v_i - w_i for v_i, w_i in zip(v, w)]
        
    def scalar_multiply(self, c, v):
        '''
        스칼라배
        '''
        return [c * v_i for v_i in v]


    def stochastic_gradient_descent(self, x, y, delta, learning_rate_0=0.01):
        '''
        확률적 경사하강법 구현
        x, y = 입력 데이터
        delta = α, β 초기값

        100회동안 변화 없이 반복될경우 함수 종료됨
        '''
        min_delta, min_value = None, float("inf")
        no_improvement_cnt = 0
        inf_loop_cnt = 0
        learning_rate = learning_rate_0

        '''사용할 함수 정의'''
        target_fn = self.squared_error
        gradient_fn = self.get_gradient

        while((no_improvement_cnt < 100)):
            inf_loop_cnt += 1

            value = sum(target_fn(x_i, y_i, delta) for x_i, y_i in zip(x, y))

            if value < min_value:
                '''
                새로운 최소값을 찾으면 업데이트하고
                반복 횟수, step size 초기화
                '''
                min_delta, min_value = delta, value
                if inf_loop_cnt%20 == 1:
                    print('min delta update : ', min_delta)
                no_improvement_cnt = 0
                learning_rate = learning_rate_0
            else:
                '''
                최소값이 찾아지지 않으면 반복회수를 늘리고
                step size 축소
                '''
                no_improvement_cnt += 1
                if (no_improvement_cnt%20 == 5):
                    print("no improvement cnt : ", no_improvement_cnt)
                learning_rate *= 0.9

            '''
            delta 업데이트
            '''
            idxs = [i for i in range(len(x))]
            random.shuffle(idxs)

            batch_size = 20
            for i in idxs[:batch_size]:
                '''랜덤으로 뽑아 업데이트'''
                gradient_i = gradient_fn(x[i], y[i], delta)
                delta = self.vector_subtract(delta,
                            self.scalar_multiply(learning_rate, gradient_i))
        
        return min_delta

if __name__ == "__main__":

    data_n = 100
    data_x = np.linspace(-3, 3, data_n)
    data_y = 57 * data_x + np.random.randn(data_n) * 15 + 9

    #원본 데이터 보기
    plt.scatter(data_x, data_y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    #초기값 지정
    delta = [random.random(), random.random()]

    optimizer = SGD()

    #학습 시작
    alpha, beta = optimizer.stochastic_gradient_descent(data_x, data_y, delta, 0.0002)

    #결과값
    print("fianl : ", alpha, beta)

    #결과값 표시
    x = np.arange(-3, 4)
    y = beta*x + alpha

    plt.plot(x, y, linewidth=3, color='red')
    plt.scatter(data_x, data_y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()