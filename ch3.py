'''
계단 함수 구현
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0
'''

'''
numpy 배열도 지원하도록 수정한 구현
def step_function(x):
    y = x > 0
    return y.astype(int)
'''

'''
numpy 배열 지원한 계단 함수 구현
import numpy as np 
x = np.array([-1.0, 1.0, 2.0])
print("x: ", x)
y = x > 0
print("y: ", y)
y = y.astype(int)
print("y.astyp(int): ", y)
'''

'''
계단 함수의 그래프
import numpy as np 
import matplotlib.pylab as plt
def step_function(x):
    return np.array(x > 0, dtype = int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) # y축의 범위 지정
plt.show()
'''

'''
시그모이드 함수 구현
import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.array([-1.0, 1.0, 2.0])
sigmoid(x)
print("sigmoid(x): ", sigmoid(x))
'''

'''
시그모이드 함수 그래프
import numpy as np 
import matplotlib.pylab as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) # y축의 범위 지정
plt.show()
'''

'''
ReLU 함수 표현
import numpy as np
def relu(x):
    return np.maximum(0, x)
'''

'''
다차원 배열
import numpy as np
A = np.array([1, 2, 3, 4])
print(A)
print(np.ndim(A))
print(A.shape)
print(A.shape[0])

B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)
print(np.ndim(B))
print(B.shape)
'''

'''
행렬의 곱
import numpy as np
A = np.array([[1, 2], [3, 4]])
print(A.shape)
B = np.array([[5, 6], [7, 8]])
print(B.shape)
print(np.dot(A, B))
'''

'''
행렬의 곱
import numpy as np
A = np.array([[1, 2, 3], [4, 5, 6]])
print(A.shape)
B = np.array([[1, 2], [3, 4], [5, 6]])
print(B.shape)
print(np.dot(A, B))
'''

'''
신경망에서의 행렬 곱
import numpy as np
X = np.array([1, 2])
print(X.shape)
W = np.array([[1, 3, 5], [2, 4, 6]])
print(W)
print(W.shape)
Y = np.dot(X, W)
print(Y)
'''

'''
각 층의 신호 전달 구현
import numpy as np
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])
print(W1.shape)
print(X.shape)
print(B1.shape)
A1 = np.dot(X, W1) + B1

Z1 = sigmoid(A1)
print(A1)
print(Z1)


W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print(Z1.shape)
print(W2.shape) 
print(B2.shape) 

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

def identity_function(x):
    return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])
A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)  # 혹은 Y = A3

구현 정리
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)  # [0.31682708 0.69627909]
'''

'''
소프트맥스 함수 구현
import numpy as np
a = np.array([0.3, 2.9, 4.0])

exp_a = np.exp(a)  # 지수 함수
print(exp_a)
sum_exp_a = np.sum(exp_a)  # 지수 함수의 합
print(sum_exp_a)

y = exp_a / sum_exp_a
print(y)

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
'''

'''
소프트맥수 함수 구현시 주의점
import numpy as np
a = np.array([1010, 1000, 990])
np.exp(a) / np.sum(np.exp(a))

c = np.max(a)
a - c

np.exp(a - c) / np.sum(np.exp(a - c)) 

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  # 오버플로우 대책
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
'''
