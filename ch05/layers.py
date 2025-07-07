import numpy as np

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

x = np.array([[1.0, -0.5], [-2.0, 3.0]])
print(x)

mask = (x <= 0)
print(mask)

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx
    
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx
    
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # 손실 함수
        self.y = None     # softmax의 출력
        self.t = None     # 정답 레이블 (원-핫 벡터 또는 레이블 인덱스)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        # t가 원-핫 벡터인 경우와 레이블 인덱스인 경우를 모두 처리
        if self.t.size == self.y.size: # t가 원-핫 벡터인 경우
            dx = (self.y - self.t) / batch_size
        else: # t가 레이블 인덱스인 경우
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
            
        return dx
    
X = np.random.rand(2)  # 입력
W = np.random.rand(2, 3) # 가중치
B = np.random.rand(3)  # 편향

print("X.shape:", X.shape)  # (2,)
print("W.shape:", W.shape)  # (2, 3)
print("B.shape:", B.shape)  # (3,)

# 선형 변환
Y = np.dot(X, W) + B


X_dot_W = np.array([[0, 0, 0], [10, 10, 10]])
B = np.array([1, 2, 3])

print(X_dot_W)
print(X_dot_W + B)

dY = np.array([[1, 2, 3], [4, 5, 6]])
print(dY)

dB = np.sum(dY, axis=0)
print(dB)