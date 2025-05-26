# 1.5.1 numpy 불러오기기
import numpy as np
# 1.6 맷플롯립
import matplotlib.pyplot as plt
# 1.6.3 이미지 표시
from matplotlib.image import imread

# 1.5.2 numpy 배열열
'''
x = np.array([1.0, 2.0, 3.0])
print(x)
# [1. 2. 3.]
type(x)
# <class 'numpy.ndarray'>
'''

# 1.5.3 numpy 산술연산
'''
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
x + y
# array([3., 6., 9.])
x - y
# array([-1., -2., -3.])
x * y
# array([2., 8., 18.])
x / y
# array([0.5, 0.5, 0.5])
'''

# 1.5.4 numpy n차원 배열
'''
A = np.array([[1, 2,], [3, 4]])
print(A)
#[[1 2]
# [3 4]]
A.shape
# (2, 2)
A.dtype
#dtype('int64')
B = np.array([[3, 0],[0, 6]])
A + B
#array([[4, 2]
#        3, 10])
A * B
#array([[3, 0]
#        0, 24])
A * 10
#array([[10, 20]
#        30, 40])
'''

# 1.5.5 브로드캐스트
'''
A = np.array([[1, 2], [3, 4]])
B = np.array([10, 20])
A * B
#array([[10, 40]
#        30, 80])
'''

# 1.5.6 원소 접근
'''
x = np.array([[51, 55], [14, 19], [0, 4]])
print(x)
# [[51 55]
#  [14 19]
#  [ 0  4]]
x[0]
# array([51, 55])
x[0][1]
# 55

for row in x:
    print(row)
# [51 55]
# [14 19]
# [0 4]

x = x.flatten() # x를 1차원 배열로 변환(평탄화)
print(x)
# [51 55 14 19  0  4]

x > 15
# array([True, True, False, True, False, False])

x[x > 15]
# array([51, 55, 19])
'''

# 1.6.1 단순한 그래프 그리기
'''
x = np.arange(0, 6, 0.1)
y = np.sin(x)

plt.plot(x, y)
plt.show()
'''

# 1.6.2 pyplot의 기능
'''
x = np.arange(0, 6, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle="--", label="cos") # cos 함수는 점선으로 그리기
plt.xlabel("x") # x축 이름
plt.ylabel("y") # y축 이름
plt.title('sin & cos')  #제목
plt.show()
'''

# 1.6.3 이미지 표시
'''
img = imread('파일위치경로/crow.jpg') # 이미지 읽어오기(적절한 경로를 설정하세요!)

plt.imshow(img)
plt.show()
'''