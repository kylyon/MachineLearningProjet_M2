import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from typing import List

X = np.random.random((500, 2)) * 2.0 - 1.0
Y = np.array([[1, 0, 0] if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0 else
              [0, 1, 0] if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0 else
              [0, 0, 1] if -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0 else
              [0, 0, 0]for p in X])

X = X[[not np.all(arr == [0, 0, 0]) for arr in Y]]
Y = Y[[not np.all(arr == [0, 0, 0]) for arr in Y]]

points = np.array([
    [1, 1],
    [2, 1],
    [2, 2],
])
classes = np.array([
    1,
    1,
    -1
])

colors = ['blue', 'blue', 'red']

#plt.scatter(points[:, 0], points[:, 1], c=colors)
#plt.show()

plt.scatter(np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:,0], np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:,1], color='blue')
plt.scatter(np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:,0], np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:,1], color='red')
plt.scatter(np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:,0], np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:,1], color='green')
plt.show()
plt.clf()

W = np.random.uniform(-1.0, 1.0, 3)
Wr = np.random.uniform(-1.0, 1.0, 3)
Wg = np.random.uniform(-1.0, 1.0, 3)
Wb = np.random.uniform(-1.0, 1.0, 3)

print(Wr)

#test_points = []
#test_colors = []
#for row in range(0, 300):
#  for col in range(0, 300):
#    p = np.array([col / 100, row / 100])
#    c = 'lightcyan' if np.matmul(np.transpose(Wb), np.array([1.0, *p])) >= 0 else 'pink'
#    test_points.append(p)
#    test_colors.append(c)
#test_points = np.array(test_points)
#test_colors = np.array(test_colors)


#plt.scatter(test_points[:, 0], test_points[:, 1], c=test_colors)
#plt.scatter(points[:, 0], points[:, 1], c=colors)
#plt.show()

for _ in range(10000):
  k = np.random.randint(0, len(points))
  yk = classes[k]
  Xk = np.array([1.0, *points[k]])
  gXk = 1.0 if np.matmul(np.transpose(W), Xk) > 0 else -1.0
  W += 0.01 * (yk - gXk) * Xk

print(W)

for _ in range(100000):
  k = np.random.randint(0, len(X))
  yk = Y[k][0]
  Xk = np.array([1.0, *X[k]])
  gXk = 1.0 if np.matmul(np.transpose(Wr), Xk) > 0 else -1.0
  Wr += 0.01 * (yk - gXk) * Xk

print(Wr)

for _ in range(100000):
  k = np.random.randint(0, len(X))
  yk = Y[k][1]
  Xk = np.array([1.0, *X[k]])
  gXk = 1.0 if np.matmul(np.transpose(Wg), Xk) > 0 else -1.0
  Wg += 0.01 * (yk - gXk) * Xk

print(Wg)

for _ in range(100000):
  k = np.random.randint(0, len(X))
  yk = Y[k][2]
  Xk = np.array([1.0, *X[k]])
  gXk = 1.0 if np.matmul(np.transpose(Wb), Xk) > 0 else -1.0
  Wb += 0.01 * (yk - gXk) * Xk

print(Wb)


test_points = []
test_colors = []
for row in range(-100, 100):
  for col in range(-100, 100):
    p = np.array([col / 100, row / 100])
    maxMat = [np.matmul(np.transpose(Wr), np.array([1.0, *p])), np.matmul(np.transpose(Wg), np.array([1.0, *p])), np.matmul(np.transpose(Wb), np.array([1.0, *p]))]
    if p[0] == -1 and p[1] == -1:
        print(maxMat)
    if p[0] == 1 and p[1] == -1:
        print(maxMat)
    index = maxMat.index(max(maxMat))
    c = 'pink' if index == 1 else 'lightgreen' if index == 2 else 'lightcyan'
    test_points.append(p)
    test_colors.append(c)

test_points = np.array(test_points)
test_colors = np.array(test_colors)

#test_points = []
#test_colors = []
#W = [0.5668 , -0.2575 , -0.0394]
#for row in range(0, 300):
#  for col in range(0, 300):
#    p = np.array([col / 100, row / 100])
#    c = 'lightcyan' if np.matmul(np.transpose(W), np.array([1.0, *p])) >= 0 else 'pink'
#    test_points.append(p)
#    test_colors.append(c)
#test_points = np.array(test_points)
#test_colors = np.array(test_colors)

plt.scatter(test_points[:, 0], test_points[:, 1], c=test_colors)
plt.scatter(np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:,0], np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:,1], color='blue')
plt.scatter(np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:,0], np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:,1], color='red')
plt.scatter(np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:,0], np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:,1], color='green')

#plt.scatter(points[:, 0], points[:, 1], c=colors)
plt.show()

