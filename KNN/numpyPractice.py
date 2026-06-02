import numpy as np
import time

# inizialize data
X = np.array([[1, 2, 3, 5],
              [5, 9, 8, 6]])
Y = np.array([[2, 0, 1],
              [0, 4, 1]])

n, m = X.shape
_, k = Y.shape

print("X shape:", X.shape)
print("Y shape:", Y.shape)

# squared Euclidean distances
start_time = time.time()
D_loop = np.empty((m, k))
for i in range(m):
    xi = X[:, i]
    for j in range(k):
        yj = Y[:, j]
        D_loop[i, j] = np.sum((xi - yj) ** 2)   # this represents ||xi - yj||_2^2
t_loop = time.time() - start_time

print("\nLoop time:      ", t_loop, "seconds")
print("D_loop =\n", D_loop)


# D = diag(X^T X) 1_{m,k} + 1_{m,k} diag(Y^T Y)^T - 2 X^T Y
start_time = time.time()

x_sq = np.sum(X**2, axis=0)
y_sq = np.sum(Y**2, axis=0)
D_vec = x_sq[:, None] + y_sq[None, :] - 2 * (X.T @ Y)

t_vec = time.time() - start_time

print("\nVectorized time:", t_vec, "seconds")
print("D_vec =\n", D_vec)

