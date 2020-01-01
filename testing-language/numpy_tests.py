import numpy as np

M = np.array([[1, 2],
              [3, 7]])


print(np.linalg.norm(M) ** 2 / M.shape[0])
