import numpy as np

h = np.array([1,2,3,3,6,4])
X = np.array([1,2,3,4,5,6])
mature = np.bool_(h > 3)

print(mature)
print(X[mature])