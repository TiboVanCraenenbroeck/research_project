import numpy as np


a = np.zeros((5,5))
a[3] = [1.0 for i in a[3]]
a[3][3] = 1.0
print(a)
b = a.copy()
b = b.T
print(b)

d = b.T
print(d)