import numpy as np
import scipy.signal

x = np.array([0,0,0,0,0,1,1,1,1])
gamma = 1.0
period = 3

b = np.zeros((period))
acc = 1
for i in range(period):
    b[i] = acc
    acc *= gamma

y = scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

y_alt = scipy.signal.lfilter(b, [1], x[::-1], axis=0)[::-1]

print(y)