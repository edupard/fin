import numpy as np
l = []

l.append(np.array([1,2,3]))
l.append(np.array([4,5,6]))

d = np.vstack(l)
np.savetxt('test.csv', d, delimiter=',')

r_d = np.genfromtxt('test.csv', delimiter=',', dtype=np.float64)
i = 0