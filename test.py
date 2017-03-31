# import numpy as np
# l = []
#
# l.append(np.array([1,2,3]))
# l.append(np.array([4,5,6]))
#
# d = np.vstack(l)
# np.savetxt('test.csv', d, delimiter=',')
#
# r_d = np.genfromtxt('test.csv', delimiter=',', dtype=np.float64)
# i = 0

# a = 1.
# b = 1.
#
# x = tuple(map(sum, zip((a, b), (1.0,1.0))))
#
#
# print(a)
# print(b)

import time

for i in range(3):
    print('.', sep=' ', end='', flush=True)
    time.sleep(1)
print('')
print('Come on')