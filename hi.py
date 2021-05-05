import numpy as np
from Space import space,random_space

N = 10
M =10000
hi = []
for i in range(M):
    if i % (M//10) == 0:
        print("%0.1f"%(i/M))
    h = random_space(N,i)
    hi.append(h)
np.savetxt("hi.txt", hi,fmt='%e', delimiter=',')
