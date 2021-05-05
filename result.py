import numpy as np
from CD import cd_method
from Space import space,random_space

N = 10
M =10000
hi = []
result = []
for i in range(M):
    if i % (M//10) == 0:
        print("%0.1f"%(i/M))
    h = random_space(N,i)
    m,m0 = cd_method(h)
    hi.append(h)
    result.append(m)
np.savetxt("result2.txt", result,delimiter=',')
