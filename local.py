import numpy as np
from Space import space

# parameters and equations
o=100
p=3
u=0.51
ki=10**(-26) # increase the value of ki because the original value is too small 
B=2*10**6
Vu=1.1 
N0=10**(-10)
k = 30
h = space(k)

sum = ((u*p*h/ki)**(1/3))/o 
print('local:%s'%int(np.sum(sum)))


