import numpy as np
import cvxpy as cvx
import time 

# start_time = time.time()
# parameters and equations
o=100
p=3
u=0.51
ki=10**(-2) # increase the value of ki because the original value is too small 
B=2*10**6
Vu=1.1 
N0=10**(-10)
h=np.array([2.52690569892026e-06,1.44441867078275e-06,9.33368801933646e-07,2.23291617751371e-06,9.21244504161081e-07])


# optimization variables
tau_i = cvx.Variable(len(h))
fi = cvx.Variable(len(h))
ei = cvx.Variable(len(h))
a = 1 - cvx.sum(tau_i)

# optimization objective and constraints 
result = cvx.sum(-fi*10**6+(cvx.kl_div(tau_i,(cvx.multiply(ei,h)+tau_i*N0)/N0)\
                 +tau_i-(cvx.multiply(ei,h)+tau_i*N0)/N0)*B/Vu/np.log(2)) 
objective = cvx.Minimize(result)
constraints = [tau_i >= 0,a >= 0,ei >= 0,fi >= 0,ei + cvx.multiply(ki,fi**3) <= u*p*h*a]
prob = cvx.Problem(objective, constraints)
rewards = prob.solve(solver = cvx.MOSEK) # solve the problem by MOSEK


local_rate = (fi.value)*10**6
offloading_rate = B/Vu*tau_i.value*np.log2(1+ei.value*h/(N0*tau_i.value))
# mode = []
# for i in range(len(h)):
    # if local_rate[i] < offloading_rate[i]:
        # mode.append(1)
    # else:
        # mode.append(0)
# print(mode)
print(int(np.sum(local_rate + offloading_rate)))
# print("系统计算能效：")
# print((local_rate + offloading_rate)/(ei.value + ki*fi.value**3))


