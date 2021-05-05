# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 10:45:26 2018

@author: Administrator
"""
import numpy as np
import cvxpy as cvx
import time
from Space import space,random_space
    
def lr_method(h):
    
    # parameters and equations
    o=100
    p=3
    u=0.51
    ki=10**(-2) # increase the value of ki because the original value is too small 
    B=2*10**6
    Vu=1.1 
    N0=10**(-10)
    wi = np.array([1 if i%2==1 else 1 for i in range(len(h))]) # default weights [1, 1.5, 1, 1.5, 1, 1.5, ...]

    # optimization variables
    tau_i = cvx.Variable(len(h))
    fi = cvx.Variable(len(h))
    ei = cvx.Variable(len(h))
    
    # optimization objective and constraints 
    result = cvx.sum(-cvx.multiply(wi,fi)*10**6+cvx.multiply(wi,(cvx.kl_div(tau_i,(cvx.multiply(ei,h)+tau_i*N0)/N0)\
                     +tau_i-(cvx.multiply(ei,h)+tau_i*N0)/N0))*B/Vu/np.log(2)) 
    objective = cvx.Minimize(result)
    constraints = [tau_i >= 0.0,cvx.sum(tau_i) <= 1.0,ei >= 0,fi >= 0,ei + cvx.multiply(ki,fi**3) <= u*p*h*(1-cvx.sum(tau_i))]
    prob = cvx.Problem(objective, constraints)
    rewards = prob.solve(solver = cvx.MOSEK) # solve the problem by MOSEK

    local_rate = wi*(fi.value)*10**6
    offloading_rate = wi*B/Vu*tau_i.value*np.log2(1+ei.value*h/(N0*tau_i.value))
    mode = []
    for i in range(len(h)):
        if local_rate[i] < offloading_rate[i]:
            mode.append(1)
        else:
            mode.append(0)
    
    # compute the sum_rate with binary offloading
    E = u*p*h*(1-np.sum(tau_i.value)) 
    sum_rate = 0
    for i in range(len(mode)):
        if mode[i] == 1:
            sum_rate += wi[i]*B/Vu*tau_i.value[i]*np.log2(1+E[i]*h[i]/(N0*tau_i.value[i]))
        else:
            sum_rate += wi[i]*(E[i]/ki)**(1/3)*10**6
    return sum_rate,mode
    
    

if __name__ == "__main__":
        
    n = 10 # Number of repetitions
    k = 5 # Number of users
    
    total_time = []
    gain_his = []
    h = space(k)
    for i in range(n):
        if i % (n//10) == 0:
            print("%0.1f"%(i/n))
        
        # test CD method. Given h, generate the max mode
        start_time=time.time()
        gain0, M0 = lr_method(h)
        total_time.append(time.time()-start_time)
        gain_his.append(gain0)
    print("gain/max ratio: ", sum(gain_his)/n)
    print('time_cost:%s'%(sum(total_time)/n))


