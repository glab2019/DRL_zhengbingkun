# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 10:45:26 2018

@author: Administrator
"""
import numpy as np
from scipy.special import lambertw
import time
from Space import space,random_space
    
def bisection(h, M, weights=[]):
    # the bisection algorithm proposed by Suzhi BI
    # average time to find the optimal: 0.012535839796066284 s

    # parameters and equations
    o=100
    p=3
    u=0.51
    eta1=((u*p)**(1.0/3))/o
    ki=10**-26   
    eta2=u*p/10**-10
    B=2*10**6
    Vu=1.1
    epsilon=B/(Vu*np.log(2))
    x = [] # a =x[0], and tau_j = a[1:]
    
    M0=np.where(M==0)[0]
    M1=np.where(M==1)[0]
    
    hi=np.array([h[i] for i in M0])
    hj=np.array([h[i] for i in M1])
    

    if len(weights) == 0:
        # default weights [1, 1.5, 1, 1.5, 1, 1.5, ...]
        weights = [1 if i%2==1 else 1 for i in range(len(M))]
        
    wi=np.array([weights[M0[i]] for i in range(len(M0))])
    wj=np.array([weights[M1[i]] for i in range(len(M1))])
    
    
    def sum_rate(x):
        sum1=sum(wi*eta1*(hi/ki)**(1.0/3)*x[0]**(1.0/3))
        sum2=0
        for i in range(len(M1)):
            sum2+=wj[i]*epsilon*x[i+1]*np.log(1+eta2*hj[i]**2*x[0]/x[i+1])
        return sum1+sum2

    def phi(v, j):
        return 1/(-1-1/(lambertw(-1/(np.exp( 1 + v/wj[j]/epsilon))).real))

    def p1(v):
        p1 = 0
        for j in range(len(M1)):
            p1 += hj[j]**2 * phi(v, j)

        return 1/(1 + p1 * eta2)

    def Q(v):
        sum1 = sum(wi*eta1*(hi/ki)**(1.0/3))*p1(v)**(-2/3)/3
        sum2 = 0
        for j in range(len(M1)):
            sum2 += wj[j]*hj[j]**2/(1 + 1/phi(v,j))
        return sum1 + sum2*epsilon*eta2 - v

    def tau(v, j):
        return eta2*hj[j]**2*p1(v)*phi(v,j)

    # bisection starts here
    delta = 0.005
    UB = 999999999
    LB = 0
    while UB - LB > delta:
        v = (float(UB) + LB)/2
        if Q(v) > 0:
            LB = v
        else:
            UB = v

    x.append(p1(v))
    for j in range(len(M1)):
        x.append(tau(v, j))

    return sum_rate(x), x[0], x[1:]

def exhaustive_method(h):
    N = len(h)
    max_M = np.zeros(N)
    gain0 = 0
    
    for i in range(2**N):
        if i % (2**N//10) == 0:
            print("%0.1f"%(i/2**N))
        Mi = str(bin(i))[2:].rjust(N,'0')
        M = []
        for i in Mi:
            if i == '0':
                M.append(0)
            else:
                M.append(1)
        M = np.array(M)
        gain,a,Tj= bisection(h,M)
        if gain > gain0:
            gain0 = gain
            max_M = M
    return gain0, max_M


if __name__ == "__main__":
    
    k = 15
    h = space(k)
    
    # test CD method. Given h, generate the max mode
    start_time=time.time()
    gain0, M0 = exhaustive_method(h)
    total_time=time.time()-start_time
    print('max y:%s'%int(gain0))
    print('time_cost:%s'%total_time)
    
    



























