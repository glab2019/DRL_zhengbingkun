#  #################################################################
#  Deep Reinforcement Learning for Online Ofﬂoading in Wireless Powered Mobile-Edge Computing Networks
#
#  This file contains the main code of DROO. It loads the training samples saved in ./data/data_#.mat, splits the samples into two parts (training and testing data constitutes 80% and 20%), trains the DNN with training and validation samples, and finally tests the DNN with test data.
#
#  Input: ./data/data_#.mat
#    Data samples are generated according to the CD method presented in [2]. There are 30,000 samples saved in each ./data/data_#.mat, where # is the user number. Each data sample includes
#  -----------------------------------------------------------------
#  |       wireless channel gain           |    input_h            |
#  -----------------------------------------------------------------
#  |       computing mode selection        |    output_mode        |
#  -----------------------------------------------------------------
#  |       energy broadcasting parameter   |    output_a           |
#  -----------------------------------------------------------------
#  |     transmit time of wireless device  |    output_tau         |
#  -----------------------------------------------------------------
#  |      weighted sum computation rate    |    output_obj         |
#  -----------------------------------------------------------------
#
#
#  References:
#  [1] 1. Liang Huang, Suzhi Bi, and Ying-Jun Angela Zhang, "Deep Reinforcement Learning for Online Offloading in Wireless Powered Mobile-Edge Computing Networks," in IEEE Transactions on Mobile Computing, early access, 2019, DOI:10.1109/TMC.2019.2928811.
#  [2] S. Bi and Y. J. Zhang, “Computation rate maximization for wireless powered mobile-edge computing with binary computation ofﬂoading,” IEEE Trans. Wireless Commun., vol. 17, no. 6, pp. 4177-4190, Jun. 2018.
#
# version 1.0 -- July 2018. Written by Liang Huang (lianghuang AT zjut.edu.cn)
#  #################################################################


import scipy.io as sio                     # import scipy.io for .mat file I/
import numpy as np                         # import numpy
import os
# Implementated based on the PyTorch 
from memory import MemoryDNN
from optimization import bisection
import cvxpy as cvx
import time

def lr_method(h):
    
    # parameters and equations
    o=100
    p=3
    u=0.51
    ki=10**(-2) # increase the value of ki because the original value is too small 
    B=2*10**6
    Vu=1.1 
    N0=10**(-10)
    wi = np.array([1.5 if i%2==1 else 1 for i in range(len(h))]) # default weights [1, 1.5, 1, 1.5, 1, 1.5, ...]

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

def plot_rate(rate_his, rolling_intv=50):
    font1 = {'family' : 'SimHei',
    'weight' : 'normal',
    'size'   : 20,
    }
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl
    #解决中文显示问题
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    rate_array = np.asarray(rate_his)
    df = pd.DataFrame(rate_his)

    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(15, 8))

    plt.plot(np.arange(len(rate_array))+1, np.array(df.rolling(rolling_intv, min_periods=1).mean()), 'b')
    plt.fill_between(np.arange(len(rate_array))+1, df.rolling(rolling_intv, min_periods=1).min()[0], df.rolling(rolling_intv, min_periods=1).max()[0], color = 'b', alpha = 0.2)
    plt.ylabel('与最优解的比值',font1)
    plt.xlabel('训练次数',font1)
    plt.savefig("训练效果.pdf")
    plt.show()

def save_to_txt(rate_his, file_path):
    with open(file_path, 'w') as f:
        for rate in rate_his:
            f.write("%s \n" % rate)

if __name__ == "__main__":
    '''
        This algorithm generates K modes from DNN, and chooses with largest
        reward. The mode with largest reward is stored in the memory, which is
        further used to train the DNN.
        Adaptive K is implemented. K = max(K, K_his[-memory_size])
    '''

    N = 10                       # number of users
    n = 9980                    # number of time frames
    K = N                        # initialize K = N
    decoder_mode = 'OP'          # the quantization mode could be 'OP' (Order-preserving) or 'KNN'
    Memory = 1024                # capacity of memory structure
    Delta = 32                   # Update interval for adaptive K

    print('#user = %d, #channel=%d, K=%d, decoder = %s, Memory = %d, Delta = %d'%(N,n,K,decoder_mode, Memory, Delta))
    # Load data
    channel = sio.loadmat('./data/data_%d' %N)['input_h']
    rate = sio.loadmat('./data/data_%d' %N)['output_obj'] # this rate is only used to plot figures; never used to train DROO.

    # increase h to close to 1 for better training; it is a trick widely adopted in deep learning
    channel = channel * 1000000

    # generate the train and test data sample index
    # data are splitted as 80:20
    # training data are randomly sampled with duplication if n > total data size

    split_idx = int(.8 * len(channel))
    num_test = min(len(channel) - split_idx, n - int(.8 * n)) # training data size


    mem = MemoryDNN(net = [N, 120, 80, N],
                    learning_rate = 0.01,
                    training_interval=10,
                    batch_size=128,
                    memory_size=Memory
                    )

    start_time = time.time()

    rate_his = []
    rate_his_ratio = []
    mode_his = []
    k_idx_his = []
    K_his = []
    for i in range(20):
        h = channel[i,:]
        r_max, m_max = lr_method(h/1000000)
        rate_his.append(r_max)
        result = rate[i][0]
        rate_his_ratio.append(r_max/result)
        mem.encode(h,m_max)
        mode_his.append(m_max)
        
    for i in range(n):
        if i % (n//10) == 0:
           print("%0.1f"%(i/n))
        if i> 0 and i % Delta == 0:
            # index counts from 0
            if Delta > 1:
                max_k = max(k_idx_his[-Delta:-1]) +1;
            else:
                max_k = k_idx_his[-1] +1;
            K = min(max_k +1, N)

        if i < n - num_test:
            # training
            i_idx = i % split_idx
        else:
            # test
            i_idx = i - n + num_test + split_idx

        h = channel[i_idx,:]

        # the action selection must be either 'OP' or 'KNN'
        m_list = mem.decode(h, K, decoder_mode)

        r_list = []
        for m in m_list:
            r_list.append(bisection(h/1000000, m)[0])

        # encode the mode with largest reward
        mem.encode(h, m_list[np.argmax(r_list)])
        # the main code for DROO training ends here




        # the following codes store some interested metrics for illustrations
        # memorize the largest reward
        rate_his.append(np.max(r_list))
        rate_his_ratio.append(rate_his[-1] / rate[i_idx][0])
        # record the index of largest reward
        k_idx_his.append(np.argmax(r_list))
        # record K in case of adaptive K
        K_his.append(K)
        mode_his.append(m_list[np.argmax(r_list)])


    total_time=time.time()-start_time
    mem.plot_cost()
    plot_rate(rate_his_ratio)

    print("Averaged normalized computation rate:", sum(rate_his_ratio[-num_test: -1])/num_test)
    print('Total time consumed:%s'%total_time)
    print('Average time per channel:%s'%(total_time/n))

    if not os.path.exists("train"):
        os.makedirs("train")
    # save data into txt
    save_to_txt(k_idx_his, "train/k_idx_his.txt")
    save_to_txt(K_his, "train/K_his.txt")
    save_to_txt(mem.cost_his, "train/cost_his.txt")
    save_to_txt(rate_his_ratio, "train/rate_his_ratio.txt")
    save_to_txt(rate_his, "train/rate_his.txt")
    save_to_txt(mode_his, "train/mode_his.txt")
