# encoding=utf-8
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter,MultipleLocator   ### 今天的主角设置y标签

#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

font1 = {'family' : 'SimHei',
'weight' : 'normal',
'size'   : 22,
}
font2 = {'family' : 'SimHei',
'weight' : 'normal',
'size'   : 20,
}
figsize = 15,8
fig, ax = plt.subplots(figsize=figsize)
ax.spines['top'].set_visible(False) # 去边框
ax.spines['right'].set_visible(False)
ax.xaxis.set_major_locator(MultipleLocator(1000)) # 设置x轴的刻度显示
plt.tick_params(labelsize=16)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('SimHei') for label in labels]
rolling_intv = 50

x = np.arange(10000)+1
c = np.loadtxt("rate_his_ratio0.txt",delimiter=',')
c = np.asarray(c)
dc = pd.DataFrame(c)
d = np.loadtxt("rate_his_ratio1.txt",delimiter=',')
d = np.asarray(d)
dd = pd.DataFrame(d)
c1 = np.loadtxt("rate_his_ratio2.txt",delimiter=',')
c1 = np.asarray(c1)
dc1 = pd.DataFrame(c1)
c2 = np.loadtxt("rate_his_ratio3.txt",delimiter=',')
c2 = np.asarray(c2)
dc2 = pd.DataFrame(c2)

plt.plot(x, np.array(dc.rolling(rolling_intv, min_periods=1).mean()), label='m = 0 n = 10000')
plt.plot(x,  np.array(dd.rolling(rolling_intv, min_periods=1).mean()), label='m = 20 n = 9980')
plt.plot(x,  np.array(dc1.rolling(rolling_intv, min_periods=1).mean()),label='m = 200 n = 9800')
plt.plot(x,  np.array(dc2.rolling(rolling_intv, min_periods=1).mean()), label='m = 2000 n = 8000')
plt.legend(prop=font2)

plt.margins(0)
ax.set_xlabel("训练次数",font1) #X轴标签
ax.set_ylabel("优化结果与最优解的比值",font1)  #Y轴标签
fig.align_labels() 
plt.grid()
plt.savefig("计算结果.pdf") #保存图
plt.show()  #显示图