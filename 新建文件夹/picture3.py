import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
rolling_intv = 200

font1 = {'family' : 'SimHei',
'weight' : 'normal',
'size'   : 12,
}

figsize = 5.5,4.5
fig,ax = plt.subplots(figsize = figsize)

plt.tick_params(labelsize=8)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('SimHei') for label in labels]

c = np.array(np.loadtxt("K_his0.txt",delimiter=','))
dc = pd.DataFrame(c)
plt.plot(np.arange(len(c))+1, np.array(dc.rolling(rolling_intv, min_periods=1).mean()))
plt.ylabel('K值大小')
plt.xlabel('训练次数')
plt.savefig("K值.pdf")
plt.show()