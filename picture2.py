# encoding=utf-8
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

font1 = {'family' : 'SimHei',
'weight' : 'normal',
'size'   : 12,
}

figsize = 5.5,4.5
fig,ax = plt.subplots(figsize = figsize)

plt.tick_params(labelsize=8)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('SimHei') for label in labels]

time = [265.03,238.59,222.03,296.91]
patches = ax.bar(x=[1,2,3,4],height=time,width=0.5,
         tick_label=['m = 0 n = 10000','m = 20 n = 9800','m = 200 n = 9800','m = 2000 n = 8000'])
# axes.bar()中的参数说明：
# x表示不同长方条的位置坐标
# height表示长方条的高度
# width表示长方条的宽度
# tick_label表示不同长方条对应的标签
# axes的返回值是一个容器，里面包含着所有的长方条
# axes.bar()还有一些其他参数，可以参考官方文档
 
ax.set_yticks(np.arange(0,400,50))   # 设置y轴上的数值范围
    
# 给条形图添加数据标记
for rect in patches:
    height = rect.get_height()
    if height != 0:
        ax.text(rect.get_x() + rect.get_width()/10, height + 10,'{:.2f}'.format(height))
ax.set_xlabel("训练次数",font1) #X轴标签
ax.set_ylabel("训练时间/s",font1)  #Y轴标签
plt.savefig("训练时间.pdf") #保存图
plt.show()  #显示图