# encoding=utf-8
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter   ### 今天的主角设置y标签

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
fig, ax = plt.subplots(figsize=figsize)

x = [5,6,7,8,9,10,11,12,13,14,15]
c = [805303,901720,999316,1090452,1178629,1266044,1350050,1431343,1512439,1592253,1670315]
d = [374631,449123,523640,598171,672711,747257,821806,896359,970913,1045470,1120028]
c1 = [771098,858779,940648,1017415,1089716,1158093,1222998,1284810,1343847,1400378,1454635]
c2 = [795310,901720,979344,1068224, 1151103,1244157,1326293,1395994,1488170,1566486,1648438]

plt.ylim(0,2000000)
plt.plot(x, c, marker='s',label='最优解',markerfacecolor='none',clip_on=False,linewidth = 1, ms = 7)
plt.plot(x, c2, marker='o',label='深度强化学习',markerfacecolor='none',clip_on=False,linewidth = 1, ms = 7)
plt.plot(x, c1, marker='d',label='完全卸载',markerfacecolor='none',clip_on=False,linewidth = 1, ms = 7,color = 'r')
plt.plot(x, d, marker='*',label='完全本地计算',markerfacecolor='none',clip_on=False,linewidth = 1, ms = 7)
plt.xticks(x)
plt.legend()

# def formatnum(x, pos):
    # return '$%.0f$x$10^{5}$' % (x/100000)   #注意修改两处的值，一个为x的除数，一个为对应的指数
# formatter = FuncFormatter(formatnum)
# ax.yaxis.set_major_formatter(formatter)

ax = plt.gca()  # 获取当前图像的坐标轴信息
ax.yaxis.get_major_formatter().set_powerlimits((0,1)) # 将坐标轴的base number设置为一位。

plt.margins(0)
ax.set_xlabel("WD的数量",font1) #X轴标签
ax.set_ylabel("总计算速率(bits/s)",font1)  #Y轴标签
fig.align_labels() 
plt.grid()
plt.savefig("计算结果.pdf") #保存图
plt.show()  #显示图