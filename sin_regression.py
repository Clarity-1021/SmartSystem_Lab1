import numpy as np
import bp_network_regression
import math
import matplotlib.pyplot as plt


# 获得训练样本点
data_size = 10000
xs = np.linspace(-math.pi, math.pi, data_size)
train_data = []
ys = []
for x in xs:
    ys.append(math.sin(x))
    train_data.append((x, ys[-1]))

# 获得随机测试点
Xs = np.random.random(99) * math.pi * 2 - math.pi
Xs = sorted(Xs)
develop_data = []
for x in Xs:
    develop_data.append((x, math.sin(x)))
print('---data_got---')

r = 0.001  # 学习率
epoch_count = 3000  # 迭代次数
batch_size = 1  # 调整权重每批样本数量

net = bp_network_regression.BP_network([1, 13, 1])
outs = net.regress(r, train_data, develop_data, epoch_count, batch_size)

# 画图，训练样本和随机测试结果
plt.figure()
plt.plot(xs, ys)
plt.plot(Xs, outs, color='red', linestyle='--')
plt.show()



