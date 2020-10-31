import numpy as np
import bp_network_classify
import matplotlib.pyplot as plt

train_path = 'npy/train_600.npy'  # 训练集
develop_path = 'npy/develop_20.npy'  # 开发集

train_data = np.load(train_path, allow_pickle=True)
develop_data = np.load(develop_path, allow_pickle=True)
print('---data_got---')

r = 0.05  # 学习率
epoch_count = 300  # 迭代次数
batch_size = 5  # 调整权重每批样本数量
lmda = 0.001  # lmda
save_flag = True  # 存不存权重的Flag
save_index = 17  # 存的编号

net = bp_network_classify.BP_network([784, 99, 12])
ri_rts_t, errs_t, ri_rts_d, errs_d, epoch_count =net.classify(r, train_data, develop_data, epoch_count, batch_size, lmda, save_index)
xs = range(1, epoch_count+1)

plt.figure()
plt.plot(xs, ri_rts_t)
plt.plot(xs, ri_rts_d, color='red', linestyle='--')
plt.show()

plt.figure()
plt.plot(xs, errs_t)
plt.plot(xs, errs_d, color='red', linestyle='--')
plt.show()