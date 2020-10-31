import numpy as np


# weighted_output_i是第i层神经元加权求和并加上偏移量后的[mx1]的列向量
def sigmoided(weighted_output_i):
    return 1.0 / (1.0 + np.exp(-weighted_output_i))


# output_1是第i层输出层的的输出，是一个列向量，行数是第i层神经元的个数
def derivative_of_sigmoided_weighted_sum_with_sigmoid(output_i):
    return np.multiply(output_i, 1 - output_i)


# output_1是第i层输出层的的输出，是一个列向量，行数是第i层神经元的个数
def derivative_of_error_with_output(output_practical, output_desired):
    return output_practical - output_desired


# 用输出结果的分类index生成[1xn]的行向量，其中下标为gold_target的为1，其余为0
def vectorized_result(gold_target):
    result = np.zeros((1, 1))
    result[0][0] = gold_target
    return result


class BP_network(object):

    # nums为一个存有每一层神经元个数的List
    def __init__(self, nums):
        self.nums = nums
        self.layer_size = len(nums)  # nums的大小表示总层数
        # self.weights = [np.random.randn(n, m) for n, m in zip(nums[: -1], nums[1:])]
        # self.biases = [np.random.randn(1, m) for m in nums[1:]]
        # self.weights = [np.random.uniform(-0.1, 0.1, (n, m)) for n, m in zip(nums[: -1], nums[1:])]
        # self.biases = [np.random.uniform(0, 0.1, (1, m)) for m in nums[1:]]
        self.weights = [np.random.random((n, m))*0.001 - 0.0005 for n, m in zip(nums[: -1], nums[1:])]  # 权重，每一层为一个[nxm]的矩阵，初始化范围为（-0.05,0.05）
        self.biases = [np.random.random((1, m))*-0.0005 for m in nums[1:]]  # 偏移量，每一层为一个[1xm]的行向量，初始化范围为（-0.05,0.05）
        self.max_develop_right_rate = 0.0  # 开发集最大的正确率

    # r是学习率，train_data是训练集，develop_data是开发集，lmda为正则项的权重
    # epoch_count是对训练集中所有样本进行迭代的次数，batch_size为每次调整权重和偏移量需要back_prop的样本个数
    def regress(self, r, train_data, develop_data, epoch_count, batch_size):
        train_data_count = len(train_data)  # 训练集中样本的个数
        # 对训练集中的样本进行epoch_count次的迭代
        for epoch_index in range(1, epoch_count + 1):
            np.random.shuffle(train_data)  # 每次迭代前对训练集中的样本进行混洗
            batch_datas = [train_data[cur: cur + batch_size] for cur in range(0, train_data_count, batch_size)]
            # 对同一批次的样本取调整量的平均
            for batch_data in batch_datas:
                delta_ws_sum = [np.zeros(w.shape) for w in self.weights]  # 同一批次一共需要调整的权重
                delta_bs_sum = [np.zeros(b.shape) for b in self.biases]  # 同一批次一共需要调整的偏移量
                for input_i, output_i in batch_data:
                    delta_ws, delta_bs = self.back_prop_to_get_delta(input_i, output_i)
                    delta_ws_sum = [sum + delta for sum, delta in zip(delta_ws_sum, delta_ws)]
                    delta_bs_sum = [sum + delta for sum, delta in zip(delta_bs_sum, delta_bs)]
                # 调整权重
                self.weights = [old_weight - delta_sum * (r / batch_size) for old_weight, delta_sum in zip(self.weights, delta_ws_sum)]
                # 调整偏移量
                self.biases = [old_biase - delta_sum * (r / batch_size) for old_biase, delta_sum in zip(self.biases, delta_bs_sum)]

            # 输出error
            error = self.get_error(develop_data)
            print("epoch[{}/{}] error={}".format(epoch_index, epoch_count, error))

            if error < 0.01:
                ys = []
                for input_i, output_i in develop_data:
                    ys.append(self.get_output_practical(input_i))
                return ys
        ys = []
        for input_i, output_i in develop_data:
            ys.append(self.get_output_practical(input_i))
        return ys

    def get_output_practical(self, input_i):
        output_i = input_i
        for w, b in zip(self.weights, self.biases):
            # output_i最后会是倒数第二层神经元的输出，为倒数第三层神经元加权求和并过了sigmoid函数的输出
            output_i = input_i
            input_i = sigmoided(np.dot(input_i, w) + b)
        # 倒数第二层神经元输出只需要加权求即为输出层神经元的输出，无需再过sigmoid
        return (np.dot(output_i, self.weights[-1]) + self.biases[-1])[0][0]

    # 通过当前的权重和偏移量得到每一层神经元的输出
    def get_sigmoided_weighted_sums_with_last_softmax(self, output_1):
        # output_1为第1层（输入层）各神经元的输出，一个[1xn]的行向量
        sigmoided_weighted_sums = []  # 第1层到最后一层（输出层）每一层的输出
        for w, b in zip(self.weights, self.biases):
            sigmoided_weighted_sums.append(output_1)
            output_1 = sigmoided(np.dot(output_1, w) + b)
        # 倒数第二层神经元输出只需要加权求和即为输出层神经元的输出，无需再过sigmoid
        sigmoided_weighted_sums.append((np.dot(sigmoided_weighted_sums[-1], self.weights[-1]) + self.biases[-1])[0][0])
        # 里面存了layer_size个行向量，假设第i行输出层有n个神经元，第i个行向量是一个[1xn]的行向量
        # 第i个行向量的第j列存的是第i层输出层的第j个神经元的输出（sigmoided_weighted_sum）
        return sigmoided_weighted_sums

    # 通过反向传播获得需要调整的权重和偏移量
    def back_prop_to_get_delta(self, output_1, gold_target):
        delta_ws = [np.zeros(w.shape) for w in self.weights]  # 需要调整的权重
        delta_bs = [np.zeros(b.shape) for b in self.biases]  # 需要调整的偏移量
        # 假设第一层输入层有n个神经元，最后一层输出层有m个神经元
        # input_1为第1层（输入层）各神经元的输出，一个[1xn]的行向量
        # output_s为第1层到最后一层，每一层各神经元的输出结果，
        # 假设第i层有k个神经元，第k层为[1xk]的行向量
        output_s = self.get_sigmoided_weighted_sums_with_last_softmax(output_1)
        # output_practical为根据当前的权重和偏移量算出的实际输出层的输出
        output_practical = output_s[-1]
        # d_e_w_o为error对输出层求导的结果，是一个[1xm]的行向量
        d_e_w_o = derivative_of_error_with_output(output_practical, gold_target)
        # 输出层的前一层的输出到输出层的输出需要加的各偏移量的调整量，一个[1xm]的行向量
        delta_bs[-1] = d_e_w_o
        # 假设输出层前一层的神经元个数为l
        # 输出层的前一层的输出到输出层的输出需要乘的各权重的调整量，一个[lxm]的矩阵
        delta_ws[-1] = np.dot(output_s[-2].transpose(), delta_bs[-1])
        for inverted_layer_num in range(2, self.layer_size):
            # 前一层->当前层->后一层
            cur = -inverted_layer_num  # 当前层的倒序index
            front = -inverted_layer_num + 1  # 后一层的倒序index
            output_back = output_s[-inverted_layer_num - 1]  # 前一层的输出
            output_cur = output_s[cur]  # 当前层的输出
            # output_front = output_s[front]  # 后一层的输出
            # 假设前一层神经元的个数为i个，当前层神经元个数为j个，后一层神经元的个数为k个
            # d_e_w_o是error链式求导到到对前一层输出外面套的sigmoid求导的结果，是一个[1xk]的行向量
            # d_e_w_o = np.multiply(d_e_w_o, derivative_of_sigmoided_weighted_sum_with_sigmoid(output_front))
            # self.weights[front].transpose()是当前层输出到后一层输出需要乘权重，是一个([jxk]->)[kxj]的矩阵
            # d_e_w_o是error链式求导到当前层输出到后一层输出需要乘的各权重的结果，是一个[1xj]的行向量
            d_e_w_o = np.dot(delta_bs[front], self.weights[front].transpose())
            # d_o_w_s是当前层输出对当前层输出外面套的sigmoid求导的结果，是一个[1xj]的行向量
            d_o_w_s = derivative_of_sigmoided_weighted_sum_with_sigmoid(output_cur)
            # 前一层输出到当前层输出需要加的各偏移量的调整量，是一个[1xj]的矩阵
            delta_bs[cur] = np.multiply(d_e_w_o, d_o_w_s)
            # output_back.transpose()是前一层输出，是一个([1xi]->)[ix1]的矩阵
            # 前一层输出到当前层输出需要乘的各权重的调整量，是一个[ixj]的矩阵
            delta_ws[cur] = np.dot(output_back.transpose(), delta_bs[cur])
        return delta_ws, delta_bs

    # 获得结果的平均损失
    def get_error(self, data):
        error = 0.0
        for input_i, output_i in data:
            output_practical = self.get_output_practical(input_i)
            error += np.abs(output_i - output_practical)
        return error / len(data)
