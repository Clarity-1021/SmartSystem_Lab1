from PIL import Image
import numpy as np


# 提取像素点矩阵和正确分类结果tuple的List
# 从每类汉字的第start张开始提取size张图片的输入和输出
def extract_input_output(start, size):
    data = []  # 训练集
    # 12类汉字，每个取前train_size张作为测试集
    for i in range(1, 13):
        for j in range(start, start + size):
            image_path = 'train/%d/%d.bmp' % (i, j)  # 训练集图片的地址
            # print(image_path)
            image = Image.open(image_path)  # 训练图片
            # print(image)
            # 图片转化为长度为728的list，此时矩阵为boolean list，加0可以把boolean值转为01
            input_data = []
            img_array = np.asarray(image)
            for x in range(image.size[0]):
                for y in range(image.size[1]):
                    input_data.append(img_array[x, y] + 0)
            input_matrix  = np.zeros((1, 784))
            input_matrix[0] = input_data
            input_output = (input_matrix, i - 1)
            # print(input_output)
            data.append(input_output)
    # print(data)
    return data


def get_data(train_start, train_size, develop_start, develop_size):
    train_data = extract_input_output(train_start, train_size)
    develop_data = extract_input_output(develop_start, develop_size)
    # print('train_data=\n', train_data)
    # print('develop=\n',develop_data)
    return train_data, develop_data


train_strt = 1
train_siz = 600
# train_data = extract_input_output(train_strt, train_siz)
develop_strt = 601
develop_siz = 20
train_data, develop_data = get_data(train_strt, train_siz, develop_strt, develop_siz)

print('--data_got--')

train_name = 'npy/train_600'
develop_name = 'npy/develop_20'

np.save(train_name, train_data)
np.save(develop_name, develop_data)
train_path = train_name + '.npy'
develop_patn = develop_name + '.npy'
a = np.load(train_path, allow_pickle=True)
b = np.load(develop_patn, allow_pickle=True)
