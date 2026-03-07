"""
数据处理
    加载并预处理数据集   data_preprocess
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import minmax_scale
import src
def data_preprocess_series(dataset_name, dataset_path, state=2020):
    data = pd.read_csv(dataset_path + dataset_name)
    # src.RL_PPO_overlap.visualize_dataset(dataset_name, data, data)
    label_col = data.columns[-1]

    # 提取并转换标签列
    label = data[label_col].astype(float)
    label = pd.DataFrame(label)
    data.drop(label_col, axis=1, inplace=True)  # 移除原始标签列

    data.insert(0, 'label', label.values.ravel())

    # 分割数据集
    data0 = data[data['label'] == 0]
    data1 = data[data['label'] == 1]

    features = data.iloc[:, 1:]  # 特征部分
    return data, data0, data1, features, label


def data_preprocess(dataset_name, dataset_path, state=2020):
    """
    加载并预处理数据集
    返回经过标准化等处理后的数据集
    """
    data = pd.read_csv(dataset_path + dataset_name)

    # 去掉所有样本在某个特征上的取值都相同的列
    data = data.loc[:, data.nunique() > 1]  # 删除取值唯一的列

    # 确保少数类样本标签为1，多数类样本标签为0
    label_col = data.columns[-1]
    label_list = list(set(data[label_col]))

    # 根据样本数量判断类别
    count0 = len(data[data[label_col] == label_list[0]])
    count1 = len(data) - count0
    if count0 * 2 < count1:
        cla = {label_list[0]: 1, label_list[1]: 0}
    else:
        cla = {label_list[0]: 0, label_list[1]: 1}

    # 提取并转换标签列
    label = data[label_col].map(cla).astype(float)
    label = pd.DataFrame(label)
    data.drop(label_col, axis=1, inplace=True)  # 移除原始标签列
    samples = data.to_numpy()

    # 特征处理：对非数值型特征做OneHot编码
    encoder = OneHotEncoder()
    del_index = []
    lst = []
    for col_index in range(samples.shape[1]):
        try:
            samples[:, col_index] = samples[:, col_index].astype('float32')
        except ValueError:
            del_index.append(col_index)
            encoded = encoder.fit_transform(samples[:, col_index].reshape(-1, 1)).toarray()
            lst.append(encoded)

    # 删除无法转换为数值的列，并拼接OneHot编码结果
    samples = np.delete(samples, del_index, axis=1)
    for array in lst:
        samples = np.concatenate((samples, array), axis=1)

    # 归一化处理
    samples = minmax_scale(samples)

    # 重构数据集并插入标签列
    data = pd.DataFrame(samples)
    data.insert(0, 'label', label.values.ravel())

    # 分割数据集
    data0 = data[data['label'] == 0]
    data1 = data[data['label'] == 1]

    features = data.iloc[:, 1:]  # 特征部分
    return data, data0, data1, features, label