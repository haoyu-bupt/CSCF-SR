import src
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import copy
import time

current_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = current_path + '/data/'
os.makedirs(current_path + '/performance/', exist_ok=True)
performance_path = current_path + '/performance/'
performance = pd.DataFrame(columns=['IR', 'dataset_name', 'F1', 'G-mean', 'CF_Flip_Ratio'])

device = torch.device("cpu")
# GPU 设备检查
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 设置使用1号GPU
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("CUDA 可用:", torch.cuda.is_available())
print("当前设备:", device)
print("可用的 GPU 数量:", torch.cuda.device_count())
print("当前 GPU 名称:", torch.cuda.get_device_name(0))  # 0 是默认设备索引

if torch.cuda.is_available():
    print("环境变量 CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "未设置"))
else:
    print("警告：未检测到 GPU，使用 CPU 运行！")

# 清理 CUDA 缓存
torch.cuda.empty_cache()

print(torch.__version__)          # 应显示 2.0+
print(torch.cuda.is_available())  # 应输出 True
print(torch.version.cuda)         # 应显示 11.8 或 12.x


if __name__ == '__main__':
    index = 0
    for dataset_name in os.listdir(dataset_path):
        index += 1
        print('*' * 30, f"{index}: {dataset_name}", '*' * 30)
        try:
            data, data0, data1, features, label = src.data_preprocess.data_preprocess(dataset_name, dataset_path)
            # data, data0, data1, features, label = src.data_preprocess.data_preprocess_series(dataset_name, dataset_path)
        except Exception as e:
            print(f"错误({dataset_name}):", e)
            continue

        IR = len(data0) / len(data1)
        f, g, r = [], [], []

        # src.RL_PPO_overlap.visualize_dataset(dataset_name, data, data)
        num_epochs = src.config.num_epochs
        pretrain_epochs = src.config.pretrain_epochs
        batch_size = src.config.batch_size

        fold = 0

        # 使用 StratifiedKFold
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)
        for train_index, test_index in kf.split(features, label):
            fold += 1
            print('*' * 30, 'fold:', fold, '*' * 30, )
            X_train_val, X_test = features.iloc[train_index].values, features.iloc[test_index].values
            y_train_val, y_test = label.iloc[train_index].values, label.iloc[test_index].values
            y_train_val = y_train_val.squeeze(1)
            y_test = y_test.squeeze(1)

            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=fold)

            print('all y_train:', y_train.shape)
            print('pos y_train:', y_train.sum())
            print('all y_test:', y_test.shape)
            print('pos y_test:', y_test.sum())

            # 确保测试集包含正类样本
            assert y_test.sum() > 0, "测试集未包含正类样本！"

            # 转换为张量
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

            # 构建数据加载器
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            input_dim = X_train.shape[1]
            critic = src.RL_PPO_overlap.Critic(input_dim).to(device)
            actor0 = src.RL_PPO_overlap.Actor(input_dim).to(device)  # 处理多数类（y=0）
            actor1 = src.RL_PPO_overlap.Actor(input_dim).to(device)  # 处理少数类（y=1）

            critic = src.RL_PPO_overlap.pretrain_critic(critic, train_loader, epochs=pretrain_epochs)

            # final_cf_samples_pretrain, f1_pretrain, gmean_pretrain, cf_flip_ratio_pretrain = src.RL_PPO_overlap.evaluate_model(critic, actor0, actor1,
            #                                                                                X_test_tensor, y_test_tensor, 0.5)


            f1, gmean, cf_flip_ratio = src.RL_PPO_overlap.train_joint(
                dataset_name, data, critic, actor0, actor1,
                X_test_tensor, y_test_tensor,
                X_val_tensor, y_val_tensor, train_loader, epochs=num_epochs
            )

            f.append(f1)
            g.append(gmean)
            r.append(cf_flip_ratio)

        F1 = np.mean(f)
        G_mean = np.mean(g)
        CF_Flip_Ratio = np.mean(r)
        print("\n当前结果（反向扰动后）:")
        print(f"F1: {F1:.4f}")
        print(f"G-mean: {G_mean:.4f}")
        print(f"CF_flip_Ratio: {CF_Flip_Ratio:.4f}")
        print('f:', f)
        print('g:', g)
        print('r:', r)
        print(f"{index} 完成: {dataset_name}")

        performance.loc[index] = [IR, dataset_name, F1, G_mean, CF_Flip_Ratio]
        performance.to_csv(performance_path + '/performance-pima.csv', index=False)

    print('end')