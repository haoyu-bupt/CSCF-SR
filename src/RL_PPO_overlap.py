import src
import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import time


class Actor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        return self.net(x) * 0.1


class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.value = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        shared = self.shared(x)
        pred = torch.sigmoid(self.classifier(shared))
        # pred = self.classifier(shared)
        q_val = self.value(shared)
        return pred, q_val


class LambdaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, dist):
        a = self.net(dist)
        lambda_val = 0.1 / (1 + torch.exp(-a * dist))
        return lambda_val


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):

        # 计算 logits 的 sigmoid 概率
        # p = torch.sigmoid(inputs)
        p = inputs

        # 计算交叉熵损失
        bce_loss = nn.BCELoss()(inputs, targets)
        # bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # 计算 Focal Loss
        alpha = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha * ((1 - p) ** self.gamma) * bce_loss

        # 根据 reduction 参数进行损失归约
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def pretrain_critic(critic, loader, epochs):
    optimizer = optim.Adam(critic.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    # criterion = FocalLoss(alpha=1-(1/IR), gamma=2.0)

    for epoch in range(epochs):
        for X, y in loader:
            optimizer.zero_grad()
            preds, _ = critic(X)
            y = y.view_as(preds)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
        print(f"预训练第{epoch + 1}轮，损失: {loss.item():.4f}")

    return critic


class PPOTrainer:
    def __init__(self, critic, actor0, actor1):
        self.critic = critic
        self.actors = [actor0, actor1]
        # 获取设备信息
        self.device = next(critic.parameters()).device
        self.lambda_net = LambdaNet().to(self.device)

        self.optimizers = {
            'critic': optim.Adam(critic.parameters(), lr=1e-4),
            'actor0': optim.Adam(actor0.parameters(), lr=3e-4),
            'actor1': optim.Adam(actor1.parameters(), lr=3e-4),
            'lambda_net': optim.Adam(self.lambda_net.parameters(), lr=1e-3)
        }

        self.gamma = 0.99
        self.clip_epsilon = src.config.clip_epsilon
        self.ppo_epochs = src.config.ppo_epochs
        self.actual_step_list = []

    def compute_reward(self, x_orig, x_cf, y_orig):
        flip_weight = src.config.flip_weight
        pert_weight = src.config.pert_weight
        # 并且使用torch.no_grad()来避免梯度计算，因为这只是在评估样本，不需要更新网络
        with torch.no_grad():
            # pred_orig, _ = self.critic(x_orig)
            pred_cf, _ = self.critic(x_cf)
        # pred_orig = pred_orig.squeeze(1)
        pred_cf = pred_cf.squeeze(1)
        # 当反事实样本的预测类别与原始样本的真实类别不同时，值为1；否则为0
        flip = (y_orig.round() != pred_cf.round()).float()
        # flip = flip.squeeze()
        # 计算每个样本扰动向量的L2范数,系数表示这个成本在最终奖励中的权重较小
        pert_cost = torch.norm(x_cf - x_orig, dim=1)
        # 标签翻转奖励，成功使模型改变预测结果时获得高奖励
        # 扰动成本惩罚，扰动越大，惩罚越大
        cost = flip_weight * flip - pert_weight * pert_cost
        return cost


    # 多步扰动生成反事实
    def generate_multi_step_cf(self, x_orig, y_orig, actor_idx):
        steps = src.config.steps
        batch_size = x_orig.shape[0]
        x_cf = x_orig.clone()
        # 记录每个样本是否已经翻转标签
        flipped_mask = torch.zeros(batch_size, dtype=torch.bool, device=x_orig.device)
        # 确保y_orig是1D张量
        if y_orig.dim() > 1:
            y_orig = y_orig.squeeze()
        for i in range(steps):
            # 如果所有样本都已翻转，提前退出
            if flipped_mask.all():
                break

            with torch.no_grad():
                delta = self.actors[actor_idx](x_cf)
                # 指数衰减步长
                step_size = 1 * (0.8 ** i)

                # 只对未翻转的样本进行扰动
                for j in range(batch_size):
                    if not flipped_mask[j].item():  # 使用.item()转换为Python bool
                        x_cf[j] = x_cf[j] + step_size * delta[j]

                # 检查哪些样本的标签已经翻转
                pred, _ = self.critic(x_cf)

                # 确保pred是1D张量
                if pred.dim() > 1:
                    pred = pred.squeeze()
                elif pred.dim() == 0:  # 如果是标量，扩展为1D张量
                    pred = pred.unsqueeze(0)

                # 确保都是1D张量且长度为batch_size
                if pred.dim() > 0:
                    assert pred.shape[0] == batch_size, f"pred shape: {pred.shape}, expected batch_size: {batch_size}"
                if y_orig.dim() > 0:
                    assert y_orig.shape[
                               0] == batch_size, f"y_orig shape: {y_orig.shape}, expected batch_size: {batch_size}"

                newly_flipped = (pred.round() != y_orig.round()) & (~flipped_mask)

                # 更新翻转状态
                flipped_mask = flipped_mask | newly_flipped

        return x_cf, i + 1  # 返回实际步数

    def train_step(self, X_batch, y_batch):

        mask_major = y_batch.squeeze() == 0
        X_major = X_batch[mask_major]
        y_major = y_batch[mask_major]
        X_minor = X_batch[~mask_major]
        y_minor = y_batch[~mask_major]

        experiences = []

        # print('num X_major', X_major.size(0))
        # print('num X_minor', X_minor.size(0))

        # 处理多数类样本
        if X_major.size(0) > 0:
            with torch.no_grad():
                # delta = self.actors[0](X_major)
                # x_cf = X_major + delta
                # print("delta shape:", delta.shape)
                x_cf, actual_steps = self.generate_multi_step_cf(X_major, y_major, 0)
                delta = x_cf - X_major
                rewards = self.compute_reward(X_major, x_cf, torch.zeros(len(X_major), device=delta.device))
            self.actual_step_list.append(actual_steps)
            experiences.append((X_major, delta.detach(), rewards, 0, y_major))

        # 处理少数类样本
        if X_minor.size(0) > 0:
            with torch.no_grad():
                # delta = self.actors[1](X_minor)
                # print("delta shape:", delta.shape)
                # x_cf = X_minor + delta
                x_cf, actual_steps = self.generate_multi_step_cf(X_minor, y_minor, 1)
                delta = x_cf - X_minor
                rewards = self.compute_reward(X_minor, x_cf, torch.ones(len(X_minor), device=delta.device))
            self.actual_step_list.append(actual_steps)
            experiences.append((X_minor, delta.detach(), rewards, 1, y_minor))

        for ppo in range(self.ppo_epochs):
            # print('po_epochs', ppo)
            for x_orig, delta, rewards, actor_idx, y_subset in experiences:
                # 确保mu和scale在同一个设备
                actor = self.actors[actor_idx]
                mu = actor(x_orig)
                scale = torch.tensor(0.1, device=mu.device)  # scale的设备
                dist = torch.distributions.Normal(mu, scale)
                new_log_probs = dist.log_prob(delta).sum(1)

                with torch.no_grad():
                    old_log_probs = dist.log_prob(delta).sum(1)
                    _, values = self.critic(x_orig + delta)
                    values = values.squeeze()
                advantages = rewards - values

                # 计算PPO损失
                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # 更新Actor
                self.optimizers[f'actor{actor_idx}'].zero_grad()
                actor_loss.backward()
                self.optimizers[f'actor{actor_idx}'].step()

                # 计算λ并生成反向扰动
                delta = delta.view(-1, delta.shape[-1])  # 保证形状统一为 [batch_size, feature_dim]
                delta_norm = torch.norm(delta, dim=1, keepdim=True)  # 得到 [batch_size, 1]
                # print("delta shape:", delta.shape)
                # print("delta_norm shape:", delta_norm.shape)
                lambda_val = self.lambda_net(delta_norm)

                x_new = x_orig - lambda_val * delta
                x_cf = x_orig + delta

                # 更新Critic和LambdaNet
                preds_new, _ = self.critic(x_new)
                _, values = self.critic(x_cf)
                preds_new = preds_new.squeeze(1)
                values = values.squeeze(1)

                value_loss = nn.MSELoss()(values, rewards)
                # criterion = FocalLoss(alpha=1-(1/IR), gamma=2.0)
                # class_loss = criterion(preds_new, y_subset)
                class_loss = nn.BCELoss()(preds_new, y_subset)
                # class_weight = torch.tensor([1.0, IR])  # 根据IR调整少数类权重
                # criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight[1])
                # class_loss = criterion(preds_new, y_subset)
                total_loss = (1 - src.config.weight_class) * value_loss + src.config.weight_class * class_loss

                self.optimizers['critic'].zero_grad()
                self.optimizers['lambda_net'].zero_grad()
                total_loss.backward()
                self.optimizers['critic'].step()
                self.optimizers['lambda_net'].step()


def train_joint(dataset_name, data, critic, actor0, actor1, X_test, y_test, X_val, y_val, train_loader, epochs=100):
    trainer = PPOTrainer(critic, actor0, actor1)
    highest_score = -1
    best_f1 = best_gmean = best_cf_flip_ratio = -1
    count_not_improve = 0
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            trainer.train_step(X_batch, y_batch)

        # 用验证集寻找最优分类阈值
        if epoch % 10 == 0:
            with torch.no_grad():
                preds_val, _ = critic(X_val)
                preds_val = preds_val.cpu().numpy().flatten()
                y_val_np = y_val.cpu().numpy().flatten()
                best_val_f1  = best_thresh = 0
                for thresh in np.linspace(0.01, 0.99, 100):
                    y_pred = (preds_val >= thresh).astype(int)
                    f1 = f1_score(y_val_np, y_pred, zero_division=0)
                    if f1> best_val_f1:
                        best_val_f1, best_thresh = f1, thresh
                print('best_thresh:', best_thresh, 'best_val_f1:', best_val_f1)

                best_thresh = 0.5
                cf_minor_numpy, f1, gmean, cf_flip_ratio_minor = src.RL_PPO_overlap.evaluate_model(data, critic, actor0, actor1,
                                                                                               X_test, y_test, best_thresh)
                # # 只有用二维数据集的时候可以用这个可视化
                # visualize_dataset(dataset_name, data, cf_minor_numpy)
                score = f1 + gmean
                if score > highest_score:
                    highest_score = score
                    best_f1 = f1
                    best_gmean = gmean
                    best_cf_flip_ratio = cf_flip_ratio_minor
                    count_not_improve = 0
                else:
                    count_not_improve += 1

                print(f"第{epoch + 1}轮训练, best_f1:{best_f1:.4f}, best_gmean:{best_gmean:.4f}, best_cf_flip_ratio:{best_cf_flip_ratio:.4f}, count_not_improve: {count_not_improve:.1f}")
                if best_f1 != 0 and count_not_improve == 10:
                    break

    return best_f1, best_gmean, best_cf_flip_ratio


def evaluate_model(data, critic, actor0, actor1, X_test, y_test, thresh):
    """评估模型性能并生成反事实样本"""
    critic.eval()
    actor0.eval()
    actor1.eval()
    device = next(critic.parameters()).device

    # 提取特征和标签
    features = data.iloc[:, 1:].values
    labels = data.iloc[:, 0].values

    # 转换为Tensor并移到模型所在的设备
    X = torch.tensor(features, dtype=torch.float32).to(device)
    y = torch.tensor(labels, dtype=torch.float32).to(device)

    with torch.no_grad():
        # 预测原始测试数据

        preds, _ = critic(X_test)

        preds = preds.cpu().numpy().flatten()
        y_true = y_test.cpu().numpy().flatten()
        y_pred = (preds >= thresh).astype(int)

        # 计算评估指标
        # y_true = y_test.cpu().numpy()
        # y_pred_np = y_pred.cpu().numpy()
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        gmean = np.sqrt(sensitivity * specificity) if sensitivity != 0 and specificity != 0 else 0

        # 生成反事实样本
        cf_samples = torch.zeros_like(X)
        mask_major = y == 0
        indices_major = torch.where(mask_major)[0]
        indices_minor = torch.where(~mask_major)[0]

        # 多数类应用actor0
        if len(indices_major) > 0:
            X_major = X[indices_major]
            delta_major = actor0(X_major)
            cf_samples[indices_major] = X_major + delta_major

        # 少数类应用actor1
        if len(indices_minor) > 0:
            X_minor = X[indices_minor]
            delta_minor = actor1(X_minor)
            cf_samples[indices_minor] = X_minor + delta_minor

        # 验证反事实效果
        cf_pred, _ = critic(cf_samples)
        cf_pred = cf_pred.cpu().numpy().flatten()
        cf_pred = (cf_pred >= thresh).astype(int)

        # 计算少数类翻转率
        if len(indices_minor) > 0:
            cf_pred_minor = cf_pred[indices_minor.cpu().numpy()]
            # y_minor = y_true[indices_minor.cpu().numpy()]
            cf_flip_ratio_minor = (1 != cf_pred_minor).mean()
        else:
            cf_flip_ratio_minor = 0.0

    # print(f"\n测试结果：")
    print(f"混淆矩阵：\n    0预测\t1预测\n0实际：\t{tn}\t{fp}\n1实际：\t{fn}\t{tp}")
    print(f"F1: {f1:.4f}, G-mean: {gmean:.4f}, 少数类反事实翻类率: {cf_flip_ratio_minor:.2f}")

    cf_minor_numpy = cf_samples[indices_minor].cpu().numpy() if len(indices_minor) > 0 else np.array([])

    return cf_minor_numpy, f1, gmean, cf_flip_ratio_minor

# 可视化函数
def visualize_dataset(dataset_name, data_df, cf_samples):
    plt.figure(figsize=(4, 4))

    # 分离两类样本 此处第一列是标签，第二三列是特征
    data = data_df.values
    maj = data[data[:, 0] == 0]
    min = data[data[:, 0] == 1]

    # 绘制散点图
    plt.scatter(maj[:, 1], maj[:, 2], c='skyblue', s=30, edgecolor='k')
    plt.scatter(min[:, 1], min[:, 2], c='coral', s=50, edgecolor='k', marker='^')
    plt.scatter(cf_samples[:, 0], cf_samples[:, 1], c='red', s=50, edgecolor='k')


    plt.title('Counterfactual Explanations (' + dataset_name + ')', fontsize=14)
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.show()