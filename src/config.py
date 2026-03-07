# 训练相关参数
num_epochs = 200
pretrain_epochs = 10
ppo_epochs = 5
batch_size = 32
# learning_rate = 0.001

steps = 7
weight_class = 0.8 # Critic网络中分类损失的权重
clip_epsilon = 0.2 # PPO裁剪阈值{0.1,0.15,0.2,0.25,0.3}
flip_weight = 50.0 # 标签翻转奖励权重{10, 30, 50, 70, 90}
pert_weight = 0.05 # 扰动代价权重{0.01, 0.05, 0.1, 0.5, 1}