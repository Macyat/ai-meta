import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


# 假设数据格式：输入特征601维，输出为标量（如水质参数）
class SyntheticDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.X = torch.randn(num_samples, 601)  # 601维输入特征
        self.y = torch.randn(num_samples)  # 模拟真实标签

        # 添加噪声（模拟手持设备误差）
        noise = 0.3 * torch.randn(num_samples)  # 随机噪声
        bias = 0.1 * self.X[:, 0]  # 系统误差（与第一个特征相关）
        self.y_noisy = self.y + noise + bias  # 含噪声标签

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_noisy[idx], self.y[idx]  # 返回含噪声标签和真实标签


# 定义动态加权损失函数
class DynamicWeightedLoss(nn.Module):
    def __init__(self, base_loss=nn.MSELoss(reduction="none"), alpha=0.5):
        """
        :param alpha: 权重衰减系数，控制对高损失样本的抑制强度（越大则抑制越强）
        """
        super().__init__()
        self.base_loss = base_loss
        self.alpha = alpha

    def forward(self, inputs, targets, reduction="mean"):
        # 计算每个样本的损失（shape: [batch_size]）
        losses = self.base_loss(inputs.squeeze(), targets)

        # 动态计算权重（指数衰减函数）
        weights = torch.exp(-self.alpha * losses.detach())

        # 归一化权重（保持梯度稳定性）
        weights = weights / (weights.sum() + 1e-8) * len(weights)

        # 加权损失
        weighted_loss = (weights * losses).sum()

        return weighted_loss if reduction == "sum" else weighted_loss / len(losses)


# 定义神经网络模型（输入601维，输出1维）
class SpectralModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(601, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


# 训练配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpectralModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 使用动态加权损失（alpha=0.5）
criterion = DynamicWeightedLoss(alpha=0.5)

# 数据加载（假设本地实验室数据）
train_dataset = SyntheticDataset(num_samples=2000)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 训练循环
for epoch in range(100):
    model.train()
    total_loss = 0.0

    for batch_X, batch_y_noisy, _ in train_loader:  # 注意：训练时使用含噪声标签
        batch_X, batch_y_noisy = batch_X.to(device), batch_y_noisy.to(device)

        # 前向传播
        preds = model(batch_X).squeeze()

        # 计算动态加权损失
        loss = criterion(preds, batch_y_noisy)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # 验证（假设有少量干净标签）
    model.eval()
    with torch.no_grad():
        # 使用合成数据中的真实标签验证
        X_val, _, y_val = train_dataset[:100]  # 取100个样本验证
        val_preds = model(X_val.to(device)).squeeze()
        val_loss = nn.MSELoss()(val_preds, y_val.to(device)).item()

    print(
        f"Epoch {epoch + 1} | Train Loss: {total_loss / len(train_loader):.4f} | Val Loss: {val_loss:.4f}"
    )
