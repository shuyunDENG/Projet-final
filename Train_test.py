import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
#from model_timewarp import create_timewarp_model
import os

class MolecularDataset(Dataset):
    """分子动力学数据集 - 改进版本"""
    def __init__(self, data_indices, full_data, num_atom_types=4):
        """
        Args:
            data_indices: データ索引列表
            full_data: 完整数据数组
            num_atom_types: 原子类型数量
        """
        self.data_indices = data_indices
        self.full_data = full_data
        self.num_atom_types = num_atom_types
        _, _, self.num_atoms, _ = full_data.shape

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        data_idx = self.data_indices[idx]
        pair = self.full_data[data_idx]  # [2, num_atoms, 6]

        # 分离时间步
        t_data = pair[0]      # [num_atoms, 6] - t时刻数据
        t_tau_data = pair[1]  # [num_atoms, 6] - t+τ时刻数据

        # 分离位置和速度
        x_coords = t_data[:, :3]      # t时刻位置
        x_velocs = t_data[:, 3:]      # t时刻速度
        y_coords = t_tau_data[:, :3]  # t+τ时刻位置
        y_velocs = t_tau_data[:, 3:]  # t+τ时刻速度

        #----保持平移不变性，对称----
        x_coords = x_coords - np.mean(x_coords, axis=0, keepdims=True)
        y_coords = y_coords - np.mean(y_coords, axis=0, keepdims=True)
        # --------------

        # alanine-dipeptide的原子类型序列
        atom_types = torch.tensor([0, 1, 0, 2, 3, 3, 3, 0, 3, 0, 2, 1, 3, 0, 0, 2, 3, 3, 3, 3, 3, 3], dtype=torch.long)

        return {
            'atom_types': atom_types,
            'x_coords': torch.FloatTensor(x_coords),
            'x_velocs': torch.FloatTensor(x_velocs),
            'y_coords': torch.FloatTensor(y_coords),
            'y_velocs': torch.FloatTensor(y_velocs),
            'data_idx': data_idx  # 添加数据索引用于调试
        }

def analyze_data_distribution(data_path):
    """分析数据分布"""
    print("=== 数据分析 ===")
    data = np.load(data_path)
    print(f"数据形状: {data.shape}")

    # 分析位置和速度的分布
    positions = data[:, :, :, :3].reshape(-1, 3)
    velocities = data[:, :, :, 3:].reshape(-1, 3)

    print(f"位置统计:")
    print(f"  均值: {positions.mean(axis=0)}")
    print(f"  标准差: {positions.std(axis=0)}")
    print(f"  范围: {positions.min(axis=0)} 到 {positions.max(axis=0)}")

    print(f"速度统计:")
    print(f"  均值: {velocities.mean(axis=0)}")
    print(f"  标准差: {velocities.std(axis=0)}")
    print(f"  范围: {velocities.min(axis=0)} 到 {velocities.max(axis=0)}")

    return data

def normalize_data(data):
    """标准化数据"""
    print("=== 数据标准化 ===")
    normalized_data = data.copy()

    # 分别标准化位置和速度
    positions = data[:, :, :, :3]
    velocities = data[:, :, :, 3:]

    # 计算全局统计量
    pos_mean = positions.mean()
    pos_std = positions.std()
    vel_mean = velocities.mean()
    vel_std = velocities.std()

    print(f"位置标准化: 均值={pos_mean:.4f}, 标准差={pos_std:.4f}")
    print(f"速度标准化: 均值={vel_mean:.4f}, 标准差={vel_std:.4f}")

    # 标准化
    normalized_data[:, :, :, :3] = (positions - pos_mean) / pos_std
    normalized_data[:, :, :, 3:] = (velocities - vel_mean) / vel_std

    return normalized_data, (pos_mean, pos_std, vel_mean, vel_std)

def compute_physics_metrics(model, dataloader, device):
    """计算物理相关的指标"""
    model.eval()
    total_position_error = 0
    total_velocity_error = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            atom_types = batch['atom_types'].to(device)
            x_coords = batch['x_coords'].to(device)
            x_velocs = batch['x_velocs'].to(device)
            y_coords = batch['y_coords'].to(device)
            y_velocs = batch['y_velocs'].to(device)

            # 预测
            pred_coords, pred_velocs, _ = model(
                atom_types, x_coords, x_velocs, reverse=True
            )

            # 计算L2误差
            pos_error = torch.mean((pred_coords - y_coords) ** 2)
            vel_error = torch.mean((pred_velocs - y_velocs) ** 2)

            total_position_error += pos_error.item()
            total_velocity_error += vel_error.item()
            total_samples += 1

    return total_position_error / total_samples, total_velocity_error / total_samples

def train_timewarp_model_debug(
    data_path='training_pairs_augmented_1.npy',
    num_epochs=100,
    batch_size=8,
    learning_rate=5e-4,
    test_size=0.2,
    normalize=True,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """改进的训练函数，包含调试功能"""

    print("=== 开始训练诊断 ===")

    # 1. 分析和加载数据
    full_data = analyze_data_distribution(data_path)

    # 2. 数据标准化
    if normalize:
        full_data, norm_stats = normalize_data(full_data)

    # 3. 划分训练集和测试集
    num_samples = len(full_data)
    indices = np.arange(num_samples)
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=42, shuffle=True
    )

    print(f"训练集大小: {len(train_indices)}")
    print(f"测试集大小: {len(test_indices)}")

    # 检查是否有重叠
    overlap = set(train_indices) & set(test_indices)
    print(f"训练集和测试集重叠: {len(overlap)} 个样本")

    # 4. 创建数据集
    train_dataset = MolecularDataset(train_indices, full_data)
    test_dataset = MolecularDataset(test_indices, full_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 5. 创建模型
    config = {
        'num_atom_types': 4,
        'embedding_dim': 32,
        'hidden_dim': 64,
        'num_transformer_layers': 2,
        'num_flow_layers': 4,
        'lengthscales': [0.2, 0.8, 1.5]  # 调整长度尺度，避免整除问题
    }

    model = create_timewarp_model(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"模型参数量: {sum([p.numel() for p in model.parameters()]):,}")

    # 6. 训练历史记录
    train_losses = []
    test_losses = []
    position_errors = []
    velocity_errors = []

    # 7. 训练循环
    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0
        num_train_batches = 0

        for batch in train_loader:
            atom_types = batch['atom_types'].to(device)
            x_coords = batch['x_coords'].to(device)
            x_velocs = batch['x_velocs'].to(device)
            y_coords = batch['y_coords'].to(device)
            y_velocs = batch['y_velocs'].to(device)

            optimizer.zero_grad()
            z_coords, z_velocs, log_likelihood = model(
                atom_types, x_coords, x_velocs, y_coords, y_velocs, reverse=False
            )

            clip_limit = 10.0  # 可调
            coords_penalty = torch.relu(torch.abs(z_coords) - clip_limit).sum() / z_coords.numel()
            loss = -log_likelihood.mean() + 0.001 * coords_penalty
            loss.backward()


            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()


            train_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = train_loss / num_train_batches

        # 测试
        model.eval()
        test_loss = 0
        num_test_batches = 0

        with torch.no_grad():
            for batch in test_loader:
                atom_types = batch['atom_types'].to(device)
                x_coords = batch['x_coords'].to(device)
                x_velocs = batch['x_velocs'].to(device)
                y_coords = batch['y_coords'].to(device)
                y_velocs = batch['y_velocs'].to(device)

                z_coords, z_velocs, log_likelihood = model(
                    atom_types, x_coords, x_velocs, y_coords, y_velocs, reverse=False
                )

                loss = -log_likelihood.mean()
                test_loss += loss.item()
                num_test_batches += 1

        avg_test_loss = test_loss / num_test_batches

        # 计算物理指标
        if epoch % 10 == 0:
            pos_error, vel_error = compute_physics_metrics(model, test_loader, device)
            position_errors.append(pos_error)
            velocity_errors.append(vel_error)

            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Test Loss: {avg_test_loss:.4f}, "
                  f"Pos Error: {pos_error:.6f}, "
                  f"Vel Error: {vel_error:.6f}")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Test Loss: {avg_test_loss:.4f}")

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

        # 早停检查
        if epoch > 20 and len(test_losses) > 10:
            recent_test_losses = test_losses[-10:]
            if all(recent_test_losses[i] <= recent_test_losses[i+1] for i in range(len(recent_test_losses)-1)):
                print(f"早停：测试损失连续10个epoch上升")
                break

    # 8. 绘制训练曲线
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train_Loss_1', alpha=0.7)
    plt.plot(test_losses, label='Test_Loss_1', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training_vs_Test Loss_1')
    plt.yscale('log')

    plt.subplot(1, 3, 2)
    epochs_with_metrics = list(range(0, len(position_errors) * 10, 10))
    plt.plot(epochs_with_metrics, position_errors, 'o-', label='Position_Error_1')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('Position_Prediction_Error_1')
    plt.yscale('log')

    plt.subplot(1, 3, 3)
    plt.plot(epochs_with_metrics, velocity_errors, 'o-', label='Velocity_Error_1', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('Velocity_Prediction_Error_1')
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig('training_diagnostics.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 9. 保存模型和统计信息
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'position_errors': position_errors,
        'velocity_errors': velocity_errors,
        'normalization_stats': norm_stats if normalize else None
    }, 'timewarp_model_debug_addition.pth')

    return model, train_losses, test_losses

def test_model_predictions(model_path='timewarp_model_debug_addition.pth'):
    """测试模型预测质量"""
    print("=== 测试模型预测质量 ===")

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model = create_timewarp_model(checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 生成测试数据
    batch_size = 5
    num_atoms = 22
    atom_types = torch.tensor([[0, 1, 0, 2, 3, 3, 3, 0, 3, 0, 2, 1, 3, 0, 0, 2, 3, 3, 3, 3, 3, 3]] * batch_size)
    x_coords = torch.randn(batch_size, num_atoms, 3) * 0.1  # 小的随机坐标
    x_velocs = torch.randn(batch_size, num_atoms, 3) * 0.01  # 小的随机速度

    with torch.no_grad():
        # 多次预测，检查一致性
        predictions = []
        for _ in range(10):
            pred_coords, pred_velocs, _ = model(atom_types, x_coords, x_velocs, reverse=True)
            predictions.append((pred_coords, pred_velocs))

        # 计算预测的方差
        coord_preds = torch.stack([p[0] for p in predictions])
        veloc_preds = torch.stack([p[1] for p in predictions])

        coord_variance = torch.var(coord_preds, dim=0).mean()
        veloc_variance = torch.var(veloc_preds, dim=0).mean()

        print(f"坐标预测方差: {coord_variance:.6f}")
        print(f"速度预测方差: {veloc_variance:.6f}")

        # 检查预测的合理性
        pred_coords_mean = coord_preds.mean(dim=0)
        pred_velocs_mean = veloc_preds.mean(dim=0)

        print(f"预测坐标范围: {pred_coords_mean.min():.4f} 到 {pred_coords_mean.max():.4f}")
        print(f"预测速度范围: {pred_velocs_mean.min():.4f} 到 {pred_velocs_mean.max():.4f}")

# 使用示例
if __name__ == "__main__":
    # 诊断训练
    model, train_losses, test_losses = train_timewarp_model_debug(
        data_path='training_pairs_augmented_1.npy',
        num_epochs=100,
        batch_size=8,
        learning_rate=5e-4,
        normalize=True
    )

    # 测试模型质量
    test_model_predictions()
