import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import math

class AtomEmbedder(nn.Module):
    """原子嵌入层"""
    def __init__(self, num_atom_types: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_atom_types, embedding_dim)
    
    def forward(self, atom_types: Tensor) -> Tensor:
        """
        Args:
            atom_types: [batch_size, num_atoms] - 原子类型索引
        Returns:
            [batch_size, num_atoms, embedding_dim] - 原子嵌入
        """
        return self.embedding(atom_types)

class KernelSelfAttention(nn.Module):
    """
    Kernel Self-Attention (基于 RBF 核的自注意力)
    论文方程 (10) 和 (11)
    """
    def __init__(self, input_dim: int, output_dim: int, lengthscales: list):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lengthscales = lengthscales
        self.num_heads = len(lengthscales)
        
        # 每个头的变换矩阵 V
        self.value_projections = nn.ModuleList([
            nn.Linear(input_dim, output_dim // self.num_heads) 
            for _ in range(self.num_heads)
        ])
        
        # 最终的输出投影
        self.output_projection = nn.Linear(output_dim, output_dim)
        
    def forward(self, features: Tensor, coords: Tensor) -> Tensor:
        """
        Args:
            features: [batch_size, num_atoms, input_dim] - 输入特征
            coords: [batch_size, num_atoms, 3] - 原子坐标
        Returns:
            [batch_size, num_atoms, output_dim] - 输出特征
        """
        batch_size, num_atoms, _ = features.shape
        
        # 计算原子间距离矩阵
        # coords: [B, N, 3] -> [B, N, 1, 3] 和 [B, 1, N, 3]
        coords_i = coords.unsqueeze(2)  # [B, N, 1, 3]
        coords_j = coords.unsqueeze(1)  # [B, 1, N, 3]
        
        # 计算距离平方: ||x_i - x_j||^2
        distances_sq = torch.sum((coords_i - coords_j) ** 2, dim=-1)  # [B, N, N]
        
        # 多头注意力
        head_outputs = []
        for head_idx, lengthscale in enumerate(self.lengthscales):
            # 计算 RBF 核注意力权重 (方程 10)
            attention_weights = torch.exp(-distances_sq / (lengthscale ** 2))  # [B, N, N]
            
            # 归一化权重
            attention_weights = attention_weights / (attention_weights.sum(dim=-1, keepdim=True) + 1e-8)
            
            # 应用注意力 (方程 11)
            value = self.value_projections[head_idx](features)  # [B, N, output_dim//num_heads]
            attended_features = torch.bmm(attention_weights, value)  # [B, N, output_dim//num_heads]
            
            head_outputs.append(attended_features)
        
        # 拼接多头输出
        multi_head_output = torch.cat(head_outputs, dim=-1)  # [B, N, output_dim]
        
        # 最终投影
        return self.output_projection(multi_head_output)

class TransformerBlock(nn.Module):
    """Transformer 块 (包含 Kernel Self-Attention)"""
    def __init__(self, hidden_dim: int, lengthscales: list):
        super().__init__()
        self.kernel_attention = KernelSelfAttention(hidden_dim, hidden_dim, lengthscales)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward 网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
    def forward(self, features: Tensor, coords: Tensor) -> Tensor:
        """
        Args:
            features: [batch_size, num_atoms, hidden_dim]
            coords: [batch_size, num_atoms, 3]
        Returns:
            [batch_size, num_atoms, hidden_dim]
        """
        # 自注意力 + 残差连接
        attended = self.kernel_attention(features, coords)
        features = self.norm1(features + attended)
        
        # Feed-forward + 残差连接
        ffn_output = self.ffn(features)
        features = self.norm2(features + ffn_output)
        
        return features

class RealNVPCouplingLayer(nn.Module):
    """RealNVP 耦合层"""
    def __init__(self, input_dim: int, hidden_dim: int, condition_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.split_dim = input_dim // 2
        
        # 用于预测 scale 和 shift 的网络
        self.scale_net = nn.Sequential(
            nn.Linear(self.split_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim - self.split_dim)
        )
        
        self.shift_net = nn.Sequential(
            nn.Linear(self.split_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim - self.split_dim)
        )
        
    def forward(self, x: Tensor, condition: Tensor, reverse: bool = False) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: [batch_size, num_atoms, input_dim] - 输入数据
            condition: [batch_size, num_atoms, condition_dim] - 条件信息
            reverse: 是否反向传播
        Returns:
            transformed_x: 变换后的数据
            log_det_jacobian: 对数雅可比行列式
        """
        batch_size, num_atoms, _ = x.shape
        
        # 分割输入
        x1 = x[:, :, :self.split_dim]
        x2 = x[:, :, self.split_dim:]
        
        # 准备条件输入
        condition_input = torch.cat([x1, condition], dim=-1)
        
        # 计算 scale 和 shift
        scale = self.scale_net(condition_input)
        shift = self.shift_net(condition_input)
        
        # 应用变换
        if not reverse:
            # 前向传播
            y1 = x1
            y2 = x2 * torch.exp(scale) + shift
            log_det = scale.sum(dim=-1)  # [batch_size, num_atoms]
        else:
            # 反向传播 (采样)
            y1 = x1
            y2 = (x2 - shift) * torch.exp(-scale)
            log_det = -scale.sum(dim=-1)  # [batch_size, num_atoms]
        
        # 拼接输出
        y = torch.cat([y1, y2], dim=-1)
        
        return y, log_det

class FlowLayers(nn.Module):
    """归一化流层"""
    def __init__(self, input_dim: int, hidden_dim: int, condition_dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            RealNVPCouplingLayer(input_dim, hidden_dim, condition_dim)
            for _ in range(num_layers)
        ])
        
    def forward(self, x: Tensor, condition: Tensor, reverse: bool = False) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: [batch_size, num_atoms, input_dim]
            condition: [batch_size, num_atoms, condition_dim]
            reverse: 是否反向传播
        Returns:
            transformed_x: 变换后的数据
            total_log_det: 总的对数雅可比行列式
        """
        total_log_det = torch.zeros(x.shape[0], x.shape[1], device=x.device)
        
        layers = self.layers if not reverse else reversed(self.layers)
        
        for layer in layers:
            x, log_det = layer(x, condition, reverse)
            total_log_det += log_det
        
        return x, total_log_det

class TimewarpModel(nn.Module):
    """完整的 Timewarp 模型"""
    def __init__(
        self,
        num_atom_types: int,
        embedding_dim: int,
        hidden_dim: int,
        num_transformer_layers: int,
        num_flow_layers: int,
        lengthscales: list = [0.1, 0.2, 0.5, 1.0, 2.0]
    ):
        super().__init__()
        
        # 1. 原子嵌入器
        self.atom_embedder = AtomEmbedder(num_atom_types, embedding_dim)
        
        # 2. 输入投影
        self.input_projection = nn.Linear(embedding_dim + 6, hidden_dim)  # embedding + coords + velocs
        
        # 3. Transformer 块
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, lengthscales)
            for _ in range(num_transformer_layers)
        ])
        
        # 4. 归一化流层
        self.flow_layers = FlowLayers(
            input_dim=6,  # 3D coords + 3D velocities
            hidden_dim=hidden_dim,
            condition_dim=hidden_dim,
            num_layers=num_flow_layers
        )
        
        # 5. 先验分布参数
        self.register_parameter('log_scale_coords', nn.Parameter(torch.zeros(1)))
        self.register_parameter('log_scale_velocs', nn.Parameter(torch.zeros(1)))
        
    def forward(
        self,
        atom_types: Tensor,      # [batch_size, num_atoms]
        x_coords: Tensor,        # [batch_size, num_atoms, 3]
        x_velocs: Tensor,        # [batch_size, num_atoms, 3]
        y_coords: Tensor = None, # [batch_size, num_atoms, 3]
        y_velocs: Tensor = None, # [batch_size, num_atoms, 3]
        reverse: bool = False    # 是否为采样模式
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            atom_types: 原子类型
            x_coords: 输入坐标
            x_velocs: 输入速度
            y_coords: 目标坐标 (训练时使用)
            y_velocs: 目标速度 (训练时使用)
            reverse: 是否为采样模式
        Returns:
            output_coords: 输出坐标
            output_velocs: 输出速度
            log_likelihood: 对数似然 (训练时)
        """
        batch_size, num_atoms = atom_types.shape
        
        # 1. 原子嵌入
        atom_embeddings = self.atom_embedder(atom_types)  # [B, N, embedding_dim]
        
        # 2. 构建输入特征
        input_features = torch.cat([
            atom_embeddings,
            x_coords,
            x_velocs
        ], dim=-1)  # [B, N, embedding_dim + 6]
        
        # 3. 输入投影
        features = self.input_projection(input_features)  # [B, N, hidden_dim]
        
        # 4. 通过 Transformer 块
        for transformer_block in self.transformer_blocks:
            features = transformer_block(features, x_coords)
        
        if not reverse:
            # 训练模式：计算对数似然
            if y_coords is None or y_velocs is None:
                raise ValueError("训练模式需要提供目标坐标和速度")
            
            # 5. 目标数据
            y_data = torch.cat([y_coords, y_velocs], dim=-1)  # [B, N, 6]
            
            # 6. 通过流层
            z_data, log_det = self.flow_layers(y_data, features, reverse=False)
            
            # 7. 计算先验对数概率
            z_coords = z_data[:, :, :3]
            z_velocs = z_data[:, :, 3:]
            
            scale_coords = torch.exp(self.log_scale_coords)
            scale_velocs = torch.exp(self.log_scale_velocs)
            
            log_prior_coords = -0.5 * torch.sum((z_coords / scale_coords) ** 2, dim=-1)
            log_prior_velocs = -0.5 * torch.sum((z_velocs / scale_velocs) ** 2, dim=-1)
            log_prior = log_prior_coords + log_prior_velocs
            
            # 8. 计算总对数似然
            log_likelihood = log_prior + log_det
            
            return z_coords, z_velocs, log_likelihood
        
        else:
            # 采样模式：生成新的坐标和速度
            
            # 5. 从先验分布采样
            scale_coords = torch.exp(self.log_scale_coords)
            scale_velocs = torch.exp(self.log_scale_velocs)
            
            z_coords = torch.randn(batch_size, num_atoms, 3, device=x_coords.device) * scale_coords
            z_velocs = torch.randn(batch_size, num_atoms, 3, device=x_coords.device) * scale_velocs
            
            z_data = torch.cat([z_coords, z_velocs], dim=-1)
            
            # 6. 通过流层反向传播
            y_data, _ = self.flow_layers(z_data, features, reverse=True)
            
            # 7. 分离坐标和速度
            output_coords = y_data[:, :, :3]
            output_velocs = y_data[:, :, 3:]
            
            return output_coords, output_velocs, None

def create_timewarp_model(config: dict) -> TimewarpModel:
    """创建 Timewarp 模型的工厂函数"""
    return TimewarpModel(
        num_atom_types=config.get('num_atom_types', 10),
        embedding_dim=config.get('embedding_dim', 64),
        hidden_dim=config.get('hidden_dim', 128),
        num_transformer_layers=config.get('num_transformer_layers', 4),
        num_flow_layers=config.get('num_flow_layers', 8),
        lengthscales=config.get('lengthscales', [0.1, 0.2, 0.5, 1.0, 2.0])
    )

# 示例配置
example_config = {
    'num_atom_types': 10,        # 原子类型数量
    'embedding_dim': 64,         # 嵌入维度
    'hidden_dim': 128,           # 隐藏层维度
    'num_transformer_layers': 4,  # Transformer 层数
    'num_flow_layers': 8,        # 流层数
    'lengthscales': [0.1, 0.2, 0.5, 1.0, 2.0]  # 核函数的长度尺度
}

# 使用示例
if __name__ == "__main__":
    # 创建模型
    model = create_timewarp_model(example_config)
    
    # 准备输入数据
    batch_size = 2
    num_atoms = 10
    
    atom_types = torch.randint(0, 10, (batch_size, num_atoms))
    x_coords = torch.randn(batch_size, num_atoms, 3)
    x_velocs = torch.randn(batch_size, num_atoms, 3)
    y_coords = torch.randn(batch_size, num_atoms, 3)
    y_velocs = torch.randn(batch_size, num_atoms, 3)
    
    # 训练模式
    z_coords, z_velocs, log_likelihood = model(
        atom_types, x_coords, x_velocs, y_coords, y_velocs, reverse=False
    )
    
    print(f"训练模式 - 对数似然形状: {log_likelihood.shape}")
    
    # 采样模式
    pred_coords, pred_velocs, _ = model(
        atom_types, x_coords, x_velocs, reverse=True
    )
    
    print(f"采样模式 - 预测坐标形状: {pred_coords.shape}")
    print(f"采样模式 - 预测速度形状: {pred_velocs.shape}")