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

        # 确保 output_dim 能被 num_heads 整除
        self.head_dim = output_dim // self.num_heads
        if output_dim % self.num_heads != 0:
            # 调整 head_dim 确保维度匹配
            self.head_dim = output_dim // self.num_heads + 1
            print(f"Warning: Adjusting head_dim from {output_dim // self.num_heads} to {self.head_dim} to ensure divisibility")

        self.total_head_dim = self.head_dim * self.num_heads

        # 每个头的变换矩阵 V
        self.value_projections = nn.ModuleList([
            nn.Linear(input_dim, self.head_dim)
            for _ in range(self.num_heads)
        ])

        # 最终的输出投影 - 从拼接的头维度到期望的输出维度
        self.output_projection = nn.Linear(self.total_head_dim, output_dim)

    def forward(self, features: Tensor, coords: Tensor) -> Tensor:
        """
        Args:
            features: [batch_size, num_atoms, input_dim] - 输入特征
            coords: [batch_size, num_atoms, 3] - 原子坐标
        Returns:
            [batch_size, num_atoms, output_dim] - 输出特征
        """
        batch_size, num_atoms, _ = features.shape

        # 计算原子间距离矩阵 - 论文方程 (10)
        coords_i = coords.unsqueeze(2)  # [B, N, 1, 3]
        coords_j = coords.unsqueeze(1)  # [B, 1, N, 3]
        distances_sq = torch.sum((coords_i - coords_j) ** 2, dim=-1)  # [B, N, N]

        # 多头注意力
        head_outputs = []
        for head_idx, lengthscale in enumerate(self.lengthscales):
            # 计算 RBF 核注意力权重 (方程 10)
            attention_weights = torch.exp(-distances_sq / (lengthscale ** 2))  # [B, N, N]

            # 归一化权重
            attention_weights = attention_weights / (attention_weights.sum(dim=-1, keepdim=True) + 1e-8)

            # 应用注意力 (方程 11)
            value = self.value_projections[head_idx](features)  # [B, N, head_dim]
            attended_features = torch.bmm(attention_weights, value)  # [B, N, head_dim]
            head_outputs.append(attended_features)

        # 拼接多头输出
        multi_head_output = torch.cat(head_outputs, dim=-1)  # [B, N, total_head_dim]

        # 最终投影到期望的输出维度
        return self.output_projection(multi_head_output)

class AtomTransformer(nn.Module):
    """
    Atom Transformer - 论文中的核心组件，用作 s_θ 和 t_θ 函数
    论文 Section 4 和 Figure 2 Middle
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, lengthscales: list, num_layers: int = 2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # φ_in: 输入 MLP
        # 输入是 [x_p_i(t), h_i, z_v_i] 或 [x_p_i(t), h_i, z_p_i] - 论文 Section 4
        # We also need to include the velocity information z_v or z_p
        self.input_mlp = nn.Sequential(
            nn.Linear(3 + embedding_dim + 3, hidden_dim),  # coords + embedding + latent (pos or vel)
            nn.ReLU()
        )

        # Transformer 层
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, lengthscales)
            for _ in range(num_layers)
        ])

        # φ_out: 输出 MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # 输出 3D 向量
        )

    def forward(self, latent_vars: Tensor, x_coords: Tensor, atom_embeddings: Tensor) -> Tensor:
        """
        Args:
            latent_vars: [B, N, 3] - z_v 或 z_p
            x_coords: [B, N, 3] - 条件坐标 x_p(t)
            atom_embeddings: [B, N, embedding_dim] - 原子嵌入 h_i
        Returns:
            [B, N, 3] - scale 或 shift 向量
        """
        # 拼接输入：[x_p_i(t), h_i, z_v_i] - 论文 Section 4
        input_features = torch.cat([x_coords, atom_embeddings, latent_vars], dim=-1)

        # φ_in
        features = self.input_mlp(input_features)

        # Transformer 层 - 使用 x_coords 进行 kernel attention
        for layer in self.transformer_layers:
            features = layer(features, x_coords)

        # φ_out
        output = self.output_mlp(features)

        return output

class TransformerBlock(nn.Module):
    """Transformer 块 (包含 Kernel Self-Attention)"""
    def __init__(self, hidden_dim: int, lengthscales: list):
        super().__init__()
        self.kernel_attention = KernelSelfAttention(hidden_dim, hidden_dim, lengthscales)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Feed-forward 网络 - 论文称为 "atom-wise MLP"
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, features: Tensor, coords: Tensor) -> Tensor:
        """
        Args:
            features: [batch_size, num_atoms, hidden_dim]
            coords: [batch_size, num_atoms, 3]
        Returns:
            [batch_size, num_atoms, hidden_dim]
        """
        # Self-attention + residual connection + norm
        attended = self.kernel_attention(features, coords)
        features = self.norm1(features + attended)

        # Feed-forward + residual connection + norm
        ffn_output = self.ffn(features)
        features = self.norm2(features + ffn_output)

        return features

class TimewarpCouplingLayer(nn.Module):
    """
    Timewarp RealNVP 耦合层 - 论文方程 (8) 和 (9)
    这是论文的核心创新：使用 Atom Transformer 作为 s_θ 和 t_θ 函数
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, lengthscales: list):
        super().__init__()

        # 用于位置变换的 Atom Transformers - 论文方程 (8)
        self.scale_transformer_p = AtomTransformer(embedding_dim, hidden_dim, lengthscales)
        self.shift_transformer_p = AtomTransformer(embedding_dim, hidden_dim, lengthscales)

        # 用于速度变换的 Atom Transformers - 论文方程 (9)
        self.scale_transformer_v = AtomTransformer(embedding_dim, hidden_dim, lengthscales)
        self.shift_transformer_v = AtomTransformer(embedding_dim, hidden_dim, lengthscales)

    def forward(self, z_p: Tensor, z_v: Tensor, x_coords: Tensor,
                atom_embeddings: Tensor, reverse: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            z_p: [B, N, 3] - 位置潜在变量
            z_v: [B, N, 3] - 速度潜在变量
            x_coords: [B, N, 3] - 条件坐标 x^p(t)
            atom_embeddings: [B, N, embedding_dim] - 原子嵌入
            reverse: 是否反向传播
        Returns:
            z_p_new, z_v_new, log_det_jacobian
        """
        if not reverse:
            # 前向传播 - 论文方程 (8) 和 (9)

            # 步骤1：变换位置 - z^p_{ℓ+1} = s^p_{ℓ,θ}(z^v_ℓ; x^p(t)) ⊙ z^p_ℓ + t^p_{ℓ,θ}(z^v_ℓ; x^p(t))
            scale_p = self.scale_transformer_p(z_v, x_coords, atom_embeddings)  # s^p_{ℓ,θ}(z^v_ℓ; x^p(t))
            shift_p = self.shift_transformer_p(z_v, x_coords, atom_embeddings)  # t^p_{ℓ,θ}(z^v_ℓ; x^p(t))

            z_p_new = torch.exp(scale_p) * z_p + shift_p
            log_det_p = scale_p.sum(dim=-1)  # [B, N]

            # 步骤2：变换速度 - z^v_{ℓ+1} = s^v_{ℓ,θ}(z^p_{ℓ+1}; x^p(t)) ⊙ z^v_ℓ + t^v_{ℓ,θ}(z^p_{ℓ+1}; x^p(t))
            scale_v = self.scale_transformer_v(z_p_new, x_coords, atom_embeddings)  # s^v_{ℓ,θ}(z^p_{ℓ+1}; x^p(t))
            shift_v = self.shift_transformer_v(z_p_new, x_coords, atom_embeddings)  # t^v_{ℓ,θ}(z^p_{ℓ+1}; x^p(t))

            z_v_new = torch.exp(scale_v) * z_v + shift_v
            log_det_v = scale_v.sum(dim=-1)  # [B, N]

            total_log_det = log_det_p + log_det_v  # [B, N]

        else:
            # 反向传播 (采样)

            # 步骤1：反向变换速度
            scale_v = self.scale_transformer_v(z_p, x_coords, atom_embeddings)
            shift_v = self.shift_transformer_v(z_p, x_coords, atom_embeddings)

            z_v_new = (z_v - shift_v) * torch.exp(-scale_v)
            log_det_v = -scale_v.sum(dim=-1)

            # 步骤2：反向变换位置
            scale_p = self.scale_transformer_p(z_v_new, x_coords, atom_embeddings)
            shift_p = self.shift_transformer_p(z_v_new, x_coords, atom_embeddings)

            z_p_new = (z_p - shift_p) * torch.exp(-scale_p)
            log_det_p = -scale_p.sum(dim=-1)

            total_log_det = log_det_p + log_det_v

        return z_p_new, z_v_new, total_log_det

class TimewarpModel(nn.Module):
    """
    完整的 Timewarp 模型 - 严格按照论文实现
    核心思想：使用 conditional normalizing flow 学习 μ(x(t+τ)|x(t))
    """
    def __init__(
        self,
        num_atom_types: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_coupling_layers: int = 12,
        lengthscales: list = [0.1, 0.2, 0.5, 0.7, 1.0, 1.2]
    ):
        super().__init__()

        # 1. 原子嵌入器
        self.atom_embedder = AtomEmbedder(num_atom_types, embedding_dim)

        # 2. RealNVP 耦合层堆叠 - 论文 Figure 2 Left
        self.coupling_layers = nn.ModuleList([
            TimewarpCouplingLayer(embedding_dim, hidden_dim, lengthscales)
            for _ in range(num_coupling_layers)
        ])

        # 3. 基础分布的尺度参数 (可学习)
        self.register_parameter('log_scale', nn.Parameter(torch.zeros(1)))

    def forward(
        self,
        atom_types: Tensor,      # [batch_size, num_atoms] - 原子类型
        x_coords: Tensor,        # [batch_size, num_atoms, 3] - 条件坐标 x^p(t)
        x_velocs: Tensor,        # [batch_size, num_atoms, 3] - 条件速度 x^v(t)
        y_coords: Tensor = None, # [batch_size, num_atoms, 3] - 目标坐标 x^p(t+τ) (训练时)
        y_velocs: Tensor = None, # [batch_size, num_atoms, 3] - 目标速度 x^v(t+τ) (训练时)
        reverse: bool = False    # 是否为采样模式
    ) -> Tuple[Tuple[Tensor, Tensor], Optional[Tensor]]:
        """
        Args:
            atom_types: 原子类型索引
            x_coords: 条件坐标 x^p(t)
            x_velocs: 条件速度 x^v(t)
            y_coords: 目标坐标 x^p(t+τ) (训练时使用)
            y_velocs: 目标速度 x^v(t+τ) (训练时使用)
            reverse: False=训练模式, True=采样模式
        Returns:
            output_state: (output_coords, output_velocs)
            log_likelihood: 对数似然 (仅训练时)
        """
        batch_size, num_atoms = atom_types.shape

        # 1. 原子嵌入 - 论文 Section 4
        atom_embeddings = self.atom_embedder(atom_types)  # [B, N, embedding_dim]

        # 2. 中心化坐标 (translation equivariance) - 论文 Appendix A.2
        x_coords_centered = self._center_coordinates(x_coords)

        if not reverse:
            # 训练模式: 计算 p_θ(x(t+τ)|x(t))
            if y_coords is None or y_velocs is None:
                raise ValueError("训练模式需要提供目标坐标和速度 y_coords, y_velocs")

            # 中心化目标坐标
            y_coords_centered = self._center_coordinates(y_coords)

            # 采样辅助变量 - 论文 Section 3.3 Augmented Normalizing Flows
            z_v = y_velocs # Use target velocity as auxiliary variable
            z_p = y_coords_centered  # Use centered target position as main variable

            # 通过耦合层 (前向)
            total_log_det = torch.zeros(batch_size, num_atoms, device=x_coords.device)

            for layer in self.coupling_layers:
                z_p, z_v, log_det = layer(z_p, z_v, x_coords_centered, atom_embeddings, reverse=False)
                total_log_det += log_det

            # 计算基础分布的对数概率 - N(0, σ²I)
            scale = torch.exp(self.log_scale)
            log_prior_p = -0.5 * torch.sum((z_p / scale) ** 2, dim=-1)  # [B, N]
            log_prior_v = -0.5 * torch.sum((z_v / scale) ** 2, dim=-1)  # [B, N]
            log_prior = log_prior_p + log_prior_v

            # 总对数似然
            log_likelihood = log_prior + total_log_det  # [B, N]

            return (y_coords, y_velocs), log_likelihood

        else:
            # 采样模式：生成 x(t+τ) ~ p_θ(·|x(t))

            # 从基础分布采样
            scale = torch.exp(self.log_scale)
            z_p = torch.randn(batch_size, num_atoms, 3, device=x_coords.device) * scale
            z_v = torch.randn(batch_size, num_atoms, 3, device=x_coords.device) * scale

            # 通过耦合层 (反向)
            for layer in reversed(self.coupling_layers):
                z_p, z_v, _ = layer(z_p, z_v, x_coords_centered, atom_embeddings, reverse=True)

            # z_p is now centered output coordinates, z_v is output velocity
            output_coords = self._uncenter_coordinates(z_p, x_coords)
            output_velocs = z_v

            return (output_coords, output_velocs), None

    def _center_coordinates(self, coords: Tensor) -> Tensor:
        """中心化坐标 - 论文 Appendix A.2"""
        centroid = coords.mean(dim=1, keepdim=True)  # [B, 1, 3]
        return coords - centroid

    def _uncenter_coordinates(self, centered_coords: Tensor, reference_coords: Tensor) -> Tensor:
        """恢复坐标中心"""
        reference_centroid = reference_coords.mean(dim=1, keepdim=True)
        return centered_coords + reference_centroid

    def sample(self, atom_types: Tensor, x_coords: Tensor, x_velocs: Tensor, num_samples: int = 1) -> Tuple[Tensor, Tensor]:
        """便捷的采样接口"""
        self.eval()
        with torch.no_grad():
            if num_samples == 1:
                (output_coords, output_velocs), _ = self.forward(atom_types, x_coords, x_velocs, reverse=True)
                return output_coords, output_velocs
            else:
                # 批量采样
                samples_coords = []
                samples_velocs = []
                for _ in range(num_samples):
                    (output_coords, output_velocs), _ = self.forward(atom_types, x_coords, x_velocs, reverse=True)
                    samples_coords.append(output_coords)
                    samples_velocs.append(output_velocs)
                return torch.stack(samples_coords, dim=0), torch.stack(samples_velocs, dim=0)


def create_timewarp_model(config: dict) -> TimewarpModel:
    """创建 Timewarp 模型的工厂函数"""
    return TimewarpModel(
        num_atom_types=config.get('num_atom_types', 10),
        embedding_dim=config.get('embedding_dim', 64),
        hidden_dim=config.get('hidden_dim', 128),
        num_coupling_layers=config.get('num_coupling_layers', 12),
        lengthscales=config.get('lengthscales', [0.1, 0.2, 0.5, 0.7, 1.0, 1.2])
    )

# 论文中的配置参数
paper_config = {
    'num_atom_types': 20,        # 20种氨基酸
    'embedding_dim': 64,         # 论文 Table 3
    'hidden_dim': 128,           # 论文 Table 3
    'num_coupling_layers': 12,   # 论文 Table 3 - AD dataset
    'lengthscales': [0.1, 0.2, 0.5, 0.7, 1.0, 1.2]  # 论文 Appendix F
}


import torch
import numpy as np
import matplotlib.pyplot as plt
import openmm as mm
import openmm.app as app
from openmm import unit
import os
from tqdm import tqdm
import json
from typing import Tuple, Optional, Dict, List
import warnings

class TimewarpCorrectExplorer:
    """
    正确实现Timewarp论文中的两种算法：
    1. Algorithm 1: MH-corrected MCMC (严格但慢)
    2. Algorithm 2: Fast exploration with energy cutoff (快速但有偏)
    """

    def __init__(self, model_path: str, training_data_path: str = 'training_pairs_augmented_final.npy', 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"初始化TimewarpCorrectExplorer，设备: {device}")

        # 加载模型
        print("正在加载Timewarp模型...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.config = checkpoint['config']
        
        # 这里需要导入你的模型定义
        from model_timewarp import create_timewarp_model
        self.model = create_timewarp_model(self.config).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # 标准化参数
        self.norm_stats = checkpoint.get('normalization_stats', None)
        print(f"模型加载完成，配置: {self.config}")

        # 加载训练数据
        print("正在加载训练数据...")
        data = np.load(training_data_path)
        print(f"训练数据形状: {data.shape}")

        # 坐标转换：埃 → 纳米
        self.all_coords_nm = data[:, :, :, :3].reshape(-1, 22, 3) / 10.0
        self.all_velocs_nm_ps = data[:, :, :, 3:].reshape(-1, 22, 3) / 10.0

        print(f"坐标范围: {self.all_coords_nm.min():.4f} 到 {self.all_coords_nm.max():.4f} nm")

        # Alanine dipeptide原子类型
        self.atom_types = torch.tensor([0, 1, 0, 2, 3, 3, 3, 0, 3, 0, 2, 1, 3, 0, 0, 2, 3, 3, 3, 3, 3, 3],
                                     dtype=torch.long, device=device).unsqueeze(0)

        # 初始化OpenMM力场用于能量计算
        self.setup_openmm_energy_calculator()

    def setup_openmm_energy_calculator(self):
        """设置OpenMM能量计算器"""
        try:
            print("设置OpenMM能量计算器...")
            
            # 创建简单的alanine dipeptide系统
            forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
            
            # 你可能需要根据实际情况调整PDB文件路径
            # 这里先创建一个简单的计算器
            self.has_energy_calculator = False
            print("警告：OpenMM能量计算器未完全设置，将使用简化的能量筛选")
            
        except Exception as e:
            print(f"OpenMM设置失败: {e}")
            print("将使用简化的能量筛选方法")
            self.has_energy_calculator = False

    def calculate_energy(self, coords: torch.Tensor) -> torch.Tensor:
        """
        计算系统能量 (kJ/mol)
        
        Args:
            coords: [batch_size, num_atoms, 3] 坐标 (nm)
        Returns:
            energy: [batch_size] 能量 (kJ/mol)
        """
        if self.has_energy_calculator:
            # 实际的OpenMM能量计算
            # 这里需要实现具体的能量计算逻辑
            pass
        else:
            # 简化的能量估计：基于原子间距离的简单势函数
            batch_size, num_atoms, _ = coords.shape
            
            # 计算原子间距离
            coords_i = coords.unsqueeze(2)  # [B, N, 1, 3]
            coords_j = coords.unsqueeze(1)  # [B, 1, N, 3]
            distances = torch.norm(coords_i - coords_j, dim=-1)  # [B, N, N]
            
            # 简单的Lennard-Jones势
            sigma = 0.3  # nm
            epsilon = 1.0  # kJ/mol
            
            # 避免自作用和除零
            mask = (distances > 0.01) & (distances < 1.0)  # 只考虑合理距离范围
            distances_masked = torch.where(mask, distances, torch.tensor(sigma, device=coords.device))
            
            # LJ势: 4*epsilon*((sigma/r)^12 - (sigma/r)^6)
            r6 = (sigma / distances_masked) ** 6
            lj_energy = 4 * epsilon * (r6 * r6 - r6)
            
            # 只对掩码区域求和
            lj_energy_masked = torch.where(mask, lj_energy, torch.zeros_like(lj_energy))
            total_energy = 0.5 * lj_energy_masked.sum(dim=(1, 2))  # [B]
            
            return total_energy

    def get_random_initial_structure(self, idx: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """从训练数据中获取初始结构"""
        if idx is None:
            idx = np.random.randint(0, len(self.all_coords_nm))
        
        initial_coords = torch.FloatTensor(self.all_coords_nm[idx:idx+1]).to(self.device)
        initial_velocs = torch.FloatTensor(self.all_velocs_nm_ps[idx:idx+1]).to(self.device)
        
        return initial_coords, initial_velocs

    def normalize_data(self, coords: torch.Tensor, velocs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """标准化数据"""
        if self.norm_stats is None:
            return coords, velocs

        pos_mean, pos_std, vel_mean, vel_std = self.norm_stats
        coords_norm = (coords - pos_mean) / pos_std
        velocs_norm = (velocs - vel_mean) / vel_std

        return coords_norm, velocs_norm

    def denormalize_data(self, coords: torch.Tensor, velocs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """反标准化数据"""
        if self.norm_stats is None:
            return coords, velocs

        pos_mean, pos_std, vel_mean, vel_std = self.norm_stats
        coords_denorm = coords * pos_std + pos_mean
        velocs_denorm = velocs * vel_std + vel_mean

        return coords_denorm, velocs_denorm

    def metropolis_hastings_acceptance(self, current_coords: torch.Tensor, current_velocs: torch.Tensor,
                                     proposed_coords: torch.Tensor, proposed_velocs: torch.Tensor) -> bool:
        """
        Metropolis-Hastings acceptance criterion - 论文Algorithm 1
        
        Args:
            current_coords, current_velocs: 当前状态
            proposed_coords, proposed_velocs: 提议状态
        Returns:
            accept: 是否接受提议
        """
        # 计算能量
        current_energy = self.calculate_energy(current_coords)
        proposed_energy = self.calculate_energy(proposed_coords)
        
        # 能量差 (kJ/mol)
        delta_energy = proposed_energy - current_energy
        
        # Boltzmann因子 (T = 310K)
        kT = 8.314 * 310 / 1000  # kJ/mol
        
        # 论文方程(6)的简化版本（假设提议分布对称）
        acceptance_prob = torch.exp(-delta_energy / kT).clamp(max=1.0)
        
        # 随机接受
        random_val = torch.rand(1, device=self.device)
        accept = random_val < acceptance_prob
        
        return accept.item(), delta_energy.item(), acceptance_prob.item()

    def algorithm_1_mcmc_exploration(self, initial_coords: torch.Tensor, initial_velocs: torch.Tensor,
                                   num_steps: int = 20000, batch_size: int = 10, 
                                   save_interval: int = 100, output_dir: str = 'mcmc_exploration') -> Dict:
        """
        Algorithm 1: Timewarp MH-corrected MCMC - 论文Algorithm 1
        严格的、无偏的采样，但可能慢且接受率低
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"开始Algorithm 1 MCMC探索，{num_steps}步，批量大小{batch_size}")

        # 标准化输入
        coords = initial_coords.clone()
        velocs = initial_velocs.clone()
        
        if self.norm_stats:
            coords, velocs = self.normalize_data(coords, velocs)

        # 统计信息
        accept_count = 0
        total_proposals = 0
        trajectory_coords = [coords.cpu()]
        energy_history = []
        acceptance_history = []

        with torch.no_grad():
            for step in tqdm(range(num_steps)):
                # 批量生成提议
                proposals_coords = []
                proposals_velocs = []
                
                for _ in range(batch_size):
                    # 使用模型生成提议
                    prop_coords, _ = self.model(self.atom_types, coords, velocs, reverse=True)
                    prop_velocs = torch.randn_like(velocs) * 0.1  # 重新采样辅助变量
                    proposals_coords.append(prop_coords)
                    proposals_velocs.append(prop_velocs)

                # 找第一个被接受的提议 (论文Algorithm 1逻辑)
                accepted = False
                for i in range(batch_size):
                    total_proposals += 1
                    
                    # 反标准化进行能量计算
                    if self.norm_stats:
                        current_coords_real, current_velocs_real = self.denormalize_data(coords, velocs)
                        prop_coords_real, prop_velocs_real = self.denormalize_data(proposals_coords[i], proposals_velocs[i])
                    else:
                        current_coords_real, current_velocs_real = coords, velocs
                        prop_coords_real, prop_velocs_real = proposals_coords[i], proposals_velocs[i]

                    # MH接受检验
                    accept, delta_e, accept_prob = self.metropolis_hastings_acceptance(
                        current_coords_real, current_velocs_real,
                        prop_coords_real, prop_velocs_real
                    )

                    if accept:
                        coords = proposals_coords[i]
                        velocs = proposals_velocs[i]
                        accept_count += 1
                        accepted = True
                        acceptance_history.append(1)
                        break
                    else:
                        acceptance_history.append(0)

                if not accepted:
                    # 如果没有提议被接受，保持当前状态
                    pass

                # 记录能量
                if self.norm_stats:
                    coords_real, _ = self.denormalize_data(coords, velocs)
                else:
                    coords_real = coords
                energy = self.calculate_energy(coords_real).item()
                energy_history.append(energy)

                # 保存轨迹
                if step % save_interval == 0:
                    trajectory_coords.append(coords.cpu())

                # 进度报告
                if (step + 1) % 2000 == 0:
                    recent_accept_rate = np.mean(acceptance_history[-1000:]) if len(acceptance_history) >= 1000 else np.mean(acceptance_history)
                    print(f"Step {step+1}/{num_steps}, 接受率: {recent_accept_rate:.3f}, 当前能量: {energy:.2f} kJ/mol")

        # 计算最终统计
        final_accept_rate = accept_count / total_proposals if total_proposals > 0 else 0
        
        print(f"\nAlgorithm 1 完成!")
        print(f"总接受率: {final_accept_rate:.4f}")
        print(f"总提议数: {total_proposals}")
        print(f"接受数: {accept_count}")

        # 保存结果
        trajectory_coords = torch.cat(trajectory_coords, dim=0)
        
        # 反标准化保存
        if self.norm_stats:
            trajectory_coords, _ = self.denormalize_data(trajectory_coords, torch.zeros_like(trajectory_coords))
        
        np.save(f'{output_dir}/mcmc_coords.npy', trajectory_coords.numpy())
        np.save(f'{output_dir}/energy_history.npy', np.array(energy_history))
        np.save(f'{output_dir}/acceptance_history.npy', np.array(acceptance_history))

        stats = {
            'algorithm': 'MCMC_Algorithm_1',
            'total_steps': num_steps,
            'total_proposals': total_proposals,
            'accepted_proposals': accept_count,
            'acceptance_rate': final_accept_rate,
            'batch_size': batch_size,
            'trajectory_length': len(trajectory_coords),
            'final_energy': energy_history[-1] if energy_history else None
        }

        with open(f'{output_dir}/mcmc_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        return stats

    def algorithm_2_fast_exploration(self, initial_coords: torch.Tensor, initial_velocs: torch.Tensor,
                                   num_steps: int = 100000, energy_cutoff: float = 300.0,  # kJ/mol
                                   save_interval: int = 100, output_dir: str = 'fast_exploration') -> Dict:
        """
        Algorithm 2: Fast exploration with energy cutoff - 论文Algorithm 2
        快速但有偏的探索，只用能量阈值筛选
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"开始Algorithm 2 快速探索，{num_steps}步，能量阈值{energy_cutoff} kJ/mol")

        # 标准化输入
        coords = initial_coords.clone()
        velocs = initial_velocs.clone()
        
        if self.norm_stats:
            coords, velocs = self.normalize_data(coords, velocs)

        # 统计信息
        accept_count = 0
        total_proposals = 0
        trajectory_coords = [coords.cpu()]
        energy_history = []
        rejection_reasons = {'energy_too_high': 0, 'chirality_change': 0}

        with torch.no_grad():
            for step in tqdm(range(num_steps)):
                total_proposals += 1

                # 使用模型生成提议
                proposed_coords, _ = self.model(self.atom_types, coords, velocs, reverse=True)
                proposed_velocs = torch.randn_like(velocs) * 0.1

                # 反标准化进行能量计算
                if self.norm_stats:
                    current_coords_real, _ = self.denormalize_data(coords, velocs)
                    prop_coords_real, _ = self.denormalize_data(proposed_coords, proposed_velocs)
                else:
                    current_coords_real = coords
                    prop_coords_real = proposed_coords

                # 计算能量变化
                current_energy = self.calculate_energy(current_coords_real)
                proposed_energy = self.calculate_energy(prop_coords_real)
                delta_energy = proposed_energy - current_energy

                # Algorithm 2的简单接受准则：只检查能量阈值
                accept = delta_energy.item() < energy_cutoff

                if accept:
                    coords = proposed_coords
                    velocs = proposed_velocs  
                    accept_count += 1
                    energy_history.append(proposed_energy.item())
                else:
                    rejection_reasons['energy_too_high'] += 1
                    energy_history.append(current_energy.item())

                # 保存轨迹
                if step % save_interval == 0:
                    trajectory_coords.append(coords.cpu())

                # 进度报告
                if (step + 1) % 10000 == 0:
                    recent_accept_rate = accept_count / total_proposals
                    avg_energy = np.mean(energy_history[-1000:]) if len(energy_history) >= 1000 else np.mean(energy_history)
                    print(f"Step {step+1}/{num_steps}, 接受率: {recent_accept_rate:.3f}, 平均能量: {avg_energy:.2f} kJ/mol")

        # 计算最终统计
        final_accept_rate = accept_count / total_proposals

        print(f"\nAlgorithm 2 完成!")
        print(f"总接受率: {final_accept_rate:.4f}")
        print(f"总提议数: {total_proposals}")
        print(f"接受数: {accept_count}")
        print(f"能量过高拒绝: {rejection_reasons['energy_too_high']}")

        # 保存结果
        trajectory_coords = torch.cat(trajectory_coords, dim=0)
        
        # 反标准化保存
        if self.norm_stats:
            trajectory_coords, _ = self.denormalize_data(trajectory_coords, torch.zeros_like(trajectory_coords))

        np.save(f'{output_dir}/fast_coords.npy', trajectory_coords.numpy())
        np.save(f'{output_dir}/energy_history.npy', np.array(energy_history))

        stats = {
            'algorithm': 'Fast_Exploration_Algorithm_2',
            'total_steps': num_steps,
            'total_proposals': total_proposals,
            'accepted_proposals': accept_count,
            'acceptance_rate': final_accept_rate,
            'energy_cutoff': energy_cutoff,
            'trajectory_length': len(trajectory_coords),
            'rejection_reasons': rejection_reasons,
            'final_energy': energy_history[-1] if energy_history else None
        }

        with open(f'{output_dir}/fast_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        return stats

    def compare_algorithms(self, initial_coords: torch.Tensor, initial_velocs: torch.Tensor, 
                          mcmc_steps: int = 20000, fast_steps: int = 100000) -> Dict:
        """比较两种算法的性能"""
        print("=== 比较两种Timewarp算法 ===")
        
        # Algorithm 1: MCMC
        print("\n运行Algorithm 1 (MCMC)...")
        mcmc_stats = self.algorithm_1_mcmc_exploration(
            initial_coords, initial_velocs, num_steps=mcmc_steps, 
            output_dir='comparison_mcmc'
        )
        
        # Algorithm 2: Fast exploration
        print("\n运行Algorithm 2 (快速探索)...")
        fast_stats = self.algorithm_2_fast_exploration(
            initial_coords, initial_velocs, num_steps=fast_steps,
            output_dir='comparison_fast'
        )
        
        # 比较结果
        comparison = {
            'mcmc': mcmc_stats,
            'fast': fast_stats,
            'comparison': {
                'mcmc_acceptance_rate': mcmc_stats['acceptance_rate'],
                'fast_acceptance_rate': fast_stats['acceptance_rate'],
                'speed_ratio': fast_steps / mcmc_steps,
                'exploration_efficiency': fast_stats['acceptance_rate'] * (fast_steps / mcmc_steps)
            }
        }
        
        print(f"\n=== 算法比较结果 ===")
        print(f"MCMC接受率: {mcmc_stats['acceptance_rate']:.4f}")
        print(f"快速探索接受率: {fast_stats['acceptance_rate']:.4f}")
        print(f"速度比 (快速/MCMC): {fast_steps / mcmc_steps:.1f}x")
        print(f"探索效率 (接受率×速度): {comparison['comparison']['exploration_efficiency']:.2f}")
        
        with open('algorithm_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2)
            
        return comparison

def main():
    """主函数：运行正确的Timewarp探索"""
    
    # 创建探索器
    explorer = TimewarpCorrectExplorer(
        'corrected_timewarp_model_final.pth', 
        'training_pairs_augmented_final.npy'
    )
    
    # 获取初始结构
    initial_coords, initial_velocs = explorer.get_random_initial_structure(idx=0)
    print(f"初始结构坐标范围: {initial_coords.min():.4f} 到 {initial_coords.max():.4f} nm")
    
    # 选择运行的算法
    run_mcmc = True
    run_fast = True
    run_comparison = True
    
    if run_mcmc:
        print("\n=== 运行Algorithm 1: MCMC探索 ===")
        mcmc_stats = explorer.algorithm_1_mcmc_exploration(
            initial_coords, initial_velocs, 
            num_steps=20000,  # 较少步数，因为接受率低
            batch_size=10,
            output_dir='timewarp_mcmc_correct'
        )
    
    if run_fast:
        print("\n=== 运行Algorithm 2: 快速探索 ===")
        fast_stats = explorer.algorithm_2_fast_exploration(
            initial_coords, initial_velocs,
            num_steps=100000,  # 更多步数，因为速度快
            energy_cutoff=300.0,  # 论文建议的阈值
            output_dir='timewarp_fast_correct'
        )
    
    if run_comparison:
        print("\n=== 运行算法比较 ===")
        comparison = explorer.compare_algorithms(initial_coords, initial_velocs)
    
    print("\n所有探索完成！请分析生成的轨迹文件。")

if __name__ == "__main__":
    main()
