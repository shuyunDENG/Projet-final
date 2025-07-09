import torch
import numpy as np
import matplotlib.pyplot as plt
import openmm as mm
import openmm.app as app
from openmm import unit
import os
from tqdm import tqdm


class TimewarpExplorer:
    """Timewarp模型探索器"""

    def __init__(self, model_path, training_data_path='training_pairs_augmented_final.npy', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

        # 加载模型
        print("正在加载Timewarp模型...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.config = checkpoint['config']
        self.model = create_timewarp_model(self.config).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # 标准化参数（如果有的话）
        self.norm_stats = checkpoint.get('normalization_stats', None)

        print(f"模型加载完成，设备: {device}")
        print(f"模型配置: {self.config}")

        # 诊断标准化参数
        if self.norm_stats:
            pos_mean, pos_std, vel_mean, vel_std = self.norm_stats
            print(f"标准化参数检查:")
            print(f"  pos_std min/max: {pos_std.min():.6f} / {pos_std.max():.6f}")
            print(f"  pos_mean range: {pos_mean.min():.6f} to {pos_mean.max():.6f}")

            if pos_std.min() < 0.001:
                print("警告：标准差过小，可能导致坐标范围被严重压缩！")

        # 加载训练数据并进行单位转换
        print("正在加载和处理训练数据...")
        data = np.load(training_data_path)
        print(f"训练数据形状: {data.shape}")

        # 提取所有坐标和速度
        all_coords = data[:, :, :, :3].reshape(-1, 22, 3)  # [所有时间点, 22原子, 3坐标] 单位：埃
        all_velocs = data[:, :, :, 3:].reshape(-1, 22, 3)  # [所有时间点, 22原子, 3速度] 单位：埃/ps

        # 单位转换：埃 → 纳米
        self.all_coords_nm = all_coords / 10.0  # 埃 → 纳米
        self.all_velocs_nm_ps = all_velocs / 10.0  # 埃/ps → nm/ps

        print(f"原始坐标范围: {all_coords.min():.3f} 到 {all_coords.max():.3f} Å")
        print(f"转换后坐标范围: {self.all_coords_nm.min():.3f} 到 {self.all_coords_nm.max():.3f} nm")
        print(f"原始速度范围: {all_velocs.min():.3f} 到 {all_velocs.max():.3f} Å/ps")
        print(f"转换后速度范围: {self.all_velocs_nm_ps.min():.3f} 到 {self.all_velocs_nm_ps.max():.3f} nm/ps")

        # alanine-dipeptide的原子类型
        self.atom_types = torch.tensor([0, 1, 0, 2, 3, 3, 3, 0, 3, 0, 2, 1, 3, 0, 0, 2, 3, 3, 3, 3, 3, 3],
                                     dtype=torch.long, device=device).unsqueeze(0)  # [1, 22]

    def get_random_initial_structure(self, idx=None):
        """从训练数据中随机选择一个初始结构"""
        if idx is None:
            idx = np.random.randint(0, len(self.all_coords_nm))
        initial_coords = torch.FloatTensor(self.all_coords_nm[idx:idx+1])  # [1, 22, 3] nm
        initial_velocs = torch.FloatTensor(self.all_velocs_nm_ps[idx:idx+1])  # [1, 22, 3] nm/ps
        return initial_coords, initial_velocs

    def denormalize_data(self, coords, velocs):
        """反标准化数据"""
        if self.norm_stats is None:
            return coords, velocs

        pos_mean, pos_std, vel_mean, vel_std = self.norm_stats

        coords_denorm = coords * pos_std + pos_mean
        velocs_denorm = velocs * vel_std + vel_mean

        return coords_denorm, velocs_denorm

    def normalize_data(self, coords, velocs):
        """标准化数据"""
        if self.norm_stats is None:
            return coords, velocs

        pos_mean, pos_std, vel_mean, vel_std = self.norm_stats

        coords_norm = (coords - pos_mean) / pos_std
        velocs_norm = (velocs - vel_mean) / vel_std

        return coords_norm, velocs_norm

    def explore_with_noise(self,
                          initial_coords,
                          initial_velocs,
                          num_steps=10000,
                          noise_scale=0.05,
                          adaptive_noise=True,
                          save_interval=100,
                          output_dir='exploration_results_noise'):
        """
        带噪声的探索方法，无能量筛选

        Args:
            initial_coords: [1, 22, 3] 初始坐标 (nm)
            initial_velocs: [1, 22, 3] 初始速度 (nm/ps)
            num_steps: 探索步数
            noise_scale: 噪声强度
            adaptive_noise: 是否使用自适应噪声
            save_interval: 保存间隔
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)

        # 转换到设备并标准化
        coords = initial_coords.clone().to(self.device)
        velocs = initial_velocs.clone().to(self.device)

        if self.norm_stats:
            coords, velocs = self.normalize_data(coords, velocs)
            print(f"标准化后初始坐标范围: {coords.min():.4f} to {coords.max():.4f}")

        trajectory_coords = [coords.cpu().numpy()]
        trajectory_velocs = [velocs.cpu().numpy()]

        print(f"开始带噪声探索 {num_steps} 步，初始噪声scale={noise_scale}...")

        with torch.no_grad():
            for step in tqdm(range(num_steps)):
                # 自适应噪声调整
                current_noise_scale = noise_scale
                if adaptive_noise:
                    if step < 2000:
                        current_noise_scale = noise_scale * 2.0  # 前期增大噪声
                    elif step > 7000:
                        current_noise_scale = noise_scale * 0.5  # 后期减小噪声

                # 添加噪声增加多样性
                noise_coords = coords + torch.randn_like(coords) * current_noise_scale
                noise_velocs = velocs + torch.randn_like(velocs) * current_noise_scale

                # 生成proposal
                new_coords, _ = self.model(
                    self.atom_types, noise_coords, noise_velocs, reverse=True
                )

                # 直接接受，无任何筛选或裁剪
                coords = new_coords
                velocs = new_velocs

                # 保存轨迹
                if step % save_interval == 0:
                    trajectory_coords.append(coords.cpu().numpy())
                    trajectory_velocs.append(velocs.cpu().numpy())

                # 进度报告
                if (step + 1) % 2000 == 0:
                    coord_range = f"{coords.min():.3f} to {coords.max():.3f}"
                    print(f"Step {step+1}/{num_steps}, coords range: {coord_range}, noise_scale: {current_noise_scale:.4f}")

        # 保存结果
        trajectory_coords = np.concatenate(trajectory_coords, axis=0)
        trajectory_velocs = np.concatenate(trajectory_velocs, axis=0)

        print(f"标准化空间轨迹坐标范围: {trajectory_coords.min():.4f} to {trajectory_coords.max():.4f}")

        # 反标准化用于保存
        if self.norm_stats:
            coords_tensor = torch.FloatTensor(trajectory_coords)
            velocs_tensor = torch.FloatTensor(trajectory_velocs)
            coords_denorm, velocs_denorm = self.denormalize_data(coords_tensor, velocs_tensor)
            trajectory_coords = coords_denorm.numpy()
            trajectory_velocs = velocs_denorm.numpy()
            print(f"反标准化后轨迹坐标范围: {trajectory_coords.min():.4f} to {trajectory_coords.max():.4f}")

        np.save(f'{output_dir}/exploration_coords.npy', trajectory_coords)
        np.save(f'{output_dir}/exploration_velocs.npy', trajectory_velocs)

        stats = {
            'total_steps': num_steps,
            'trajectory_length': len(trajectory_coords),
            'noise_scale': noise_scale,
            'adaptive_noise': adaptive_noise,
            'method': 'noise_enhanced',
            'coord_range': [float(trajectory_coords.min()), float(trajectory_coords.max())],
            'coord_std': float(trajectory_coords.std())
        }

        print(f"\n噪声探索完成!")
        print(f"总步数: {stats['total_steps']}")
        print(f"轨迹长度: {stats['trajectory_length']}")
        print(f"坐标范围: {stats['coord_range'][0]:.4f} to {stats['coord_range'][1]:.4f} nm")
        print(f"坐标标准差: {stats['coord_std']:.4f} nm")

        import json
        with open(f'{output_dir}/exploration_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        return trajectory_coords, trajectory_velocs, stats

    def explore_multiple_noise_levels(self, initial_coords, initial_velocs, base_output_dir='noise_exploration'):
        """测试多个噪声级别"""
        noise_scales = [0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0]

        results = {}
        for noise_scale in noise_scales:
            print(f"\n=== 测试噪声级别: {noise_scale} ===")
            output_dir = f'{base_output_dir}_scale_{noise_scale}'

            trajectory_coords, trajectory_velocs, stats = self.explore_with_noise(
                initial_coords=initial_coords,
                initial_velocs=initial_velocs,
                num_steps=8000,
                noise_scale=noise_scale,
                adaptive_noise=True,
                save_interval=50,
                output_dir=output_dir
            )

            results[noise_scale] = {
                'output_dir': output_dir,
                'stats': stats,
                'coords_range': stats['coord_range'],
                'coords_std': stats['coord_std']
            }

        # 打印对比结果
        print(f"\n=== 噪声级别对比结果 ===")
        for noise_scale, result in results.items():
            print(f"Noise {noise_scale}: 坐标范围 {result['coords_range'][0]:.4f} to {result['coords_range'][1]:.4f}, "
                  f"标准差 {result['coords_std']:.4f}")

        return results

    def analyze_exploration(self, coords_file, energies_file=None):
        """分析探索结果"""
        coords = np.load(coords_file)  # [T, 22, 3]

        print(f"轨迹分析:")
        print(f"  帧数: {len(coords)}")
        print(f"  坐标范围: {coords.min():.3f} to {coords.max():.3f} nm")
        print(f"  坐标标准差: {coords.std():.3f} nm")

        # 计算RMSD
        ref_coords = coords[0]
        rmsds = []
        for frame in coords:
            rmsd = np.sqrt(np.mean((frame - ref_coords)**2))
            rmsds.append(rmsd)

        print(f"  相对初始构象的RMSD范围: {min(rmsds):.3f} to {max(rmsds):.3f} nm")
        print(f"  平均RMSD: {np.mean(rmsds):.3f} nm")

        return rmsds

def load_initial_structure(pdb_path=None):
    """加载初始结构"""
    if pdb_path and os.path.exists(pdb_path):
        # 从PDB文件加载
        pdb = app.PDBFile(pdb_path)
        coords = pdb.positions.value_in_unit(unit.nanometer)
        coords_tensor = torch.FloatTensor(coords).unsqueeze(0)  # [1, N, 3]
    else:
        # 使用随机初始坐标
        print("使用随机初始坐标")
        coords_tensor = torch.randn(1, 22, 3) * 0.1  # [1, 22, 3] in nm

    # 初始速度设为零或小随机值
    velocs_tensor = torch.randn(1, 22, 3) * 0.01  # 小随机速度

    return coords_tensor, velocs_tensor

# 使用示例
if __name__ == "__main__":
    # 创建探索器
    explorer = TimewarpExplorer('corrected_timewarp_model_final.pth', 'training_pairs_augmented_final.npy')

    # 从训练数据获取初始结构
    initial_coords, initial_velocs = explorer.get_random_initial_structure(idx=0)
    print(f"使用训练数据idx=0作为初始结构，坐标范围: {initial_coords.min():.4f} to {initial_coords.max():.4f} nm")

    # 方案1: 测试单个噪声级别
    print("\n=== 方案1: 单个噪声级别测试 ===")
    trajectory_coords, trajectory_velocs, stats = explorer.explore_with_noise(
        initial_coords=initial_coords,
        initial_velocs=initial_velocs,
        num_steps=100000,
        noise_scale=0.05,
        adaptive_noise=True,
        save_interval=50,
        output_dir='timewarp_noise_exploration'
    )

    # 分析结果
    rmsds = explorer.analyze_exploration('timewarp_noise_exploration/exploration_coords.npy')

    # 方案2: 测试多个噪声级别
    print("\n=== 方案2: 多噪声级别对比 ===")
    results = explorer.explore_multiple_noise_levels(initial_coords, initial_velocs)

    print("\n探索完成! 请检查生成的文件并进行Ramachandran plot分析")
