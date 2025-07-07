import torch
import numpy as np
import matplotlib.pyplot as plt
import openmm as mm
import openmm.app as app
from openmm import unit
import os
from tqdm import tqdm

%cd /content/drive/MyDrive/Resultats/Timewarp

class TimewarpExplorer:
    """Timewarp模型探索器"""

    def __init__(self, model_path, training_data_path='training_pairs_augmented_1.npy', device='cuda' if torch.cuda.is_available() else 'cpu'):
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

        # 能量函数（可选）
        self.energy_fn = None
        self.setup_energy_function()

    def get_random_initial_structure(self):
        """从训练数据中随机选择一个初始结构"""
        idx = np.random.randint(0, len(self.all_coords_nm))
        initial_coords = torch.FloatTensor(self.all_coords_nm[idx:idx+1])  # [1, 22, 3] nm
        initial_velocs = torch.FloatTensor(self.all_velocs_nm_ps[idx:idx+1])  # [1, 22, 3] nm/ps
        return initial_coords, initial_velocs

    def setup_energy_function(self):
        """设置OpenMM能量函数（可选的能量筛选）"""
        try:
            # 加载alanine-dipeptide的拓扑和力场（隐式溶剂）
            pdb = app.PDBFile('alanine-dipeptide-solvated.pdb')  # 你的PDB路径
            forcefield = app.ForceField('amber14-all.xml', 'implicit/gbn2.xml')

            # 创建系统（隐式溶剂）
            system = forcefield.createSystem(
                pdb.topology,
                nonbondedMethod=app.NoCutoff,
                constraints=app.HBonds
            )

            # 创建context
            integrator = mm.VerletIntegrator(0.002*unit.picoseconds)
            self.context = mm.Context(system, integrator)

            # 计算参考能量
            pdb_coords = pdb.positions.value_in_unit(unit.nanometer)
            self.context.setPositions(pdb_coords)
            state = self.context.getState(getEnergy=True)
            self.reference_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            print(f"参考能量: {self.reference_energy:.2f} kJ/mol")

            def energy_function(coords_tensor):
                """计算给定坐标的势能"""
                try:
                    # coords_tensor: [1, 22, 3] in nm
                    coords_np = coords_tensor.cpu().numpy().squeeze(0)  # [22, 3]

                    # 检查坐标合理性
                    if np.any(np.isnan(coords_np)) or np.isinf(coords_np):
                        return float('inf')

                    # 设置坐标
                    self.context.setPositions(coords_np * unit.nanometer)

                    # 计算能量
                    state = self.context.getState(getEnergy=True)
                    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

                    return energy

                except Exception as e:
                    print(f"能量计算失败: {e}")
                    return float('inf')

            self.energy_fn = energy_function
            print("隐式溶剂能量函数设置成功")

        except Exception as e:
            print(f"能量函数设置失败，将跳过能量筛选: {e}")
            self.energy_fn = None
            self.reference_energy = None

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


    def explore(self,
                initial_coords,
                initial_velocs,
                num_steps=10000,
                energy_cutoff_kj=None,   # 大幅降低阈值！
                adaptive_cutoff=False,  # 自适应阈值
                save_interval=100,
                output_dir='exploration_results'):
        """
        使用Timewarp模型进行构象空间探索

        Args:
            initial_coords: [1, 22, 3] 初始坐标 (nm)
            initial_velocs: [1, 22, 3] 初始速度 (nm/ps)
            num_steps: 探索步数
            energy_cutoff_kj: 初始能量筛选阈值 (kJ/mol)
            adaptive_cutoff: 是否使用自适应阈值
            save_interval: 保存间隔
            output_dir: 输出目录
        """

        os.makedirs(output_dir, exist_ok=True)

        # 转换到设备并标准化
        coords = initial_coords.clone().to(self.device)
        velocs = initial_velocs.clone().to(self.device)

        if self.norm_stats:
            coords, velocs = self.normalize_data(coords, velocs)

        # 存储轨迹
        trajectory_coords = []
        trajectory_velocs = []
        energies = []

        # 初始能量
        current_energy = None
        if self.energy_fn:
            initial_coords_denorm, _ = self.denormalize_data(coords, velocs)
            current_energy = self.energy_fn(initial_coords_denorm)
            energies.append(current_energy)
            print(f"初始能量: {current_energy:.2f} kJ/mol")

            # 设置相对于参考能量的阈值
            if hasattr(self, 'reference_energy') and self.reference_energy is not None and energy_cutoff_kj is None:
                energy_cutoff_kj = 20 # Default relative cutoff
                print(f"使用默认相对能量阈值: {energy_cutoff_kj} kJ/mol")


        trajectory_coords.append(coords.cpu().numpy())
        trajectory_velocs.append(velocs.cpu().numpy())

        accepted_steps = 0
        rejected_steps = 0
        recent_energies = []

        print(f"开始探索 {num_steps} 步...")

        with torch.no_grad():
            for step in tqdm(range(num_steps)):
                # 生成proposal
                new_coords, new_velocs, _ = self.model(
                    self.atom_types, coords, velocs, reverse=True
                )

                # 能量筛选（如果有能量函数且设置了阈值）
                accept = True
                if self.energy_fn and current_energy is not None and energy_cutoff_kj is not None:
                    # 反标准化用于能量计算
                    new_coords_denorm, _ = self.denormalize_data(new_coords, new_velocs)

                    try:
                        new_energy = self.energy_fn(new_coords_denorm)

                        # 检查能量是否合理
                        if np.any(np.isnan(new_energy)) or np.isinf(new_energy):
                            accept = False
                            rejected_steps += 1
                        else:
                            delta_energy = new_energy - current_energy

                            # 自适应阈值
                            current_cutoff = energy_cutoff_kj
                            if adaptive_cutoff and len(recent_energies) > 100:
                                # 基于最近能量变化调整阈值
                                recent_std = np.std(recent_energies[-100:])
                                current_cutoff = max(energy_cutoff_kj, recent_std * 2)

                            if delta_energy > current_cutoff:
                                accept = False
                                rejected_steps += 1
                            else:
                                current_energy = new_energy
                                accepted_steps += 1
                                energies.append(new_energy)
                                recent_energies.append(new_energy)

                                # 限制recent_energies长度
                                if len(recent_energies) > 500:
                                    recent_energies = recent_energies[-300:]
                    except Exception as e:
                        # If energy calculation fails, still accept this step but log
                        if step % 1000 == 0:
                            print(f"Energy calculation failed (step {step}): {e}")
                        accept = True
                        accepted_steps += 1
                else:
                    # No energy function or no cutoff set, always accept
                    accepted_steps += 1

                # Accept or reject proposal
                if accept:
                    clip_limit = 0.5
                    coords = torch.clamp(new_coords, -clip_limit, clip_limit)
                    velocs = new_velocs  # 通常速度不用clip，除非你也希望

                    # Save trajectory
                    if step % save_interval == 0:
                        trajectory_coords.append(coords.cpu().numpy())
                        trajectory_velocs.append(velocs.cpu().numpy())

                # Report progress periodically and adjust strategy
                if (step + 1) % 1000 == 0:
                    accept_rate = accepted_steps / (accepted_steps + rejected_steps) if (accepted_steps + rejected_steps) > 0 else 1.0

                    # If acceptance rate is too low, relax the cutoff
                    if accept_rate < 0.1 and adaptive_cutoff and energy_cutoff_kj is not None:
                        energy_cutoff_kj *= 1.5
                        print(f"Step {step+1}/{num_steps}, Acceptance Rate: {accept_rate:.3f}, Adjusting cutoff to: {energy_cutoff_kj:.1f}")
                    elif accept_rate > 0.8 and adaptive_cutoff and energy_cutoff_kj is not None:
                        energy_cutoff_kj *= 0.9
                        print(f"Step {step+1}/{num_steps}, Acceptance Rate: {accept_rate:.3f}, Adjusting cutoff to: {energy_cutoff_kj:.1f}")
                    else:
                        print(f"Step {step+1}/{num_steps}, Acceptance Rate: {accept_rate:.3f}")


        # Save results
        trajectory_coords = np.concatenate(trajectory_coords, axis=0)  # [T, 22, 3]
        trajectory_velocs = np.concatenate(trajectory_velocs, axis=0)  # [T, 22, 3]

        # Denormalize for saving
        if self.norm_stats:
            coords_tensor = torch.FloatTensor(trajectory_coords)
            velocs_tensor = torch.FloatTensor(trajectory_velocs)
            coords_denorm, velocs_denorm = self.denormalize_data(coords_tensor, velocs_tensor)
            trajectory_coords = coords_denorm.numpy()
            trajectory_velocs = velocs_denorm.numpy()


        np.save(f'{output_dir}/exploration_coords.npy', trajectory_coords)
        np.save(f'{output_dir}/exploration_velocs.npy', trajectory_velocs)

        if energies:
            np.save(f'{output_dir}/exploration_energies.npy', np.array(energies))

        # Statistics
        final_accept_rate = accepted_steps / (accepted_steps + rejected_steps) if (accepted_steps + rejected_steps) > 0 else 1.0

        stats = {
            'total_steps': num_steps,
            'accepted_steps': accepted_steps,
            'rejected_steps': rejected_steps,
            'acceptance_rate': final_accept_rate,
            'trajectory_length': len(trajectory_coords),
            'final_energy_cutoff': energy_cutoff_kj,
            'energy_range': [float(min(energies)), float(max(energies))] if energies else None
        }

        print(f"\nExploration completed!")
        print(f"Total steps: {stats['total_steps']}")
        print(f"Accepted steps: {stats['accepted_steps']}")
        print(f"Rejected steps: {stats['rejected_steps']}")
        print(f"Final Acceptance Rate: {stats['acceptance_rate']:.3f}")
        print(f"Trajectory Length: {stats['trajectory_length']}")
        if stats['final_energy_cutoff'] is not None:
            print(f"Final energy cutoff: {stats['final_energy_cutoff']:.1f} kJ/mol")


        # Save statistics
        import json
        with open(f'{output_dir}/exploration_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        return trajectory_coords, trajectory_velocs, energies, stats

    def analyze_exploration(self, coords_file, energies_file=None):
        """Analyze exploration results"""
        coords = np.load(coords_file)  # [T, 22, 3]

        print(f"Trajectory analysis:")
        print(f"  Number of steps: {len(coords)}")
        print(f"  Coordinate range: {coords.min():.3f} to {coords.max():.3f} nm")
        print(f"  Coordinate standard deviation: {coords.std():.3f} nm")

        # Calculate RMSD
        ref_coords = coords[0]
        rmsds = []
        for frame in coords:
            rmsd = np.sqrt(np.mean((frame - ref_coords)**2))
            rmsds.append(rmsd)

        print(f"  RMSD range relative to initial conformation: {min(rmsds):.3f} to {max(rmsds):.3f} nm")


        # If energy data is available
        if energies_file and os.path.exists(energies_file):
            energies = np.load(energies_file)
            print(f"  Energy range: {energies.min():.1f} to {energies.max():.1f} kJ/mol")
            print(f"  Energy standard deviation: {energies.std():.1f} kJ/mol")


        return rmsds

def load_initial_structure(pdb_path=None):
    """Load initial structure"""
    if pdb_path and os.path.exists(pdb_path):
        # Load from PDB file
        pdb = app.PDBFile(pdb_path)
        coords = pdb.positions.value_in_unit(unit.nanometer)
        coords_tensor = torch.FloatTensor(coords).unsqueeze(0)  # [1, N, 3]
    else:
        # Use random initial coordinates if no PDB file
        print("Using random initial coordinates")
        coords_tensor = torch.randn(1, 22, 3) * 0.1  # [1, 22, 3] in nm

    # Initial velocities set to zero or small random values
    velocs_tensor = torch.randn(1, 22, 3) * 0.01  # Small random velocities

    return coords_tensor, velocs_tensor

# Usage example
if __name__ == "__main__":
    # Create explorer
    explorer = TimewarpExplorer('timewarp_model_debug_addition.pth', 'training_pairs_augmented_1.npy')

    explorer.energy_fn = None

    # From training data to get initial structure
    initial_coords, initial_velocs = explorer.get_random_initial_structure()
    print(f"Using initial structure from training data, coordinate range: {initial_coords.min():.4f} to {initial_coords.max():.4f} nm")

    # Start exploration (adjust parameters)
    trajectory_coords, trajectory_velocs, energies, stats = explorer.explore(
        initial_coords=initial_coords,
        initial_velocs=initial_velocs,
        num_steps=10000,
        #energy_cutoff_kj=20,      # Reduced cutoff from 300 to 20!
        #adaptive_cutoff=True,     # Enable adaptive cutoff
        save_interval=50,
        output_dir='timewarp_exploration'
    )

    # Analyze results
    rmsds = explorer.analyze_exploration(
        'timewarp_exploration/exploration_coords.npy',
        'timewarp_exploration/exploration_energies.npy'
    )

    print("Exploration completed! Results saved in timewarp_exploration/ directory")