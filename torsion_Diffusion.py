import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math

class TorsionDataset(Dataset):
    """Dataset spécialisée pour des Torsions"""
    def __init__(self, torsion_data_path):
        """
        Args:
            torsion_data_path:扭转角数据文件路径 (.npy格式)
        """
        self.torsions = np.load(torsion_data_path)  # shape: (250000, 2)

        # 将角度转换为弧度（如果还不是的话）
        if np.max(np.abs(self.torsions)) > 2 * np.pi:
            self.torsions = np.deg2rad(self.torsions)

        # 转换为torch tensor
        self.torsions = torch.FloatTensor(self.torsions)

    def __len__(self):
        return len(self.torsions)

    def __getitem__(self, idx):
        return self.torsions[idx]

class SimpleTorsionDiffusion(nn.Module):
    """简化的扭转角扩散模型"""
    def __init__(self, hidden_dim=128, num_layers=3):
        super().__init__()

        # 输入维度：2 (phi, psi) + 1 (time embedding)
        input_dim = 2 + 64  # 64 for time embedding

        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, 2))  # 输出2个角度的噪声预测

        self.network = nn.Sequential(*layers)

    def forward(self, x, t):
        """
        Args:
            x: 扭转角 [batch_size, 2]
            t: 时间步 [batch_size, 1]
        """
        # 时间嵌入
        t_embed = self.time_embed(t)

        # 拼接输入
        input_tensor = torch.cat([x, t_embed], dim=-1)

        # 预测噪声
        noise_pred = self.network(input_tensor)

        return noise_pred

class TorsionDiffusionTrainer:
    def __init__(self, model, device='cuda', num_timesteps=1000):
        self.model = model.to(device)
        self.device = device
        self.num_timesteps = num_timesteps

        # 设置噪声调度
        self.setup_noise_schedule()

    def setup_noise_schedule(self):
        """设置噪声调度"""
        # 线性调度
        self.beta = torch.linspace(0.0001, 0.02, self.num_timesteps).to(self.device)
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)

    def add_noise(self, x_0, t, noise=None):
        """在时间步t添加噪声"""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[t])
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod[t])

        # 添加噪声
        x_t = sqrt_alpha_cumprod.view(-1, 1) * x_0 + sqrt_one_minus_alpha_cumprod.view(-1, 1) * noise

        return x_t, noise

    def train_step(self, batch):
        """单步训练"""
        x_0 = batch.to(self.device)
        batch_size = x_0.shape[0]

        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)

        x_t, noise = self.add_noise(x_0, t)


        t_input = t.float().unsqueeze(-1) / self.num_timesteps  # 归一化时间
        noise_pred = self.model(x_t, t_input)

        loss = nn.MSELoss()(noise_pred, noise)
        return loss

    def train(self, dataloader, epochs=100, lr=1e-3):
        """训练循环"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()

                loss = self.train_step(batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    def sample(self, num_samples=1000):
        """从训练好的模型采样新的扭转角"""
        self.model.eval()

        with torch.no_grad():
            x = torch.randn(num_samples, 2, device=self.device)

            for t in reversed(range(self.num_timesteps)):
                t_tensor = torch.full((num_samples,), t, device=self.device)
                t_input = t_tensor.float().unsqueeze(-1) / self.num_timesteps

                noise_pred = self.model(x, t_input)

                if t > 0:
                    alpha_t = self.alpha[t]
                    beta_t = self.beta[t]
                    sqrt_alpha_t = torch.sqrt(alpha_t)

                    x = (x - beta_t / torch.sqrt(1 - self.alpha_cumprod[t]) * noise_pred) / sqrt_alpha_t

                    if t > 1:
                        z = torch.randn_like(x)
                        sigma_t = torch.sqrt(beta_t)
                        x = x + sigma_t * z
                else:
                    x = x - noise_pred

        return x.cpu().numpy()


def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 256
    epochs = 200
    lr = 1e-3
    torsion_data_path = "/content/drive/MyDrive/Resultats/Torch_SMA_DM/dataset/alanine_torsions_from_xtc.npy"

    dataset = TorsionDataset(torsion_data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Dataset size: {len(dataset)}")
    print(f"Data shape: {dataset.torsions.shape}")

    model = SimpleTorsionDiffusion(hidden_dim=128, num_layers=4)
    trainer = TorsionDiffusionTrainer(model, device=device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Training on device: {device}")

    trainer.train(dataloader, epochs=epochs, lr=lr)

    # 保存模型
    torch.save(model.state_dict(), 'torsion_diffusion_model.pth')
    print("Model saved!")

    # 生成样本测试
    print("Generating samples...")
    samples = trainer.sample(num_samples=1000)

    samples_deg = np.rad2deg(samples)

    print(f"Generated samples shape: {samples.shape}")
    print(f"Sample statistics (degrees):")
    print(f"Phi - Mean: {samples_deg[:, 0].mean():.2f}, Std: {samples_deg[:, 0].std():.2f}")
    print(f"Psi - Mean: {samples_deg[:, 1].mean():.2f}, Std: {samples_deg[:, 1].std():.2f}")

    np.save('generated_torsion_samples.npy', samples)
    print("Generated samples saved!")

if __name__ == "__main__":
    main()