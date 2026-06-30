from typing import Dict
import numpy as np
import torchinfo
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.fft
from torch.nn.utils import weight_norm
import math
import tqdm
import os
import json

from ..utils.torch_utility import EarlyStoppingTorch, DataEmbedding, adjust_learning_rate, get_gpu
from ..utils.dataset import ReconstructDataset   

# 1. Định nghĩa lớp kích hoạt Sin custom
class SinActivation(nn.Module):
    def __init__(self, omega=30.0):
        super(SinActivation, self).__init__()
        self.omega = nn.Parameter(torch.tensor(omega))
    def forward(self, x):
        return torch.sin(self.omega*x)
    
class MaskingNetwork(nn.Module):
    def __init__(self, top_k, d_model, win_size, num_experts, channels=1):
        super(MaskingNetwork, self).__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.win_size = win_size
        in_channels = channels*4
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=3, padding=1),
            # SinActivation()
            nn.ReLU()
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=5, padding=2),
            # SinActivation()
            nn.ReLU()
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=3, dilation=2, padding=2),
            # SinActivation()
            nn.ReLU()
        )
        
        self.branch4 = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=3, dilation=4, padding=4),
            # SinActivation()
            nn.ReLU()
        )
        
        self.project = nn.Conv1d(d_model * 4, num_experts*channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        
    def stft_multi_win(self, x):
        # Đầu vào: x [B, win_size, channels]
        B, win_size, channels = x.size()
        x = x.permute(0,2,1)                                                        # [B, C, win_size]
        
        n_fft_ls = [win_size, win_size // 2, win_size // 4]
        hop_length = 1
        Z = [x]
        x = x.contiguous().view(B*channels, win_size)                                            # [B*C, win_size]
        for n_fft in n_fft_ls:
            # 1. Tính số lượng tần số F của khung này
            F = n_fft // 2 + 1
            
            # 2. Thực hiện STFT -> z có shape: [B*C, F, T]
            current_window = torch.ones(n_fft, device=x.device)
            z = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, 
                            window=None, center=True, return_complex=True)
            
            # 3. Tính biên độ để lấy top K (Đảm bảo K không vượt quá số tần số F)
            a = torch.abs(z)
            current_k = min(self.top_k, F) 
            _, top_k_indices = torch.topk(a, k=current_k, dim=1) # shape: [B*C, current_k, T]
            
            # 4. Trích xuất số phức Top K và dựng lại ma trận lọc nhiễu
            top_k_z = torch.gather(z, dim=1, index=top_k_indices)
            filtered_stft = torch.zeros_like(z, device=z.device)
            filtered_stft.scatter_(dim=1, index=top_k_indices, src=top_k_z)
            
            # 5. Biến đổi ngược ISTFT
            # CỰC KỲ QUAN TRỌNG: Thêm length=win_size để ép đầu ra các vòng lặp luôn bằng nhau
            recon_x = torch.istft(filtered_stft, n_fft=n_fft, hop_length=hop_length, 
                                win_length=n_fft, window=None, center=True, 
                                return_complex=False, length=win_size) # shape luôn là: [B*C, win_size]
            recon_x = recon_x.contiguous().view(B, channels, win_size)                # [B, C, win_size]
            Z.append(recon_x)
            
        # Cách 1: Nếu muốn giữ nguyên các chiều đặc trưng độc lập độc lập -> shape: [B, win_size, len(n_fft_ls)+1]
        Z = torch.cat(Z, dim=1)                                   # [B, C*(len(n_fft_ls)+1), win_size] 
        
        # Cách 2: Nếu muốn nối phẳng các đặc trưng lại với nhau thành 2 chiều -> shape: [B, win_size * 3]
        # Z = torch.cat(Z, dim=-1) 
        
        return Z

    def forward(self, x):                               # [B, win_size, C]
        B, win_size, C = x.size()
        x = self.stft_multi_win(x)                      # [B, C*(k+1), win_size]
        x = 
        # x1 = self.branch1(x)                            # [B, d_model, win_size]
        # x2 = self.branch2(x)                            # [B, d_model, win_size]
        # x3 = self.branch3(x)                            # [B, d_model, win_size]
        # x4 = self.branch4(x)                            # [B, d_model, win_size]
        
        # x = torch.cat([x1,x2,x3,x4], dim=1)             # [B, 4*d_model, win_size]
        # x = self.project(x)                             # [B, num_experts*C, win_size]
        # x = x.view(B,self.num_experts,C,win_size)       # [B, num_experts, C, win_size]
        # x = self.softmax(x)                             # [B, num_experts, C, win_size]
        
        return x
        
class Model(nn.Module):
    def __init__(self, win_size, d_model, top_k=2, channels = 1, num_experts = 4):
        super(Model, self).__init__()
        self.num_subsequences = num_experts
        self.win_size = win_size
        
        # Định nghĩa bộ tạo mặt nạ (decomposition)
        self.soft_masking = MaskingNetwork(top_k, d_model, win_size, num_experts=self.num_subsequences, channels = channels)
        # TÍNH TOÁN NÚT THẮT CỔ CHAI THỰC SỰ
        # Đảm bảo tổng dung lượng (bottleneck_dim * num_experts) chỉ bằng win_size // 2
        bottleneck_dim = win_size // (self.num_subsequences * 2) 
        bottleneck_dim = max(1, bottleneck_dim) # Đảm bảo ít nhất là 1 chiều
        
        self.compressor = nn.ModuleList(
            [nn.Sequential(
                # nn.Conv1d(in_channels=channels, out_channels = 2, kernel_size=4, stride=4, padding=0),
                nn.Linear(win_size, bottleneck_dim))\
                # nn.GELU()) \
                # nn.Conv1d(in_channels=2, out_channels=4, kernel_size=4, stride=4, padding=0),
                # nn.GELU()) \
             for _ in range(self.num_subsequences)])
        
        self.decompressor = nn.ModuleList(
            [nn.Sequential( nn.Linear(bottleneck_dim, win_size)) \
                # nn.ConvTranspose1d(in_channels=4, out_channels=2, kernel_size=4, stride=4, padding=0),
                # nn.GELU(),
                # nn.ConvTranspose1d(in_channels=2, out_channels=channels, kernel_size=4, stride=4, padding=0, bias = False)) \
                for _ in range(self.num_subsequences)])
        
    def forward(self, x): 
        epsi = 1e-5
        B, win_size, C = x.size()
        # Giả sử x đầu vào có dạng [B, win_size]
        x = x/(torch.sqrt((x**2).sum(dim=1, keepdim=True))+epsi)            # [B, win_size, C]
        x_norm = x.clone()                                                  # [B, win_size, C]
        
        # Tính toán mặt nạ mềm
        # soft_mask = self.soft_masking(x)                                    # [B, num_subsequences, C, win_size]
        # x = x.unsqueeze(1).permute(0,1,3,2)                                 # [B, 1, C, win_size]
        # x_masked = x * soft_mask                                            # [B, num_subsequences, C, win_size]
        x_masked = x.permute(0,2,1)                                         # [B, C, win_size]
        
        # Ép qua bộ nén và bộ giải nén dung lượng thấp
        dec_x = []
        for i in range(self.num_subsequences):
            enc_x = self.compressor[i](x_masked[:, i, : ,:])                # [B, d_model, win_size//4]       
            dec_x.append(self.decompressor[i](enc_x))                       # [B, C, win_size]
        dec_x = torch.stack(dec_x, dim=1)                                   # [B, num_subsequences, C, win_size]
        
        # Tổng hợp tuyến tính (Cộng đại số không học tham số)
        x_out = dec_x.sum(dim=1)                                            # [B, C, win_size]
        x_out = x_out.permute(0,2,1)                                        # [B, win_size, C]
        
        return x_norm, dec_x, x_out
    
class PureLoss(nn.Module):
    def __init__(self, lambda_pure=0.1):
        super(PureLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.lambda_pure = lambda_pure
    
    def forward(self, batch_x_norm, batch_dec_x, batch_x_out): 
        eps = 1e-5
        # 1. Loss tái tạo 
        recon_loss = self.mse(batch_x_norm, batch_x_out)
        
        # 2. Tính ma trận Cosine tương quan bình phương
        batch_dec_x_norm = batch_dec_x / torch.sqrt((batch_dec_x ** 2).sum(dim=-1, keepdim=True) + eps)     # [B, num_experts, C, win_size]
        batch_dec_x_norm = batch_dec_x_norm.permute(0,2,1,3)                                                # [B, C, num_experts, win_size]
        cos_sim_matrix = batch_dec_x_norm @ batch_dec_x_norm.permute(0, 1, 3, 2)                            # [B, C, num_experts, num_experts]
        cos_sim_matrix_sq = cos_sim_matrix ** 2
        
        # 3. Xóa đường chéo chính
        num_sub = batch_dec_x.size(1)
        diagonal_mask = torch.eye(num_sub, device=batch_dec_x.device).unsqueeze(0)
        cos_sim_matrix_sq = cos_sim_matrix_sq * (1 - diagonal_mask)

        # pure loss trung bình trên các kênh của các mẫu trong 1 batch
        num_pairs = num_sub * (num_sub - 1)
        pure_loss = cos_sim_matrix_sq.sum() / (batch_x_norm.size(0) * batch_x_norm.size(-1) * num_pairs)
        
        # 4. Cộng tổng hợp có trọng số
        total_loss = recon_loss + self.lambda_pure*pure_loss
        
        return total_loss, recon_loss, pure_loss

class PDE():
    '''
    PDE - Pure Decomposition Expert
    '''
    def __init__(self, 
                 win_size=96,
                 d_model =32,
                 channels=1,
                 top_k=2,
                 num_experts=4,
                 epochs=10,
                 batch_size=128,
                 lr=1e-4,
                 patience=3,
                 lradj="type1",
                 validation_size=0.2):
        super().__init__()

        self.win_size = win_size
        self.top_k = top_k
        self.num_experts = num_experts
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.epochs = epochs
        self.lradj = lradj
        self.validation_size = validation_size

        self.__anomaly_score = None
        
        cuda = True
        self.y_hats = None
        
        self.cuda = cuda
        self.device = get_gpu(self.cuda)
            
        self.model = Model(win_size, d_model, top_k, channels = channels, num_experts = num_experts).float().to(self.device)
        self.model_optim = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = PureLoss()
        
        self.early_stopping = EarlyStoppingTorch(None, patience=self.patience)
        
        self.input_shape = (self.batch_size, self.win_size, channels)
        self.save_path = "/kaggle/working/logs"
        os.makedirs(self.save_path, exist_ok=True)

    def fit(self, data):
        tsTrain = data[:int((1-self.validation_size)*len(data))]
        tsValid = data[int((1-self.validation_size)*len(data)):]

        train_loader = DataLoader(
            dataset=ReconstructDataset(tsTrain, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        valid_loader = DataLoader(
            dataset=ReconstructDataset(tsValid, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        train_steps = len(train_loader)
        for epoch in range(1, self.epochs + 1):
            ## Training
            train_loss = 0
            train_recon_loss = 0
            train_pure_loss = 0
            self.model.train()
            
            loop = tqdm.tqdm(enumerate(train_loader),total=len(train_loader),leave=True)
            for i, (batch_x, _) in loop:
                self.model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                out = self.model(batch_x)
                x_norm, dec_x, x_recon = out
                loss, recon_loss, pure_loss = self.criterion(x_norm, dec_x, x_recon)
                loss.backward()
                self.model_optim.step()
                
                train_loss += loss.cpu().item()
                train_recon_loss += recon_loss.cpu().item()
                train_pure_loss += pure_loss.cpu().item()
                
                loop.set_description(f'Training Epoch [{epoch}/{self.epochs}]')
                loop.set_postfix(loss=loss.item(), avg_loss=train_loss/(i+1), avg_recon_loss=train_recon_loss/(i+1), avg_pure_loss=train_pure_loss/(i+1))
            
            ## Validation
            self.model.eval()
            total_loss = []
            
            loop = tqdm.tqdm(enumerate(valid_loader),total=len(valid_loader),leave=True)
            with torch.no_grad():
                for i, (batch_x, _) in loop:
                    batch_x = batch_x.float().to(self.device)

                    x_norm, dec_x, outputs = self.model(batch_x)

                    # if len(outputs.size()) == 2: outputs = outputs.unsqueeze(-1)
                    # outputs = outputs[:, :, f_dim:]
                    pred = outputs.detach().cpu()
                    true = x_norm.detach().cpu()
                    dec_x = dec_x.detach().cpu()

                    loss, recon_loss, pure_loss = self.criterion(true, dec_x, pred)
                    total_loss.append(loss.item())
                    loop.set_description(f'Valid Epoch [{epoch}/{self.epochs}]')
                    
            valid_loss = np.average(total_loss)
            loop.set_postfix(loss=loss.item(), valid_loss=valid_loss)
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                print("   Early stopping<<<")
                break
            
            adjust_learning_rate(self.model_optim, epoch + 1, self.lradj, self.lr)
    def decision_function(self, data):
        test_loader = DataLoader(
            dataset=ReconstructDataset(data, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduction='none')
        
        # 1. Khởi tạo các mảng Global để chứa dữ liệu cộng dồn
        N = len(data)
        C = data.shape[-1] if data.ndim > 1 else 1 # Số kênh (channels)
        
        full_scores = np.zeros(N)
        full_counts = np.zeros(N)
        
        # Lưu lại để debug
        full_recon = np.zeros((N, C))
        full_true = np.zeros((N, C))
        full_experts = np.zeros((N, self.num_experts, C))
        
        global_idx = 0
        loop = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), leave=True)
        
        with torch.no_grad():
            for i, (batch_x, _) in loop:
                batch_x = batch_x.float().to(self.device)
                if batch_x.dim() == 2:
                    batch_x = batch_x.unsqueeze(-1)
                
                B = batch_x.size(0)
                
                # Reconstruction
                x_norm, dec_x, outputs = self.model(batch_x)
                
                # Tính score theo từng điểm (Point-wise score)
                # Kích thước: [B, win_size] (đã trung bình qua các kênh)
                point_scores = torch.mean(self.anomaly_criterion(x_norm, outputs), dim=-1).cpu().numpy()
                
                # Ép kiểu và đưa về CPU
                x_norm_np = x_norm.cpu().numpy()                       # [B, win_size, C]
                outputs_np = outputs.cpu().numpy()                     # [B, win_size, C]
                dec_x_np = dec_x.permute(0, 3, 1, 2).cpu().numpy()     # ĐÚNG shape [B, win_size, num_experts, C]     
                
                # 2. Xử lý Overlap: Cộng dồn vào mảng Global
                for b in range(B):
                    start = global_idx + b
                    end = start + self.win_size
                    
                    full_scores[start:end] += point_scores[b]
                    full_true[start:end] += x_norm_np[b]
                    full_recon[start:end] += outputs_np[b]
                    full_experts[start:end] += dec_x_np[b]
                    full_counts[start:end] += 1
                    
                global_idx += B
                loop.set_description(f'Testing Phase: ')

        # Tránh chia cho 0 ở những điểm không được cover (nếu có)
        full_counts[full_counts == 0] = 1
        
        # 3. Tính trung bình (Averaging) để khử nhiễu và làm mượt
        full_scores = full_scores / full_counts
        full_true = full_true / full_counts[:, None]
        full_recon = full_recon / full_counts[:, None]
        full_experts = full_experts / full_counts[:, None, None]
        
        self.__anomaly_score = full_scores
        
        # Đóng gói dữ liệu debug
        self.debug_data = {
            "scores": full_scores,
            "true_seq": full_true,
            "recon_seq": full_recon,
            "expert_seqs": full_experts
        }
        for key, value in self.debug_data.items():
            np.save(os.path.join(self.save_path, f"{key}.npy"), value)
        
        return self.__anomaly_score
    # def decision_function(self, data):
    #     test_loader = DataLoader(
    #         dataset=ReconstructDataset(data, window_size=self.win_size),
    #         batch_size=self.batch_size,
    #         shuffle=False
    #     )
        
    #     self.model.eval()
    #     attens_energy = []
    #     y_hats = []
    #     self.anomaly_criterion = nn.MSELoss(reduction='none')
        
    #     loop = tqdm.tqdm(enumerate(test_loader),total=len(test_loader),leave=True)
    #     expert_out = []
    #     with torch.no_grad():
    #         for i, (batch_x, _) in loop:
    #             batch_x = batch_x.float().to(self.device)
    #             # reconstruction
    #             x_norm, dec_x, outputs = self.model(batch_x)
    #             expert_out.append(dec_x.permute(0,2,1))                                # dec_x: [B, num_experts, C, win_size]
    #             # criterion
    #             score = torch.mean(self.anomaly_criterion(x_norm, outputs), dim=-1)    # [B, win_size]
    #             y_hat = torch.squeeze(outputs, -1)

    #             score = score.mean(dim=1).detach().cpu().numpy()
    #             y_hat = y_hat.detach().cpu().numpy()[:, self.win_size // 2]
                
    #             attens_energy.append(score)
    #             y_hats.append(y_hat)
    #             loop.set_description(f'Testing Phase: ')

    #     attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    #     scores = np.array(attens_energy)
        
    #     y_hats = np.concatenate(y_hats, axis=0).reshape(-1)
    #     y_hats = np.array(y_hats)

    #     assert scores.ndim == 1
        
    #     import shutil
    #     self.save_path = None
    #     if self.save_path and os.path.exists(self.save_path):
    #         shutil.rmtree(self.save_path)
            
    #     self.__anomaly_score = scores
    #     self.y_hats = y_hats

    #     if self.__anomaly_score.shape[0] < len(data):
    #         self.__anomaly_score = np.array([self.__anomaly_score[0]]*math.ceil((self.win_size-1)/2) + 
    #                     list(self.__anomaly_score) + [self.__anomaly_score[-1]]*((self.win_size-1)//2))
        
    #     return self.__anomaly_score

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score
    
    def get_y_hat(self) -> np.ndarray:
        return self.y_hats
    
    def param_statistic(self, save_file):
        model_stats = torchinfo.summary(self.model, self.input_shape, verbose=0)
        with open(save_file, 'w') as f:
            f.write(str(model_stats))
