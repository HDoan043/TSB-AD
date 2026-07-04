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

class WaveExpert(nn.Module):
    def __init__(self, win_size, channels = 1):
        super(WaveExpert, self).__init__() 
        self.win_size = win_size
        
        self.fft_len = win_size // 2 + 1  
        self.range_size = win_size // 2
        
        self.compress_f = nn.Sequential(
            nn.Linear(self.fft_len, 1, bias = True),
            nn.Sigmoid()
        )
        
        self.compress_amplitude = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=2, kernel_size=self.fft_len),                            
            nn.Tanh()
        )

    def forward(self, x, local_timestamps=None):
        # x: [B, C, win_size]
        B, C, win_size = x.size()
        
        fft_x = torch.fft.rfft(x, dim=-1)                # [B, C, fft_len]
        magnitude = torch.abs(fft_x)                     # [B, C, fft_len]
        real = fft_x.real                                # [B, C, fft_len]
        imag = fft_x.imag                                # [B, C, fft_len]

        # 1. Trích xuất tần số f (số chu kỳ trong cửa sổ)
        f = self.compress_f(magnitude) * self.range_size # [B, C, 1]
        
        # 2. Trích xuất biên độ R và pha I
        amplitude = torch.stack([real, imag], dim=2)     # [B, C, 2, fft_len]
        
        # SỬA LỖI: view thành 2 kênh và chiều dài fft_len
        amplitude = amplitude.contiguous().view(B*C, 2, self.fft_len) # [B*C, 2, fft_len]
        amplitude = self.compress_amplitude(amplitude)   # [B*C, 2, 1]
        amplitude = amplitude.contiguous().view(B, C, 2) # [B, C, 2]

        # 3. Quản lý Timestamps (Bắt buộc dùng Local Time: 0 -> win_size - 1)
        if local_timestamps is None:
            # Tự động tạo nếu không truyền vào
            local_timestamps = torch.arange(win_size, dtype=x.dtype, device=x.device)
            local_timestamps = local_timestamps.view(1, 1, win_size).expand(B, C, -1)
        else:
            # An toàn cho mọi Batch Size: Đưa về chuẩn [B, C, win_size]
            local_timestamps = local_timestamps.view(B, 1, win_size).expand(-1, C, -1)
            
        # 4. CHUẨN HÓA VẬT LÝ TÍN HIỆU
        # Theta = 2 * pi * f * (t / win_size)
        theta = 2 * math.pi * f * (local_timestamps / win_size) # [B, C, win_size]

        # 5. Khôi phục
        R = amplitude[:, :, :1] # [B, C, 1]
        I = amplitude[:, :, 1:] # [B, C, 1]
        
        recon_x = R * torch.cos(theta) + I * torch.sin(theta)   # [B, C, win_size]
        return recon_x

class GlobalExpert(nn.Module):
    def __init__(self, win_size, channels=1):
        super(GlobalExpert, self).__init__()
        self.compressor = nn.Sequential(
            nn.Linear(win_size, win_size//2),
            nn.LeakyReLU(0.1),
            nn.Linear(win_size//2, win_size//4)
        )
        self.reconstructor = nn.Sequential(
            nn.Linear(win_size//4, win_size)
        )
    def forward(self, x):
        # x: [B, C, win_size]
        h = self.compressor(x)                # [B, C, win_size//4]
        x = self.reconstructor(h)              # [B, C, win_size]
        return x

class LocalExpert(nn.Module):
    def __init__(self, d_model=32):
        super(LocalExpert, self).__init__()
        kernel_size = 5
        scale_factor= 8
        # BỘ NÉN: Dùng Stride để ép giảm độ phân giải thời gian (Tạo Nút thắt)
        self.compressor = nn.Sequential(
            # Conv1d bắt đặc trưng cục bộ
            nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=kernel_size, padding=kernel_size//2),
            # nn.GELU(), # Ở mảng cục bộ này có thể dùng phi tuyến để bắt noise tốt hơn
            # AvgPool với stride=scale_factor sẽ nén chiều dài chuỗi đi 4 lần (VD: 96 -> 24)
            # Nó pha loãng hoàn toàn các gai nhọn bất thường.
            nn.AvgPool1d(kernel_size=scale_factor, stride=scale_factor) 
        )
        
        # BỘ KHÔI PHỤC: Dùng Upsample + Conv thay vì ConvTranspose để tránh gợn sóng răng cưa
        self.reconstructor = nn.Sequential(
            # Phóng to lại 4 lần bằng nội suy toán học mượt mà (24 -> 96)
            nn.Upsample(scale_factor=scale_factor, mode='linear', align_corners=False),
            # Conv1d làm mượt và đưa về 1 channel như ban đầu
            nn.Conv1d(in_channels=d_model, out_channels=1, kernel_size=kernel_size, padding=kernel_size//2)
        )
        
    def forward(self, x):                             # x: [B, C, win_size]
        B, C, win_size = x.size()
        x = x.contiguous().view(B*C, 1, win_size)     # [B*C, 1, win_size]
        h = self.compressor(x)                        # [B*C, d_model, win_size//4]
        x = self.reconstructor(h)                     # [B*C, 1, win_size]
        x = x.contiguous().view(B, C, win_size)       # [B, C, win_size]
        return x

class Expert(nn.Module):
    def __init__(self, win_size, d_model, top_k=2, channels = 1, num_experts = 5):
        super(Expert, self).__init__()
        self.num_experts = num_experts
        self.win_size = win_size
        num_experts = max(num_experts,3)
        num_global_expert = max(1, int(0.2*num_experts))
        num_local_expert = max(1, int(0.2*num_experts))
        num_wave_expert = num_experts - num_global_expert - num_local_expert
        experts = [WaveExpert(win_size) for _ in range(num_wave_expert)]
        experts.extend([GlobalExpert(win_size) for _ in range(num_global_expert)])
        experts.extend([LocalExpert(d_model = d_model) for _ in range(num_local_expert)])
        self.experts = nn.ModuleList(experts) 
        
    def forward(self, x):
        epsi = 1e-5
        B, win_size, C = x.size()
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        x_norm = (x - mean) / (std + epsi)
        x = x_norm.clone()
        
        x = x.permute(0,2,1)                                                # [B, C, win_size]
        # Ép qua bộ nén và bộ giải nén dung lượng thấp
        dec_x = []
        for expert in self.experts:
            recon_x = expert(x)                                          # x: [B, C, win_size], h: [B, C, 3 or win_size//4]
            dec_x.append(recon_x)                                         
        dec_x = torch.stack(dec_x, dim=1)                                   # [B, num_subsequences, C, win_size]
        
        # Tổng hợp tuyến tính (Cộng đại số không học tham số)
        x_out = dec_x.sum(dim=1)                                            # [B, C, win_size]
        x_out = x_out.permute(0,2,1)                                        # [B, win_size, C]
        
        return x_norm, dec_x, x_out
        
class Model(nn.Module):
    def __init__(self, win_size, d_model, top_k=2, channels=1, num_experts=5, num_group_experts=3):
        super(Model, self).__init__()
        self.experts = Expert(win_size, d_model, top_k=2, channels=1, num_experts=5)

    def forward(self, x):                                            
        x_norm, dec_x, x_out = self.experts(x)
        return x_norm, dec_x, x_out
    
class PureLoss(nn.Module):
    def __init__(self):
        super(PureLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.lambda_pure = 0.1
        self.lambda_var = 0.005
    
    def forward(self, batch_x_norm, batch_dec_x, batch_x_out): 
        eps = 1e-5
        # 1. Loss tái tạo 
        recon_loss = self.mse(batch_x_norm, batch_x_out)
        
        # 2. Tính ma trận Cosine tương quan bình phương
        # batch_dec_x_norm = batch_dec_x / torch.sqrt((batch_dec_x ** 2).sum(dim=-1, keepdim=True) + eps)     # [B, num_experts, C, win_size]
        # batch_dec_x_norm = batch_dec_x_norm.permute(0,2,1,3)                                                # [B, C, num_experts, win_size]
        # cos_sim_matrix = batch_dec_x_norm @ batch_dec_x_norm.permute(0, 1, 3, 2)                            # [B, C, num_experts, num_experts]
        # cos_sim_matrix_sq = cos_sim_matrix ** 2
        
        # # 3. Xóa đường chéo chính
        # num_sub = batch_dec_x.size(1)
        # diagonal_mask = torch.eye(num_sub, device=batch_dec_x.device).unsqueeze(0)
        # cos_sim_matrix_sq = cos_sim_matrix_sq * (1 - diagonal_mask)

        # # pure loss trung bình trên các kênh của các mẫu trong 1 batch
        # num_pairs = num_sub * (num_sub - 1)
        # pure_loss = cos_sim_matrix_sq.sum() / (batch_x_norm.size(0) * batch_x_norm.size(-1) * num_pairs)

        # # phạt nghiệm tầm thường
        # expert_std = batch_dec_x.std(dim=-1) # [B, num_experts, C]
        # var_penalty = torch.mean(1.0 / (expert_std + eps))
        
        # 4. Cộng tổng hợp có trọng số
        # total_loss = recon_loss + self.lambda_pure*pure_loss + self.lambda_var*var_penalty
        total_loss = recon_loss
        
        return total_loss
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
                 lambda_expert=0.5,
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
        self.lambda_expert = lambda_expert

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

        self.train_loader = train_loader
        
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
            train_var_loss = 0
            self.model.train()
            
            loop = tqdm.tqdm(enumerate(train_loader),total=len(train_loader),leave=True)
            for i, (batch_x, _) in loop:
                self.model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                out = self.model(batch_x)
                x_norm, dec_x, x_recon = out
                loss = self.criterion(x_norm, dec_x, x_recon)
                loss.backward()
                self.model_optim.step()
                
                train_loss += loss.cpu().item()
                # train_recon_loss += recon_loss.cpu().item()
                # train_pure_loss += pure_loss.cpu().item()
                # train_var_loss += var_loss.cpu().item()
                
                loop.set_description(f'Training Epoch [{epoch}/{self.epochs}]')
                # loop.set_postfix(loss=loss.item(), avg_loss=train_loss/(i+1), avg_recon_loss=train_recon_loss/(i+1), 
                #                  avg_pure_loss=train_pure_loss/(i+1), avg_var_loss = train_var_loss/(i+1))
                loop.set_postfix(loss=loss.item(), avg_loss=train_loss/(i+1))
            
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

                    loss = self.criterion(true, dec_x, pred)
                    total_loss.append(loss.item())
                    loop.set_description(f'Valid Epoch [{epoch}/{self.epochs}]')
                    
            valid_loss = np.average(total_loss)
            loop.set_postfix(loss=loss.item(), valid_loss=valid_loss)
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                print("   Early stopping<<<")
                break
            
            adjust_learning_rate(self.model_optim, epoch + 1, self.lradj, self.lr)
            
    def decision_function(self, data, k_ratio=0.1): # Truyền thêm tham số k_ratio
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

        top_k_channels = max(1, int(C * k_ratio)) # FIX 1: Dùng biến C
        full_counts = np.zeros(N)
        full_scores = np.zeros(N) # FIX 2: Đồng nhất tên full_scores
        
        # Lưu lại để debug
        full_recon = np.zeros((N, C))
        full_true = np.zeros((N, C))
        full_experts = np.zeros((N, self.num_experts, C))
        
        global_idx = 0
        loop = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), leave=True)
        
        with torch.no_grad(): # Đã bỏ luồng đạo hàm, thuật toán sẽ chạy cực mượt
            for i, (batch_x, _) in loop:
                batch_x = batch_x.float().to(self.device)
                
                if batch_x.dim() == 2:
                    batch_x = batch_x.unsqueeze(-1)
                
                B = batch_x.size(0)
                
                # Reconstruction
                x_norm, dec_x, outputs = self.model(batch_x)
                
                # Kích thước: [B, win_size, C]
                loss_matrix = self.anomaly_criterion(x_norm, outputs) 
                
                # ==========================================
                # LATE FUSION: KẾT HỢP ĐIỂM SỐ ĐA BIẾN
                # ==========================================
                if C > 1:
                    # Sort lỗi theo chiều Kênh (dim=-1), giảm dần
                    sorted_loss, _ = torch.sort(loss_matrix, dim=-1, descending=True)
                    
                    # Lấy Top K kênh tệ nhất: [B, win_size, top_k_channels]
                    top_k_loss = sorted_loss[:, :, :top_k_channels]
                    
                    # Tính trung bình của Top K kênh này: [B, win_size]
                    mse_scores = torch.mean(top_k_loss, dim=-1)
                else:
                    # Nếu đơn biến, lấy lỗi nguyên bản: [B, win_size]
                    mse_scores = loss_matrix.squeeze(-1)
                
                # Ép về NumPy
                mse_scores_np = mse_scores.cpu().numpy()

                # Ép kiểu và đưa về CPU (không cần .detach() vì đã dùng torch.no_grad())
                x_norm_np = x_norm.cpu().numpy()                        
                outputs_np = outputs.cpu().numpy()                      
                dec_x_np = dec_x.permute(0, 3, 1, 2).cpu().numpy()           
                
                # 2. Xử lý Overlap: Cộng dồn vào mảng Global
                for b in range(B):
                    start = global_idx + b
                    end = start + self.win_size
    
                    full_scores[start:end] += mse_scores_np[b] # FIX 3: Gọi đúng tên biến
                    full_true[start:end] += x_norm_np[b]
                    full_recon[start:end] += outputs_np[b]
                    full_experts[start:end] += dec_x_np[b]
                    full_counts[start:end] += 1
                    
                global_idx += B
                loop.set_description(f'Testing Phase: ')

        # Tránh chia cho 0 ở những điểm không được cover
        full_counts[full_counts == 0] = 1
        
        # 3. Tính trung bình (Averaging) để khử nhiễu và làm mượt
        full_scores = full_scores / full_counts # FIX 4: Bỏ [:, None] để tránh tràn RAM
        full_true = full_true / full_counts[:, None]
        full_recon = full_recon / full_counts[:, None]
        full_experts = full_experts / full_counts[:, None, None]

        self.__anomaly_score = full_scores # FIX 5: Bỏ lambda cũ
        
        # Đóng gói dữ liệu debug
        self.debug_data = {
            "scores": self.__anomaly_score,
            "true_seq": full_true,
            "recon_seq": full_recon,
            "expert_seqs": full_experts
        }
        
        if hasattr(self, 'save_path') and self.save_path:
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
