import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from sklearn import model_selection
import natsort
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)

device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('학습을 진행하는 기기:',device)

random_state2 = np.random.randint(42,size=1)
print("Random State :", random_state2)


class AE_LSTM(nn.Module):
    def __init__(self, input_size=18, hidden_size=128, latent_size=64, num_layers=1):
        super(AE_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers

        # Encoder: LSTM + Fully Connected (FC)
        self.encoder = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc_enc = nn.Sequential(
            nn.Linear(hidden_size, latent_size),
            nn.ReLU()
        )

        # Decoder: Fully Connected (FC) + LSTM
        self.fc_dec = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU()
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc_out = nn.Linear(hidden_size, input_size)  # 최종 출력

    def forward(self, x):
        B = x.size(0)  # Batch size
        x = x.transpose(1, 2)  # [B, 18, 100] → [B, 100, 18] (LSTM expects this shape)

        # Encoder
        out, (h_n, c_n) = self.encoder(x)  # LSTM Encoding
        h_last = h_n[-1]  # Last hidden state
        latent = self.fc_enc(h_last)  # [B, latent_size]

        # Decoder
        h_dec = self.fc_dec(latent)  # [B, hidden_size]
        h_dec = h_dec.unsqueeze(0).expand(self.num_layers, B, self.hidden_size)  

        # Initial input for decoder (zero tensor)
        dec_input = torch.zeros(B, 100, self.hidden_size, device=x.device)  
        dec_output, _ = self.decoder(dec_input, (h_dec, torch.zeros_like(h_dec)))  
        x_recon = self.fc_out(dec_output)  # [B, 100, 18]

        return x_recon  # Reconstructed output
    

model = AE_LSTM().to(device)
model = torch.load('Autoencoder_LSTM_zero_18.pt', map_location=device)
model.eval()



# ✅ 테스트 데이터셋 준비
# dirs = ['anormaly']
dirs = ['anormaly']

def make_test(dirs):
    data  = []
    label = []
    file_names = []
    shape_ = 0

    path = Path("C:/Users/seoze/Desktop/KSAS2025/data/csv/csv_steps_nomal_18")

    for i in dirs:
        folder_path = path / i
        files = natsort.natsorted(folder_path.iterdir())

        for file_path in files:
            if file_path.suffix == ".csv":
                rawdata = pd.read_csv(file_path).values
            elif file_path.suffix == ".npy":
                rawdata = np.load(file_path)
            else:
                continue  # 지원하지 않는 파일 형식 무시

            data.append(rawdata)
            label.append(shape_)
            file_names.append(file_path.stem[:41])

        shape_ += 1

    print('|| Num of test files:', len(data))

    df = pd.DataFrame(file_names)
    df.to_csv('test_anormaly_LSTM.csv', index=False)

    data  = np.array(data, dtype='float32')
    label = np.array(label, dtype='float32')

    test_X = torch.from_numpy(data).float()
    test_Y = torch.from_numpy(label).long()

    test_dataset = TensorDataset(test_X, test_Y)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return test_loader

def eval_model(test_loader):
    results           = []
    residuals         = []
    mse_contributions = []

    with torch.no_grad():
        for test_x, test_y in test_loader:
            test_x = test_x.to(device)
            output = model(test_x)

            # 🔹 차원 변경 (LSTM 모델은 [B, 100, 18]을 출력하므로, 입력과 일치시키기)
            test_x = test_x.transpose(1, 2)  # [B, 18, 100] → [B, 100, 18]

            temp_residual = output - test_x  # Reconstruction Error
            residual = torch.mean((temp_residual) ** 2).item()
            label = test_y.cpu().item()

            residuals.append((residual, label))

            # 🔹 변수별 MSE 기여도 계산
            temp_result = temp_residual.cpu().numpy().squeeze()
            residual_sums = [np.abs(np.sum(temp_result[:, i])) for i in range(18)]
            mse_contributions.append(residual_sums)

            results.append(residual_sums + [label])  

    df = pd.DataFrame(results)
    df.to_csv('test_results_LSTM.csv', index=False)

    return residuals, mse_contributions

# ✅ 변수별 MSE 기여도 플롯
def plot_mse_contribution(mse_contributions):
    mse_contributions = np.array(mse_contributions)
    mean_mse = np.mean(mse_contributions, axis=0)
    mse_percentage = (np.abs(mean_mse) / np.sum(np.abs(mean_mse))) * 100

    variables = ["a_x", "a_y", "a_z", "P", "Q", "R", "THR1", "THR2", "THR3", "THR4", "THR5", "THR6",
                 "Phi", "Theta", "Psi", "U", "V", "W"]

    if len(mse_percentage) > len(variables):
        mse_percentage = mse_percentage[:len(variables)]

    plt.figure(figsize=(12, 5))
    plt.bar(variables, mse_percentage)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('MSE Contribution [%]')
    plt.title(f'Time: {len(mse_contributions):.2f}')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# ✅ Reconstruction MSE 플롯
def plot_reconstruction_mse(residuals):
    mse_values = [r[0] for r in residuals]
    time_indices = range(len(mse_values))
    threshold = np.mean(mse_values) + 0.01 * np.std(mse_values)  # 임계값 설정

    plt.figure(figsize=(10, 5))
    plt.plot(time_indices, mse_values, label='MSE', color='blue')
    plt.axhline(y=threshold, color='purple', linestyle='dashed', label='Threshold')
    plt.xlabel('Time')
    plt.ylabel('MSE')
    plt.title('Reconstruction MSE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# ✅ 테스트 실행
test_loader = make_test(dirs)
residuals, mse_contributions = eval_model(test_loader)

# ✅ 시각화
plot_mse_contribution(mse_contributions)
plot_reconstruction_mse(residuals)
