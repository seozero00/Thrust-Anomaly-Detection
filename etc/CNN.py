# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import os
# import time
# from torch.autograd import Variable
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn import model_selection
# import natsort
# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path
# from sklearn.metrics import classification_report, confusion_matrix

# # CUDA 사용 여부 및 device 설정
# USE_CUDA = torch.cuda.is_available()
# print("USE_CUDA:", USE_CUDA)
# device = torch.device('cuda:0' if USE_CUDA else 'cpu')
# print('학습을 진행하는 기기:', device)

# # 랜덤 스테이트 확인
# random_state2 = np.random.randint(42, size=1)
# print("Random State:", random_state2)

# # Autoencoder 모델 정의 (1D CNN 기반)
# class AE_1DCNN(nn.Module):
#     def __init__(self):
#         super(AE_1DCNN, self).__init__()
#         # Encoder 정의
#         self.Encoder = nn.Sequential(
#             nn.Conv1d(in_channels=18, out_channels=256, kernel_size=18, padding=1, stride=1),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Conv1d(in_channels=256, out_channels=128, kernel_size=18, padding=1, stride=1),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Conv1d(in_channels=128, out_channels=128, kernel_size=18, padding=1, stride=1),
#             nn.BatchNorm1d(128),
#             nn.ReLU()
#         )
#         # Flatten 후 Fully Connected layer로 잠재 벡터 변환
#         self.flatten = nn.Flatten()
#         self.Encoder_fc = nn.Sequential(
#             nn.Linear(128 * 55, 2048), nn.ReLU(),
#             nn.Linear(2048, 1024), nn.ReLU(),
#             nn.Linear(1024, 1024), nn.ReLU()
#         )
#         self.latent_activation = nn.ReLU()  # 잠재 벡터 활성화 함수

#         # Decoder 부분 - FC Layer와 Unflatten, 그 후 1D ConvTranspose
#         self.Decoder_fc = nn.Sequential(
#             nn.Linear(1024, 1024), nn.ReLU(),
#             nn.Linear(1024, 2048), nn.ReLU(),
#             nn.Linear(2048, 128 * 55)
#         )
#         self.unflatten = nn.Unflatten(dim=1, unflattened_size=(128, 55))

#         self.Decoder = nn.Sequential(
#             nn.ConvTranspose1d(in_channels=128, out_channels=128, kernel_size=18, padding=1, stride=1),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.ConvTranspose1d(in_channels=128, out_channels=256, kernel_size=18, padding=1, stride=1),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.ConvTranspose1d(in_channels=256, out_channels=18, kernel_size=18, padding=1, stride=1)
#         )

#     def forward(self, x):
#         x = self.Encoder(x)
#         x = self.flatten(x)
#         x = self.Encoder_fc(x)
#         z = self.latent_activation(x)
#         x = self.Decoder_fc(z)
#         x = self.unflatten(x)
#         x = self.Decoder(x)
#         return x

# # 모델 로드 (저장된 모델이 있다면)
# model = AE_1DCNN().to(device)
# model = torch.load('Autoencoder_zero_18.pt', map_location=device)
# model.eval()

# # 데이터 불러오기: 여기서는 'anormaly' 디렉토리 내 파일만 사용 (필요에 따라 수정)
# dirs = ['anormaly']

# def make_test(dirs):
#     data  = []
#     label = []
#     list_ = []
#     shape_ = 0

#     # 데이터 경로 (경로에 맞게 수정)
#     path = Path("C:/Users/seoze/Desktop/KSAS2025/data/csv/csv_steps_nomal_18")

#     for i in dirs:
#         folder_path = path / i  # 폴더 경로 설정
#         files = natsort.natsorted(folder_path.iterdir())  # 파일 리스트 정렬

#         for file_path in files:
#             if file_path.suffix == ".csv":  # CSV 파일 처리
#                 rawdata = pd.read_csv(file_path).values
#             elif file_path.suffix == ".npy":  # NumPy 파일 처리
#                 rawdata = np.load(file_path)
#             else:
#                 continue  # 지원하지 않는 파일 형식은 무시

#             data.append(rawdata)
#             label.append(shape_)
#             list_.append(file_path.stem[:41])

#         shape_ += 1
#     print('|| Num of test files : ', len(data))

#     df = pd.DataFrame(list_)
#     df.to_csv('test_normal_1DCNN.csv', index=False)

#     data  = np.array(data, dtype='float32')
#     label = np.array(label, dtype='float32')

#     test_X = torch.from_numpy(data).float()
#     test_Y = torch.from_numpy(label).long()

#     test = TensorDataset(test_X, test_Y)
#     test_loader = DataLoader(test, batch_size=1, shuffle=False)
    
#     return test_loader

# # 모델 평가: 각 테스트 샘플에 대해 reconstruction error 계산 및 결과 저장
# def eval_model(test_loader):
#     results = []
#     residuals = []
#     mse_contributions = []
    
#     with torch.no_grad():
#         for test_x, test_y in test_loader:
#             test_x = test_x.to(device)
#             output = model(test_x)
#             temp_residual = output - test_x
#             # 전체 MSE 계산
#             residual = torch.mean((temp_residual) ** 2).item()
#             label = test_y.cpu().item()
            
#             residuals.append((residual, label))
#             # 각 변수별 절대 오차 합산
#             temp_result = temp_residual.cpu().numpy().squeeze()
#             residual_sums = [np.abs(np.sum(temp_result[i])) for i in range(18)]
#             mse_contributions.append(residual_sums)
#             results.append(residual_sums + [label])
    
#     df = pd.DataFrame(results)
#     df.to_csv('test_results.csv', index=False)
    
#     return residuals, mse_contributions

# # MSE 기여도를 변수별로 플롯
# def plot_mse_contribution(mse_contributions):
#     mse_contributions = np.array(mse_contributions)
#     mean_mse = np.mean(mse_contributions, axis=0)
#     mse_percentage = (np.abs(mean_mse) / np.sum(np.abs(mean_mse))) * 100

#     variables = ["a_x", "a_y", "a_z", "P", "Q", "R", "THR1", "THR2", "THR3", "THR4", "THR5", "THR6",
#                 "Phi", "Theta", "Psi", "U", "V", "W"]
#     if len(mse_percentage) > len(variables):
#         mse_percentage = mse_percentage[:len(variables)]
    
#     plt.figure(figsize=(12, 5))
#     plt.bar(variables, mse_percentage)
#     plt.xticks(rotation=45, ha='right')
#     plt.ylabel('MSE Contribution [%]')
#     plt.title(f'Time: {len(mse_contributions):.2f}')
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.show()

# # reconstruction error (MSE) 값을 시간에 따라 플롯 및 임계값 표시
# def plot_reconstruction_mse(residuals):
#     mse_values = [r[0] for r in residuals]
#     time_indices = range(len(mse_values))
#     threshold = np.mean(mse_values) + 0.1 * np.std(mse_values)  # 임계값 설정
    
#     plt.figure(figsize=(10, 5))
#     plt.plot(time_indices, mse_values, label='MSE', color='blue')
#     plt.axhline(y=threshold, color='purple', linestyle='dashed', label='Threshold')
#     plt.xlabel('Time')
#     plt.ylabel('MSE')
#     plt.title('Reconstruction MSE')
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.show()

# # 테스트 데이터 생성 및 평가 실행
# test_loader = make_test(dirs)
# residuals, mse_contributions = eval_model(test_loader)

# plot_mse_contribution(mse_contributions)
# plot_reconstruction_mse(residuals)
# mse_values = [r[0] for r in residuals]
# # 임계값 설정 (평균 + 0.1*표준편차)
# threshold = np.mean(mse_values) + 0.1 * np.std(mse_values)
# print("Threshold:", threshold)

# # 임계값을 기준으로 예측: mse가 threshold보다 크면 anomaly (1), 아니면 normal (0)
# predicted_labels = [1 if mse > threshold else 0 for mse in mse_values]

# # 기존 true_labels는 파일 로딩 시 부여된 값이나, 여기서는 전체 데이터가 정상으로 되어 있으므로
# # 실제로 1500번째 샘플 이후부터 고장(Anomaly)라고 가정하여 ground truth를 재정의합니다.
# num_samples = len(residuals)
# updated_true_labels = [0 if i < 1250 else 1 for i in range(num_samples)]

# # 성능 평가 보고서 출력
# report = classification_report(updated_true_labels, predicted_labels, target_names=['Normal', 'Anomaly'])
# print("Classification Report:\n", report)

# # 혼동행렬 출력
# cm = confusion_matrix(updated_true_labels, predicted_labels)
# print("Confusion Matrix:")
# print(cm)

# # Ground Truth 기준 상태 비율 계산
# normal_percentage = (updated_true_labels.count(0) / len(updated_true_labels)) * 100
# anomaly_percentage = (updated_true_labels.count(1) / len(updated_true_labels)) * 100
# print(f"Ground Truth - Normal: {normal_percentage:.2f}%")
# print(f"Ground Truth - Anomaly: {anomaly_percentage:.2f}%")
import torch
import torch.nn as nn
import numpy as np
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import natsort
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

# CUDA 사용 여부 설정
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('학습을 진행하는 기기:', device)

# 랜덤 시드 설정
random_state2 = np.random.randint(42, size=1)
print("Random State:", random_state2)
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

# 데이터 불러오기 (이상 데이터 포함)
dirs = ['anormaly']

def make_test(dirs):
    data, label, list_ = [], [], []
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
                continue

            data.append(rawdata)
            label.append(shape_)
            list_.append(file_path.stem[:41])

        shape_ += 1

    print('|| Num of test files:', len(data))
    df = pd.DataFrame(list_)
    df.to_csv('test_normal_DNN.csv', index=False)

    data = np.array(data, dtype='float32')
    label = np.array(label, dtype='float32')

    test_X = torch.from_numpy(data).float()
    test_Y = torch.from_numpy(label).long()

    test = TensorDataset(test_X, test_Y)
    test_loader = DataLoader(test, batch_size=1, shuffle=False)

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


# 변수별 MSE 기여도 플롯
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

# MSE 기반 이상 탐지 시각화
def plot_reconstruction_mse(residuals):
    mse_values = [r[0] for r in residuals]
    time_indices = range(len(mse_values))
    threshold = np.mean(mse_values) + 0.1 * np.std(mse_values)

    plt.figure(figsize=(10, 5))
    plt.plot(time_indices, mse_values, label='MSE', color='blue')
    plt.axhline(y=threshold, color='purple', linestyle='dashed', label='Threshold')
    plt.xlabel('Time')
    plt.ylabel('MSE')
    plt.title('Reconstruction MSE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# 테스트 실행
test_loader = make_test(dirs)
residuals, mse_contributions = eval_model(test_loader)

plot_mse_contribution(mse_contributions)
plot_reconstruction_mse(residuals)

mse_values = [r[0] for r in residuals]
threshold = np.mean(mse_values) + 0.1 * np.std(mse_values)
print("Threshold:", threshold)

predicted_labels = [1 if mse > threshold else 0 for mse in mse_values]
updated_true_labels = [0 if i < 1250 else 1 for i in range(len(residuals))]

# 성능 평가
report = classification_report(updated_true_labels, predicted_labels, target_names=['Normal', 'Anomaly'])
print("Classification Report:\n", report)

cm = confusion_matrix(updated_true_labels, predicted_labels)
print("Confusion Matrix:\n", cm)
