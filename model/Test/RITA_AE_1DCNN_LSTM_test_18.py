import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import pandas as pd
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from sklearn import model_selection
import natsort
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ============================
# 환경 설정
# ============================
USE_CUDA = torch.cuda.is_available()
print("CUDA 사용 가능 여부:", USE_CUDA)

device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('학습을 진행하는 기기:', device)

random_state2 = np.random.randint(42, size=1)
print("Random State :", random_state2)

# ============================
# 모델 정의 (Autoencoder CNN -> LSTM)
# ============================
class DeepCNN_LSTM_AE(nn.Module):
    def __init__(self, seq_len=100, input_dim=18, latent_dim=256):
        super(DeepCNN_LSTM_AE, self).__init__()

        self.seq_len = seq_len  # seq_len을 여기서 정의하여 class 내에서 사용할 수 있도록 합니다.

        # Encoder CNN
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),

            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),

            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
        )

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=512, hidden_size=latent_dim,
            num_layers=3, batch_first=True, bidirectional=True, dropout=0.3
        )

        self.to_latent = nn.Linear(latent_dim*2, latent_dim)

        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=latent_dim, hidden_size=512,
            num_layers=3, batch_first=True, dropout=0.3
        )

        # Decoder CNN
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),

            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),

            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),

            nn.ConvTranspose1d(64, input_dim, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):  # x: [B, 18, T]
        x = self.encoder_cnn(x)        # [B, 512, T/16]
        x = x.permute(0, 2, 1)         # [B, T/16, 512]

        h_seq, (h, _) = self.encoder_lstm(x)
        z = torch.mean(h_seq, dim=1)   # mean pooling
        z = self.to_latent(z)          # [B, latent_dim]

        z_repeat = z.unsqueeze(1).repeat(1, self.seq_len//16, 1)
        out, _ = self.decoder_lstm(z_repeat)
        out = out.permute(0, 2, 1)     # [B, 512, T/16]

        out = self.decoder_cnn(out)    # [B, 18, T]
        return out


# ============================
# 데이터 로딩 함수
# ============================
def make_test(dirs):
    data = []
    label = []
    list_ = []
    shape_ = 0

    path = Path("C:/Users/user/Desktop/rita 가즈아/zero/data")

    for i in dirs:
        folder_path = path / i  # 폴더 경로 설정
        files = natsort.natsorted(folder_path.iterdir())  # 파일 리스트 정렬

        for file_path in files:
            if file_path.suffix == ".csv":  # CSV 파일 처리
                rawdata = pd.read_csv(file_path).values
            elif file_path.suffix == ".npy":  # NumPy 파일 처리
                rawdata = np.load(file_path)
            else:
                continue  # 지원하지 않는 파일 형식은 무시

            data.append(rawdata)
            label.append(shape_)
            list_.append(file_path.stem[:41])

        shape_ += 1

    print('|| Num of test files:', len(data))

    df = pd.DataFrame(list_)
    df.to_csv('test_normal_1DCNN.csv', index=False)

    data = np.array(data, dtype='float32')
    label = np.array(label, dtype='float32')

    test_X = torch.from_numpy(data).float()
    test_Y = torch.from_numpy(label).long()

    test = TensorDataset(test_X, test_Y)
    test_loader = DataLoader(test, batch_size=1, shuffle=False)

    return test_loader


# ============================
# 모델 평가 함수
# ============================
def eval_model(test_loader):
    results = []
    residuals = []
    mse_contributions = []

    with torch.no_grad():
        for test_x, test_y in test_loader:
            test_x = test_x.to(device)
            output = model(test_x)

            # 텐서의 크기 맞추기
            if output.size(2) != test_x.size(2):
                # 길이가 맞지 않으면 output을 test_x에 맞게 슬라이싱
                if output.size(2) > test_x.size(2):
                    output = output[:, :, :test_x.size(2)]  # output의 길이를 줄여서 맞춤
                else:
                    # 또는 test_x에 패딩을 추가하여 크기를 맞출 수 있습니다.
                    padding_size = test_x.size(2) - output.size(2)
                    output = nn.functional.pad(output, (0, padding_size))  # 오른쪽에 패딩 추가

            # Residual 계산 (MSE)
            temp_residual = output - test_x
            residual = torch.mean((temp_residual) ** 2).item()
            label = test_y.cpu().item()

            residuals.append((residual, label))
            temp_result = temp_residual.cpu().numpy().squeeze()
            residual_sums = [np.abs(np.sum(temp_result[i])) for i in range(18)]  # 각 변수별 residual 기여도
            mse_contributions.append(residual_sums)
            results.append(residual_sums + [label])

    df = pd.DataFrame(results)
    df.to_csv('test_results.csv', index=False)

    return residuals, mse_contributions

# ============================
# 혼동행렬 시각화 함수
# ============================
# ============================
# 혼동행렬 시각화 함수
# ============================
def plot_confusion_matrix(test_loader, threshold=0.5):
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for test_x, test_y in test_loader:
            test_x = test_x.to(device)
            test_y = test_y.to(device)

            output = model(test_x)

            # Reconstruction MSE 계산
            mse = torch.mean((output - test_x) ** 2, dim=(1, 2))

            # 예측을 MSE 기준으로 threshold로 이진 분류
            predicted = (mse > threshold).float()  # threshold 이상이면 이상치로 판단

            all_labels.extend(test_y.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # 혼동행렬 계산
    cm = confusion_matrix(all_labels, all_preds)

    # 혼동행렬 시각화
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# ============================
# 시각화 함수
# ============================
def plot_mse_contribution(mse_contributions):
    mse_contributions = np.array(mse_contributions)
    mean_mse = np.mean(mse_contributions, axis=0)
    mse_percentage = (np.abs(mean_mse) / np.sum(np.abs(mean_mse))) * 100

    variables = ["a_x", "a_y", "a_z", "P", "Q", "R", "THR1", "THR2", "THR3", "THR4", "THR5", "THR6", "Phi", "Theta", "Psi", "U", "V", "W"]

    if len(mse_percentage) > len(variables):
        mse_percentage = mse_percentage[:len(variables)]

    plt.figure(figsize=(12, 5))
    plt.bar(variables, mse_percentage)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('MSE Contribution [%]')
    plt.title(f'Time: {len(mse_contributions):.2f}')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def plot_reconstruction_mse(residuals):
    mse_values = [r[0] for r in residuals]
    time_indices = range(len(mse_values))
    threshold = np.mean(mse_values) + 0.3 * np.std(mse_values)  # 임계값 설정

    plt.figure(figsize=(10, 5))
    plt.plot(time_indices, mse_values, label='MSE', color='blue')
    plt.axhline(y=threshold, color='purple', linestyle='dashed', label='Threshold')
    plt.xlabel('Time')
    plt.ylabel('MSE')
    plt.title('Reconstruction MSE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


# ============================
# 모델 불러오기 및 평가
# ============================
model = DeepCNN_LSTM_AE().to(device)
model = torch.load('Autoencoder_zero_CNNLSTM_2.pt', map_location=device)
model.eval()

dirs = ['anomal_data']  # 데이터 디렉토리 설정

test_loader = make_test(dirs)  # 데이터 로드
residuals, mse_contributions = eval_model(test_loader)  # 모델 평가

plot_mse_contribution(mse_contributions)  # MSE 기여도 시각화
plot_reconstruction_mse(residuals)  # 재구성 MSE 시각화
plot_confusion_matrix(test_loader)  # 혼동행렬 시각화
