import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report
import natsort

# ✅ 디바이스 설정
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ✅ 1D-CNN Autoencoder 모델 정의
class AE_1DCNN(nn.Module):
    def __init__(self):
        super(AE_1DCNN, self).__init__()
        self.Encoder = nn.Sequential(
            nn.Conv1d(18, 256, 18, padding=1, stride=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 128, 18, padding=1, stride=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, 18, padding=1, stride=1), nn.BatchNorm1d(128), nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.Encoder_fc = nn.Sequential(
            nn.Linear(128 * 55, 2048), nn.ReLU(),
            nn.Linear(2048, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU()
        )
        self.latent_activation = nn.ReLU()
        self.Decoder_fc = nn.Sequential(
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 2048), nn.ReLU(),
            nn.Linear(2048, 128 * 55)
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(128, 55))
        self.Decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 128, 18, padding=1, stride=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.ConvTranspose1d(128, 256, 18, padding=1, stride=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.ConvTranspose1d(256, 18, 18, padding=1, stride=1)
        )

    def forward(self, x):
        x = self.Encoder(x)
        x = self.flatten(x)
        x = self.Encoder_fc(x)
        z = self.latent_activation(x)
        x = self.Decoder_fc(z)
        x = self.unflatten(x)
        x = self.Decoder(x)
        return x

# ✅ 모델 로드
model = AE_1DCNN().to(device)
model = torch.load('Autoencoder_zero_18.pt', map_location=device)
model.eval()

# ✅ 테스트 데이터 로드 함수
def make_test(folder):
    data = []
    path = Path(f"C:/Users/seoze/Desktop/KSAS2025/data/csv/csv_steps_nomal_18/{folder}")
    files = natsort.natsorted(path.iterdir())

    for file_path in files:
        if file_path.suffix == ".csv":
            rawdata = pd.read_csv(file_path).values
        elif file_path.suffix == ".npy":
            rawdata = np.load(file_path)
        else:
            continue
        data.append(rawdata)

    data = np.array(data, dtype='float32')
    test_X = torch.from_numpy(data).float()

    test = TensorDataset(test_X)
    test_loader = DataLoader(test, batch_size=1, shuffle=False)

    return test_loader

# ✅ 모델 평가 (MSE 계산)
def eval_model(test_loader):
    residuals = []
    with torch.no_grad():
        for test_x in test_loader:
            test_x = test_x[0].to(device)
            output = model(test_x)
            temp_residual = output - test_x
            residual = torch.mean((temp_residual) ** 2).item()
            residuals.append(residual)
    return residuals

# ✅ MSE 시각화 (정상/이상 구분 없이 단순 그래프만 표시)
def plot_reconstruction_mse(residuals):
    mse_values = np.array(residuals)
    threshold = 0.05  # ✅ 고정된 임계값 사용
    time_indices = range(len(mse_values))

    plt.figure(figsize=(10, 5))
    plt.plot(time_indices, mse_values, label="MSE", color="blue", linewidth=1)
    plt.axhline(y=threshold, color='purple', linestyle='dashed', label='Threshold')
    plt.xlabel('Time')
    plt.ylabel('MSE')
    plt.title('Reconstruction MSE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# ✅ Confusion Matrix 시각화
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Abnormal"], yticklabels=["Normal", "Abnormal"])
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("Confusion Matrix")
    plt.show()

# ✅ 실행
folder = 'anormaly'  # ✅ 폴더 변경 가능
test_loader = make_test(folder)
residuals = eval_model(test_loader)

# ✅ 임계값 설정 및 예측
mse_values = np.array(residuals)
threshold = 0.05  # ✅ 임의 설정
y_pred = (mse_values > threshold).astype(int)  # 임계값보다 크면 이상(1), 작으면 정상(0)

# ✅ Confusion Matrix와 모델 평가
plot_reconstruction_mse(residuals)
plot_confusion_matrix(np.zeros_like(y_pred), y_pred)  # ✅ 모든 데이터가 정상(0)이라는 가정하에 비교

# ✅ Classification Report 출력
print("Classification Report:\n", classification_report(np.zeros_like(y_pred), y_pred, target_names=["Normal", "Abnormal"]))
