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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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


model = AE_1DCNN().to(device)
model = torch.load('Autoencoder_zero_18.pt', map_location=device)
model.eval()


def make_test(dirs):
    data, labels, folder_names = [], [], []

    path = Path("C:/Users/seoze/Desktop/KSAS2025/data/csv/csv_steps_nomal_18")

    for folder in dirs:
        folder_path = path / folder
        files = natsort.natsorted(folder_path.iterdir())

        for file_path in files:
            if file_path.suffix == ".csv":
                rawdata = pd.read_csv(file_path).values
            elif file_path.suffix == ".npy":
                rawdata = np.load(file_path)
            else:
                continue

            data.append(rawdata)
            labels.append(0 if folder == "dataset" else 1) 
            folder_names.append(folder)  

    print(f'총 테스트 데이터 개수: {len(data)}')

    data = np.array(data, dtype='float32')
    labels = np.array(labels, dtype='float32')

    test_X = torch.from_numpy(data).float()
    test_Y = torch.from_numpy(labels).long()

    test = TensorDataset(test_X, test_Y)
    test_loader = DataLoader(test, batch_size=1, shuffle=False)

    return test_loader, labels, folder_names


def eval_model(test_loader):
    residuals = []
    with torch.no_grad():
        for test_x, test_y in test_loader:
            test_x = test_x.to(device)
            output = model(test_x)
            temp_residual = output - test_x
            residual = torch.mean((temp_residual) ** 2).item()
            residuals.append((residual, test_y.cpu().item()))
    return residuals


def plot_reconstruction_mse(residuals, folder_names):
    mse_values = [r[0] for r in residuals]
    threshold = np.mean(mse_values) + 0.1 * np.std(mse_values)
    time_indices = range(3000)   

    normal_mse, abnormal_mse = np.full(3000, np.nan), np.full(3000, np.nan)

    normal_count, abnormal_count = 0, 0

    for i, folder in enumerate(folder_names):
        if folder == "dataset":
            normal_mse[normal_count] = mse_values[i]
            normal_count += 1
        else:  # "anormaly"
            abnormal_mse[abnormal_count] = mse_values[i]
            abnormal_count += 1

    plt.figure(figsize=(10, 5))

    plt.plot(time_indices, normal_mse, label="Normal", color="blue", linewidth=1)
    plt.plot(time_indices, abnormal_mse, label="Abnormal", color="red", linewidth=1)
    plt.axhline(y=threshold, color='purple', linestyle='dashed', label='Threshold')

    plt.xlabel('Time')
    plt.ylabel('MSE')
    plt.title('Reconstruction MSE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Abnormal"], yticklabels=["Normal", "Abnormal"])
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("Confusion Matrix")
    plt.show()

dirs = ['dataset', 'anormaly']
test_loader, y_true, folder_names = make_test(dirs)
residuals = eval_model(test_loader)

mse_values = [r[0] for r in residuals]
threshold = np.mean(mse_values) + 0.1 * np.std(mse_values)

y_pred = (np.array(mse_values) > threshold).astype(int)

plot_reconstruction_mse(residuals, folder_names)
plot_confusion_matrix(y_true, y_pred)

print("Classification Report:\n", classification_report(y_true, y_pred, target_names=["Normal", "Abnormal"]))
