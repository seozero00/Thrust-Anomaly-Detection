import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os, glob
import time
import pandas as pd
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from sklearn import model_selection
import natsort
import matplotlib.pyplot as plt
import csv

# ============================
# 기본 설정
# ============================
print("현재 실행 중인 경로:", os.getcwd())

USE_CUDA = torch.cuda.is_available()
print("CUDA 사용 가능 여부:", USE_CUDA)

device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('학습을 진행하는 기기:', device)

# 랜덤 시드 고정
random_state2 = np.random.randint(42, size=1)
print("Random State :", random_state2)

# ============================
# 데이터 로딩
# ============================
base_path = r"C:\Users\user\Desktop\rita 가즈아\zero\data\nomal_data"
dirs = ["CSV_Steps_back", "CSV_Steps_front", "CSV_Steps_hover", "CSV_Steps_waypoint"]

data = []
label = []

for d in dirs:
    folder = os.path.join(base_path, d)
    files = glob.glob(os.path.join(folder, "*.csv"))
    print(f"[{d}] CSV 개수: {len(files)}")

    if "waypoint" in d:
        sets = {}
        for f in files:
            set_id = os.path.basename(f).split("_")[1]  # FlightLog_20250904T224812_Step0001.csv → 20250904T224812
            if set_id not in sets:
                sets[set_id] = []
            sets[set_id].append(f)

        for sid, f_list in sets.items():
            f_list = sorted(f_list)
            for f in f_list:
                df = pd.read_csv(f, header=None)
                if df.shape == (19, 100):
                    features = df.iloc[1:, :].values  # 첫 행 제외 → 18 x 100
                else:
                    features = df.iloc[:100, :18].values.T
                data.append(features)
                label.append(features)

    else:
        for f in sorted(files):
            df = pd.read_csv(f, header=None)
            if df.shape == (19, 100):
                features = df.iloc[1:, :].values
            else:
                features = df.iloc[:100, :18].values.T
            data.append(features)
            label.append(features)

print('총 데이터 개수 : ', len(data))

# numpy 변환
data = np.array(data, dtype='float32')
label = np.array(label, dtype='float32')

# Train / Validation Split
train_X, val_X, train_Y, val_Y = model_selection.train_test_split(
    data, label, test_size=0.20, shuffle=True, random_state=random_state2[0]
)

# Tensor 변환
train_X = torch.from_numpy(train_X).float()
train_Y = torch.from_numpy(train_Y).float()
val_X = torch.from_numpy(val_X).float()
val_Y = torch.from_numpy(val_Y).float()

# Dataset + DataLoader
train = TensorDataset(train_X, train_Y)
validation = TensorDataset(val_X, val_Y)

train_loader = DataLoader(train, batch_size=256, shuffle=True)
validation_loader = DataLoader(validation, batch_size=128, shuffle=True)

# ============================
# Autoencoder 정의 (CNN -> LSTM)
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
# EarlyStopping
# ============================
class EarlyStopping:
    def __init__(self, patience=5):
        self.loss = np.inf
        self.patience = 0
        self.patience_limit = patience

    def step(self, loss, model):
        if self.loss > loss:
            self.loss = loss
            self.patience = 0
            print('find proper parameter...')
            torch.save(model, 'Autoencoder_zero_CNNLSTM_2.pt')
            print('============================\n',
                  'Model save********************\n',
                  '============================')
        else:
            self.patience += 1

    def is_stop(self):
        return self.patience >= self.patience_limit

# ============================
# 학습 준비
# ============================
epoch = 1000
patience = 60
iteration = 30
factor    = 0.08
min_lr    = 1e-10

model         = DeepCNN_LSTM_AE(seq_len=100, input_dim=18, latent_dim=64).to(device)
early_stop    = EarlyStopping(patience)
criterion_MSE = nn.MSELoss()
optimizer     = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.00005)
scheduler     = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=iteration, factor=factor, min_lr=min_lr, verbose=True
)

start = time.time()

## ============================
# Training Loop
# ============================
train_losses = []
val_losses = []
loss_file = "loss_history_lstm.csv"

# CSV 파일 헤더 생성
with open(loss_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Train Loss", "Validation Loss"])

for epoch in range(epoch):
    model.train()
    total_loss = 0
    running_corrects=0
    
    # Train
    model.train()
    Num_batch = 0
    for train_x, train_y  in train_loader:
        Num_batch = Num_batch + 1
        train_x, train_y = train_x.to(device), train_y.to(device)

        optimizer.zero_grad()
        output = model(train_x)

        train_y = train_y.transpose(1, 2)
        loss = criterion_MSE(output, train_y)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.data
    """ !!! """
    train_epoch_loss = total_loss / Num_batch
    """ !!! """
    train_losses.append(train_epoch_loss.cpu().item())
    
    print('========== Train Result =========')
    print(epoch+1)
    print('Len_train_loader',len(train_loader.dataset))
    print('Train Loss: {:.9f} '.format(train_epoch_loss))
    print('=================================')
    
    # Validation
    model.eval()
    
    total_loss = 0
    running_corrects=0
    Num_batch = 0
    with torch.no_grad():
        Num_batch = Num_batch + 1
        val_predict = []
        val_value = []
        for val_x, val_y in validation_loader:
            val_x, val_y = val_x.to(device), val_y.to(device)
            optimizer.zero_grad()
            output = model(val_x)
            val_y = val_y.transpose(1, 2) 
            loss = criterion_MSE(output, val_y)
            total_loss += loss.data
        
        """ !!! """
        val_epoch_loss = total_loss/Num_batch
        """ !!! """
        val_losses.append(val_epoch_loss)

        print('====== Validation Result ========')
        print('Len_test_loader',len(validation_loader.dataset))
        print('Validation Loss: {:.9f}'.format(val_epoch_loss))
        print('=================================')
        with open(loss_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, train_epoch_loss, val_epoch_loss])

        abs_los = abs(val_epoch_loss - train_epoch_loss)

        if val_epoch_loss < 0.006:

            early_stop.step(abs_los.item(), model)
                
            if early_stop.is_stop():
                print('============================\n',
                    'Train done\n',
                    '============================')
                break

        scheduler.step(val_epoch_loss)

print("time :", time.time() - start)

# Loss 그래프 시각화
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid()
plt.show()
