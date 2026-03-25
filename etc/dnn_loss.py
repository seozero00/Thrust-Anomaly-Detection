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
import csv
import matplotlib.pyplot as plt

path = r"C:\Users\seoze\Desktop\KSAS2025\data\csv\csv_steps_nomal_18"
files = os.listdir(path)

print(f"총 CSV 파일 개수: {len(files)}")  # CSV 파일 개수 출력
print(files)  # 파일 이름 확인


print("현재 실행 중인 경로:", os.getcwd()) 

USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)

device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('학습을 진행하는 기기:',device)

#데이터 셔플링??
random_state2 = np.random.randint(42,size=1)
print("Random State :", random_state2)

##### Define data path #####
path = r"C:\Users\seoze\Desktop\KSAS2025\data\csv\csv_steps_nomal_18"
dirs = ['dataset']

##### Data generation #####
data = []
label = []


a = 1
for i in dirs:
    files = os.listdir(os.path.join(path, i))
    
    for j in files:
        file_path = os.path.join(path, i, j)      
        df = pd.read_csv(file_path)
        
        
        if df.shape == (18, 100):
            features = df.iloc[:, :].values
        else:  
            features = df.iloc[:100, :18].values.T  
        
        
        # print(f"파일명: {j}, features.shape: {features.shape}")

        # print(f"{j} | {features.shape}")
        # if a>30:
        #     break
        # a = a+1

        data.append(features)
        label.append(features) #label도 동일 데이터 사용!
        
print('Num of data : ', len(data))
        
data  = np.array(data, dtype = 'float32') #numpy 배열로 변환
label = np.array(label, dtype = 'float32')
# print(f" data.shape: {data.shape}")

# data = data.reshape(data.shape[0], 18 * 100)
# label = label.reshape(label.shape[0], 18 * 100)

#Train, Validation 데이터 분할(머니)
train_X, val_X, train_Y, val_Y = model_selection.train_test_split(data, label, test_size = 0.20,shuffle=True, random_state=random_state2[0]) #, stratify=label)

#PyTorch Tensor로 변환
train_X = torch.from_numpy(train_X).float()
train_Y = torch.from_numpy(train_Y).float()

print(f" train_X.shape: {train_X.shape}")  # PyTorch Tensor 형태 확인


val_X = np.array(val_X, dtype='float32')
val_Y = np.array(val_Y, dtype='float32')

val_X = torch.from_numpy(val_X).float()
val_Y = torch.from_numpy(val_Y).float()

train       = TensorDataset(train_X, train_Y)
validation  = TensorDataset(val_X, val_Y)

train_loader        = DataLoader(train, batch_size = 256, shuffle=True) # --> 128
validation_loader   = DataLoader(validation, batch_size = 128, shuffle=True) # --> 64


# 변경된 하이퍼파라미터 설정
lr = 0.0001  # 기존 0.00001 → 0.0001 (학습 속도 증가) # 변경됨
weight_decay = 0.00001  # 기존보다 감소하여 최적화 # 변경됨

batch_size_train = 512  # 기존 256 → 512 (병렬 연산 증가) # 변경됨
batch_size_val = 256  # 기존 128 → 256 # 변경됨

patience = 100  # 기존 60 → 100 (Early Stopping 대기 시간 증가) # 변경됨
early_stopping_threshold = 0.0005  # 기존 0.006 → 0.0005 (더 낮은 loss까지 학습 유지) # 변경됨

scheduler_patience = 50  # 기존 30 → 50 (조금 더 천천히 줄어들도록) # 변경됨
scheduler_factor = 0.5  # 기존과 동일 (0.5)

dropout_rate = 0.3  # Dropout 추가하여 과적합 방지 # 변경됨

# DataLoader 수정
train_loader = DataLoader(train, batch_size=batch_size_train, shuffle=True)  # 변경됨
validation_loader = DataLoader(validation, batch_size=batch_size_val, shuffle=True)  # 변경됨



# 모델 수정 (Dropout 추가)
class AE_DNN(nn.Module):
    def __init__(self):
        super(AE_DNN, self).__init__()

        self.flatten = nn.Flatten()
        # Encoder 정의
        self.Encoder = nn.Sequential(
            nn.Linear(18 * 100, 2048, bias=True), nn.ReLU(), nn.BatchNorm1d(2048),
            nn.Dropout(dropout_rate),  # Dropout 추가 # 변경됨
            
            nn.Linear(2048, 2048, bias=True), nn.ReLU(), nn.BatchNorm1d(2048),
            nn.Linear(2048, 1024, bias=True), nn.ReLU(), nn.BatchNorm1d(1024),
            nn.Dropout(dropout_rate),  # Dropout 추가 # 변경됨
            
            nn.Linear(1024, 1024, bias=True), nn.ReLU(), nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024, bias=True), nn.ReLU(), nn.BatchNorm1d(1024),
            nn.Linear(1024, 256, bias=True), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),  # Dropout 추가 # 변경됨
            
            nn.Linear(256, 256, bias=True), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Linear(256, 256, bias=True), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Linear(256, 256, bias=True), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Linear(256, 256, bias=True)  # Latent vector
        )

        # Decoder
        self.Decoder = nn.Sequential(
            nn.Linear(256, 256, bias=True),  # Latent vector
            nn.Linear(256, 256, bias=True), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Linear(256, 256, bias=True), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Linear(256, 256, bias=True), nn.ReLU(), nn.BatchNorm1d(256),
            
            nn.Linear(256, 1024, bias=True), nn.ReLU(), nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024, bias=True), nn.ReLU(), nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024, bias=True), nn.ReLU(), nn.BatchNorm1d(1024),
            nn.Linear(1024, 2048, bias=True), nn.ReLU(), nn.BatchNorm1d(2048),
            nn.Dropout(dropout_rate),  # Dropout 추가 # 변경됨

            nn.Linear(2048, 2048, bias=True), nn.ReLU(), nn.BatchNorm1d(2048),
            nn.Linear(2048, 18 * 100, bias=True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(18, 100))

    def forward(self, x):
        x = self.flatten(x)  
        z = self.Encoder(x)
        x = self.Decoder(z)
        x = self.unflatten(x)
        return x



epoch = 1000
patience = 60
iteration = 30
factor    = 0.08
min_lr    = 1e-10

criterion_MAE = nn.L1Loss()
criterion_MSE = nn.MSELoss()
# 모델 및 옵티마이저 수정
model = AE_DNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # 변경됨
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=scheduler_patience, factor=scheduler_factor, min_lr=1e-10, verbose=True)  # 변경됨

# Early Stopping 기준 변경
class EarlyStopping:
    def __init__(self, patience=patience):  # 변경됨
        self.loss = np.inf
        self.patience = 0
        self.patience_limit = patience

    def step(self, loss, model):
        if self.loss > loss:
            self.loss = loss
            self.patience = 0
            print('find proper parameter...')
            torch.save(model, 'Autoencoder_DNN_zero_18.pt')
            print('============================\n',
                'Model save********************\n',
                '============================')
        else:
            self.patience += 1

    def is_stop(self):
        return self.patience >= self.patience_limit

early_stop = EarlyStopping()


# 손실값을 저장할 CSV 파일 경로
loss_file = "loss_history_dnn.csv"

# CSV 파일 헤더 생성
with open(loss_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Train Loss", "Validation Loss"])

train_losses = []
val_losses = []

# 학습 루프에서 손실값 저장 추가
for epoch in range(epoch):
    model.train()
    total_loss = 0
    
    for train_x, train_y in train_loader:
        train_x, train_y = train_x.to(device), train_y.to(device)
        optimizer.zero_grad()
        
        output = model(train_x)
        loss = criterion_MSE(output, train_y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    train_epoch_loss = total_loss / len(train_loader.dataset)
    train_losses.append(train_epoch_loss)

    print('========== Train Result =========')
    print(epoch + 1)
    print('Train Loss: {:.9f} '.format(train_epoch_loss))
    print('=================================')

    model.eval()
    total_loss = 0

    with torch.no_grad():
        for val_x, val_y in validation_loader:
            val_x, val_y = val_x.to(device), val_y.to(device)
            output = model(val_x)
            loss = criterion_MSE(output, val_y)
            total_loss += loss.item()
        
        val_epoch_loss = total_loss / len(validation_loader.dataset)
        val_losses.append(val_epoch_loss)

        print('====== Validation Result ========')
        print('Validation Loss: {:.9f}'.format(val_epoch_loss))
        print('=================================')

        # 손실값을 CSV 파일에 저장
        with open(loss_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, train_epoch_loss, val_epoch_loss])

        abs_los = abs(val_epoch_loss - train_epoch_loss)

        if val_epoch_loss < early_stopping_threshold:
            early_stop.step(abs_los, model)
            if early_stop.is_stop():
                print('============================\n',
                        'Train done\n',
                        '============================')
                break

        scheduler.step(val_epoch_loss)

# 손실값을 그래프로 시각화
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid()
plt.show()
