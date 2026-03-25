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
        if a>30:
            break
        a = a+1

        data.append(features)
        label.append(features) #label도 동일 데이터 사용!
        
print('Num of data : ', len(data))
        
data  = np.array(data, dtype = 'float32') #numpy 배열로 변환
label = np.array(label, dtype = 'float32')
# print(f"✅ data.shape: {data.shape}")

# data = data.reshape(data.shape[0], 18 * 100)
# label = label.reshape(label.shape[0], 18 * 100)

#Train, Validation 데이터 분할(머니)
train_X, val_X, train_Y, val_Y = model_selection.train_test_split(data, label, test_size = 0.20,shuffle=True, random_state=random_state2[0]) #, stratify=label)

#PyTorch Tensor로 변환
train_X = torch.from_numpy(train_X).float()
train_Y = torch.from_numpy(train_Y).float()

# print(f"✅ train_X.shape: {train_X.shape}")  # PyTorch Tensor 형태 확인


val_X = np.array(val_X, dtype='float32')
val_Y = np.array(val_Y, dtype='float32')

val_X = torch.from_numpy(val_X).float()
val_Y = torch.from_numpy(val_Y).float()

train       = TensorDataset(train_X, train_Y)
validation  = TensorDataset(val_X, val_Y)

train_loader        = DataLoader(train, batch_size = 256, shuffle=True) # --> 128
validation_loader   = DataLoader(validation, batch_size = 128, shuffle=True) # --> 64

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
            torch.save(model,'Autoencoder_LSTM_zero_18.pt')
            print('============================\n',
            'Model save********************\n',
            '============================')        
        else:
            self.patience += 1
    
    def is_stop(self):
        return self.patience >= self.patience_limit

epoch = 3
patience = 60
iteration = 30
factor    = 0.08
min_lr    = 1e-10

model         = AE_LSTM().to(device)
early_stop    = EarlyStopping(patience)
criterion_MAE = nn.L1Loss()
criterion_MSE = nn.MSELoss()
optimizer     = optim.Adam(model.parameters(),lr=0.00001, weight_decay=0.00005)
scheduler     = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience = iteration, factor = factor, min_lr = min_lr, verbose = True)

start = time.time()


early_stop = EarlyStopping()


# 손실값을 저장할 CSV 파일 경로
loss_file = "loss_history_lstm.csv"

# CSV 파일 헤더 생성
with open(loss_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Train Loss", "Validation Loss"])

train_losses = []
val_losses = []

## Model Train for epoch
for epoch in range(epoch):
    model.train()
    total_loss = 0
    running_corrects=0
    
    # Train
    model.train()
    
    for train_x, train_y  in train_loader:
        train_x, train_y = train_x.to(device), train_y.to(device)

        optimizer.zero_grad()
        output = model(train_x)

        train_y = train_y.transpose(1, 2)
        loss = criterion_MSE(output, train_y)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.data
                
    train_epoch_loss = total_loss/len(train_loader.dataset)
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

    with torch.no_grad():
        val_predict = []
        val_value = []
        for val_x, val_y in validation_loader:
            val_x, val_y = val_x.to(device), val_y.to(device)
            optimizer.zero_grad()
            output = model(val_x)
            val_y = val_y.transpose(1, 2) 
            loss = criterion_MSE(output, val_y)
            total_loss += loss.data
        
        val_epoch_loss = total_loss/len(validation_loader.dataset)
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

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid()
plt.show()
