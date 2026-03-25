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
        
        # print(f"{j} | {features.shape}")
        # if a>30:
        #     break
        # a = a+1

        data.append(features)
        label.append(features) #label도 동일 데이터 사용!
        
print('Num of data : ', len(data))
        
data  = np.array(data, dtype = 'float32') #numpy 배열로 변환
label = np.array(label, dtype = 'float32')

#Train, Validation 데이터 분할(머니)
train_X, val_X, train_Y, val_Y = model_selection.train_test_split(data, label, test_size = 0.20,shuffle=True, random_state=random_state2[0]) #, stratify=label)

#PyTorch Tensor로 변환
train_X = torch.from_numpy(train_X).float()
train_Y = torch.from_numpy(train_Y).float()

val_X = np.array(val_X, dtype='float32')
val_Y = np.array(val_Y, dtype='float32')

val_X = torch.from_numpy(val_X).float()
val_Y = torch.from_numpy(val_Y).float()

train       = TensorDataset(train_X, train_Y)
validation  = TensorDataset(val_X, val_Y)

train_loader        = DataLoader(train, batch_size = 256, shuffle=True) # --> 128
validation_loader   = DataLoader(validation, batch_size = 128, shuffle=True) # --> 64

class CNN1D_LSTM(nn.Module):
    def __init__(self, num_classes=10, hidden_size=256):
        super(CNN1D_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(18, 64, 4, 2, 1),  # 채널 수 수정 (3 -> 18)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 4, 2, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        B = x.size(0)
        x = self.cnn(x)        # [B,128,feature_dim]
        x = x.transpose(1, 2)   # [B,sequence_len,128]
        out, (h_n, c_n) = self.lstm(x)
        h_last = h_n[-1]        # [B,hidden]
        x = self.fc(h_last)     # [B, num_classes]
        return x

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
            torch.save(model,'LSTM_zero_18.pt')
            print('============================\n',
            'Model save********************\n',
            '============================')        
        else:
            self.patience += 1
    
    def is_stop(self):
        return self.patience >= self.patience_limit

epoch = 1000
patience = 60
iteration = 30
factor    = 0.08
min_lr    = 1e-10

model         = CNN1D_LSTM().to(device)
early_stop    = EarlyStopping(patience)
criterion = nn.CrossEntropyLoss() #손실함수
optimizer     = optim.Adam(model.parameters(),lr=0.00001, weight_decay=0.00005)
scheduler     = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience = iteration, factor = factor, min_lr = min_lr, verbose = True)

start = time.time()

## Model Train for epoch
for epoch in range(epoch):
    train_predict = []
    train_value = []
    total_loss = 0
    running_corrects=0
    
    # Train
    model.train()
    
    for train_x, train_y  in train_loader:
        train_x, train_y = Variable(train_x).to(device), Variable(train_y).to(device)

        optimizer.zero_grad()
        output = model(train_x)

        loss = criterion(output, train_y)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.data
                
    train_epoch_loss = total_loss/len(train_loader.dataset)

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
            val_x, val_y = Variable(val_x).to(device), Variable(val_y).to(device)
            optimizer.zero_grad()
            output = model(val_x)
            loss = criterion(output, val_y)
            total_loss += loss.data
        
        val_epoch_loss = total_loss/len(validation_loader.dataset)
        
        print('====== Validation Result ========')
        print('Len_test_loader',len(validation_loader.dataset))
        print('Validation Loss: {:.9f}'.format(val_epoch_loss))
        print('=================================')
        
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