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
import os
print("현재 실행 중인 경로:", os.getcwd()) 

USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)

device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('학습을 진행하는 기기:',device)

#데이터 셔플링??
random_state2 = np.random.randint(42,size=1)
print("Random State :", random_state2)

##### Define data path #####
path = r"C:\Users\seoze\Desktop\KSAS2025\data\csv\pqr"
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
        
        
        if df.shape == (12, 100):
            features = df.iloc[:, :].values
        else:  
            features = df.iloc[:100, :12].values.T  
        
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

class AE_1DCNN(nn.Module):
    def __init__(self):
        super(AE_1DCNN, self).__init__()

        # Encoder 정의
        self.Encoder = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=256, kernel_size=12, padding=1, stride=1), 
            nn.BatchNorm1d(256), 
            nn.ReLU(), 
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=12, padding=1, stride=1), 
            nn.BatchNorm1d(128), 
            nn.ReLU(),  
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=12, padding=1, stride=1), 
            nn.BatchNorm1d(128), 
            nn.ReLU()  
        )

        # 잠재 벡터 변환
        self.flatten = nn.Flatten()
        self.Encoder_fc = nn.Sequential(
            nn.Linear(128 * 73, 2048), nn.ReLU(),  # 9344 -> 512
            nn.Linear(2048, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU()      # 512 -> 128
            # nn.Linear(128, 32), nn.ReLU(),       # 128 -> 32
            # nn.Linear(32, 18)                     # 32 -> 3 (잠재 벡터 크기 3으로 변환)
        )
        self.latent_activation = nn.ReLU()  # 잠재 벡터 활성화 함수
        
        #다시 원래 크기로 ㄱㄱ Fully Connected Layer
        self.Decoder_fc = nn.Sequential(
            # nn.Linear(18, 32), nn.ReLU(),         # 3 -> 32
            # nn.Linear(32, 128), nn.ReLU(),       # 32 -> 128
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 2048), nn.ReLU(),      # 128 -> 512
            nn.Linear(2048, 128 * 73)              # 512 -> 1312
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(128, 73))

        # Decoder 정의
        self.Decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=128, out_channels=128, kernel_size=12, padding=1, stride=1), 
            nn.BatchNorm1d(128), 
            nn.ReLU(),  # Input (41) -> Output (44)
            nn.ConvTranspose1d(in_channels=128, out_channels=256, kernel_size=12, padding=1, stride=1), 
            nn.BatchNorm1d(256), 
            nn.ReLU(),  # Input (44) -> Output (47)
            nn.ConvTranspose1d(in_channels=256, out_channels=12, kernel_size=12, padding=1, stride=1)  # Input (47) -> Output (100)
        )

    def forward(self, x):
        x = self.Encoder(x)

        # print({x.shape})
        x = self.flatten(x)

        # print({x.shape})
        
        x = self.Encoder_fc(x)
        
        z = self.latent_activation(x)

        x = self.Decoder_fc(z)
        x = self.unflatten(x)
        x = self.Decoder(x)

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
            torch.save(model,'Autoencoder_zero.pt')
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

model         = AE_1DCNN().to(device)
early_stop    = EarlyStopping(patience)
criterion_MAE = nn.L1Loss()
criterion_MSE = nn.MSELoss()
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

        loss = criterion_MSE(output, train_y)
        
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
            loss = criterion_MSE(output, val_y)
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