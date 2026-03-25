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

USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)

device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('학습을 진행하는 기기:',device)

random_state2 = np.random.randint(42,size=1)
print("Random State :", random_state2)

class AE_1DCNN(nn.Module):
    def __init__(self):
        super(AE_1DCNN, self).__init__()

        # Encoder 정의
        self.Encoder = nn.Sequential(
            nn.Conv1d(in_channels=18, out_channels=256, kernel_size=18, padding=1, stride=1), 
            nn.BatchNorm1d(256), 
            nn.ReLU(), 
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=18, padding=1, stride=1), 
            nn.BatchNorm1d(128), 
            nn.ReLU(),  
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=18, padding=1, stride=1), 
            nn.BatchNorm1d(128), 
            nn.ReLU()  
        )

        # 잠재 벡터 변환
        self.flatten = nn.Flatten()
        self.Encoder_fc = nn.Sequential(
            nn.Linear(128 * 55, 2048), nn.ReLU(),  # 9344 -> 512
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
            nn.Linear(2048, 128 * 55)              # 512 -> 1312
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(128, 55))

        # Decoder 정의
        self.Decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=128, out_channels=128, kernel_size=18, padding=1, stride=1), 
            nn.BatchNorm1d(128), 
            nn.ReLU(),  # Input (41) -> Output (44)
            nn.ConvTranspose1d(in_channels=128, out_channels=256, kernel_size=18, padding=1, stride=1), 
            nn.BatchNorm1d(256), 
            nn.ReLU(),  # Input (44) -> Output (47)
            nn.ConvTranspose1d(in_channels=256, out_channels=18, kernel_size=18, padding=1, stride=1)  # Input (47) -> Output (100)
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

    
model = AE_1DCNN().to(device)
model = torch.load('Autoencoder_zero_18.pt', map_location=device)
model.eval()

dirs = ['normal']

def make_test(dirs):
    data  = []
    label = []
    list_ = []
    shape_ = 0

    path = "C:/Users/seoze/Desktop/KSAS2025/data/csv/csv_steps_nomal_18/dataset"
    for i in dirs:
        files = os.listdir(path)  
        sorted_file_list = natsort.natsorted(files)
        print(i, shape_)
        for j in sorted_file_list:      
            rawdata = np.load(os.path.join(path, j))

            data.append(rawdata.T)
            label.append(shape_)
            list_.append(j[0:41])
        
        shape_ += 1

    print('|| Num of train files : ',len(data))

    df = pd.DataFrame(list_)    
    df.to_csv('hexa250224_nomal_1DCNN.csv', index=False)

    data  = np.array(data, dtype = 'float32')
    label = np.array(label, dtype = 'float32')

    test_X = torch.from_numpy(data).float()
    test_Y = torch.from_numpy(label).long()

    test        = TensorDataset(test_X, test_Y)
    test_loader = DataLoader(test, batch_size = 1, shuffle=False)
    
    return test_loader
    

    
def eval_model(test_loader):
    result_score      = []
    latent_list       = []
    output_list       = []

    with torch.no_grad():
        for test_x, test_y in test_loader:
            test_x = test_x.to(device)
            output = model(test_x)
            
            temp_residul = output - test_x
            
            residual = torch.mean((temp_residul)**2)
            label = test_y.cpu().item()
            #latent_ = np.array(z.cpu())
            #latent_ = np.squeeze(latent_, axis=0)
            
            result_score.append((residual.item(), label))
            #latent_list.append(latent_)   

            # Residual
            temp_result = np.array(temp_residul.cpu())
            temp_result = np.squeeze(temp_result, axis=0)

            # Sum of residual each variable
            varList = ["a_x", "a_y", "a_z", "P", "Q", "R", "THR1", "THR2", "THR3", "THR4", "THR5", "THR6",
                    "Phi", "Theta", "Psi", "U", "V", "W"]
            residual_sums = [np.sum(temp_result[i][:]) for i in range(18)]
            residual_sums.append(label)

            output_list.append(residual_sums)
            #print(temp_result[0][:])

            # output_list.append([sum_a_x, sum_a_y, sum_a_z, sum_P, sum_Q, sum_R, sum_THR1, sum_THR2, sum_THR3, sum_THR4, sum_THR5, sum_THR6, sum_phi, sum_theta, sum_psi, sum_u, sum_v, sum_w, label])

    df = pd.DataFrame(output_list)   
    df.to_csv('hexa250224_nomal_1DCNN_result_1DCNN.csv', index=False)

    return result_score

def residual_plot(result_score):
    residuals_class0 = []
    residuals_class1 = []
    residuals_class2 = []
    for residual, label in result_score:
        if label == 0:
            residuals_class0.append(residual)
        elif label ==1:
            residuals_class1.append(residual)
        else:
            residuals_class2.append(residual)
    
    # 인덱스 생성
    indices_class0 = range(len(residuals_class0))
    indices_class1 = range(len(residuals_class1))
    indices_class2 = range(len(residuals_class2))
    
    # 그래프 그리기
    plt.figure(figsize=(10, 6))
    plt.scatter(indices_class0, residuals_class0, c='blue', marker='o', label='Label 0 : Anomaly')
    plt.scatter(indices_class1, residuals_class1, c='green', marker='o', label='Label 1 : Anomaly')
    plt.scatter(indices_class2, residuals_class2, c='red', marker='x', label='Normal')
    plt.xlabel('Sample Index')
    plt.ylabel('MSE Residual')
    plt.title('Train result from Autoencoder_hexa Output')
    plt.legend()
    plt.show()


test_loader = make_test(dirs)
result_score = eval_model(test_loader)
residual_plot(result_score)