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
from pathlib import Path

USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)

device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('학습을 진행하는 기기:',device)

random_state2 = np.random.randint(42,size=1)
print("Random State :", random_state2)

class AE_DNN(nn.Module):
    def __init__(self):
        super(AE_DNN, self).__init__()

        self.flatten = nn.Flatten()
        # Encoder 정의
        self.Encoder = nn.Sequential(
            nn.Linear(18 * 100, 2048, bias=True), nn.ReLU(), nn.BatchNorm1d(2048),            
            nn.Linear(2048, 2048, bias=True), nn.ReLU(), nn.BatchNorm1d(2048),
            
            nn.Linear(2048, 1024, bias=True), nn.ReLU(), nn.BatchNorm1d(1024),            
            nn.Linear(1024, 1024, bias=True), nn.ReLU(), nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024, bias=True), nn.ReLU(), nn.BatchNorm1d(1024),
            nn.Linear(1024, 256, bias=True), nn.ReLU(), nn.BatchNorm1d(256),
            
            nn.Linear(256, 256, bias=True), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Linear(256, 256, bias=True), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Linear(256, 256, bias=True), nn.ReLU(), nn.BatchNorm1d(256),
                        
            nn.Linear(256, 256, bias=True)  # Latent vector
        )
        #다시 원래 크기로 ㄱㄱ
        self.Decoder = nn.Sequential(
            nn.Linear(256, 256, bias=True),  # Latent vector
            
            nn.Linear(256, 256, bias=True), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Linear(256, 256, bias=True), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Linear(256, 256, bias=True), nn.ReLU(), nn.BatchNorm1d(256),
            
            nn.Linear(256, 1024, bias=True), nn.ReLU(), nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024, bias=True), nn.ReLU(), nn.BatchNorm1d(1024),            
            nn.Linear(1024, 1024, bias=True), nn.ReLU(), nn.BatchNorm1d(1024),     
            nn.Linear(1024, 2048, bias=True), nn.ReLU(), nn.BatchNorm1d(2048),
            
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
    
model = AE_DNN().to(device)
model = torch.load('Autoencoder_DNN_zero_18.pt', map_location=device)
model.eval()

dirs = ['anormaly']
# dirs = ['dataset']

def make_test(dirs):
    data  = []
    label = []
    list_ = []
    shape_ = 0

    path = Path("C:/Users/seoze/Desktop/KSAS2025/data/csv/csv_steps_nomal_18")

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
    print('|| Num of train files : ',len(data))

    df = pd.DataFrame(list_)    
    df.to_csv('test_anormal_DNN.csv', index=False)

    data  = np.array(data, dtype = 'float32')
    label = np.array(label, dtype = 'float32')

    test_X = torch.from_numpy(data).float()
    test_Y = torch.from_numpy(label).long()

    test        = TensorDataset(test_X, test_Y)
    test_loader = DataLoader(test, batch_size = 1, shuffle=False)
    
    return test_loader
    

    
# def eval_model(test_loader):
#     result_score      = []
#     latent_list       = []
#     output_list       = []

#     with torch.no_grad():
#         for test_x, test_y in test_loader:
#             test_x = test_x.to(device)
#             output = model(test_x)
            
#             temp_residul = output - test_x
            
#             residual = torch.mean((temp_residul)**2)
#             label = test_y.cpu().item()
#             #latent_ = np.array(z.cpu())
#             #latent_ = np.squeeze(latent_, axis=0)
            
#             result_score.append((residual.item(), label))
#             #latent_list.append(latent_)   

#             # Residual
#             temp_result = np.array(temp_residul.cpu())
#             temp_result = np.squeeze(temp_result, axis=0)

#             # Sum of residual each variable
#             varList = ["a_x", "a_y", "a_z", "P", "Q", "R", "THR1", "THR2", "THR3", "THR4", "THR5", "THR6",
#                     "Phi", "Theta", "Psi", "U", "V", "W"]
#             residual_sums = [np.sum(temp_result[i][:]) for i in range(18)]
#             residual_sums.append(label)

#             output_list.append(residual_sums)
#             #print(temp_result[0][:])

#             # output_list.append([sum_a_x, sum_a_y, sum_a_z, sum_P, sum_Q, sum_R, sum_THR1, sum_THR2, sum_THR3, sum_THR4, sum_THR5, sum_THR6, sum_phi, sum_theta, sum_psi, sum_u, sum_v, sum_w, label])

#     df = pd.DataFrame(output_list)   
#     df.to_csv('hexa250224_nomal_1DCNN_result_1DCNN.csv', index=False)

#     return result_score
def eval_model(test_loader):
    results           = []
    residuals         = []
    mse_contributions = []
    
    with torch.no_grad():
        for test_x, test_y in test_loader:
            test_x = test_x.to(device)
            output = model(test_x)
            temp_residual = output - test_x
            residual = torch.mean((temp_residual) ** 2).item()
            label = test_y.cpu().item()
            
            residuals.append((residual, label))
            temp_result = (temp_residual).cpu().numpy().squeeze()
            residual_sums = [np.abs(np.sum(temp_result[i])) for i in range(18)]
            mse_contributions.append(residual_sums)
            results.append(residual_sums + [label])  
    
    df = pd.DataFrame(results)
    df.to_csv('test_results_DNN.csv', index=False)
    
    return residuals, mse_contributions

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
    threshold = np.mean(mse_values) + 2 * np.std(mse_values)  # 임계값 설정
    
    plt.figure(figsize=(10, 5))
    plt.plot(time_indices, mse_values, label='MSE', color='blue')
    plt.axhline(y=threshold, color='purple', linestyle='dashed', label='Threshold')
    plt.xlabel('Time')
    plt.ylabel('MSE')
    plt.title('Reconstruction MSE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


# def residual_plot(result_score):
#     residuals_class0 = []
#     residuals_class1 = []
#     residuals_class2 = []
#     for residual, label in result_score:
#         if label == 0:
#             residuals_class0.append(residual)
#         elif label ==1:
#             residuals_class1.append(residual)
#         else:
#             residuals_class2.append(residual)
    
#     # 인덱스 생성
#     indices_class0 = range(len(residuals_class0))
#     indices_class1 = range(len(residuals_class1))
#     indices_class2 = range(len(residuals_class2))
    
#     # 그래프 그리기
#     plt.figure(figsize=(10, 6))
#     plt.scatter(indices_class0, residuals_class0, c='blue', marker='o', label='Label 0 : Anomaly')
#     plt.scatter(indices_class1, residuals_class1, c='green', marker='o', label='Label 1 : Anomaly') #???
#     plt.scatter(indices_class2, residuals_class2, c='red', marker='x', label='Normal')
#     plt.xlabel('Sample Index')
#     plt.ylabel('MSE Residual')
#     plt.title('Train result from Autoencoder_hexa Output')
#     plt.legend()
#     plt.show()


test_loader = make_test(dirs)
# result_score = eval_model(test_loader)
# residual_plot(result_score)
residuals, mse_contributions = eval_model(test_loader)

plot_mse_contribution(mse_contributions)
plot_reconstruction_mse(residuals)