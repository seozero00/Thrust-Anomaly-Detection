import matplotlib.pyplot as plt
import pandas as pd
import csv
from pathlib import Path


csv_file = Path(r'C:\Users\seoze\Desktop\KSAS2025\loss_history_dnn.csv')
df = pd.read_csv(csv_file)
train_losses = df['Train Loss']
val_losses = df['Validation Loss']


plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid()
plt.show()
