import torch
print(torch.__version__)              # Torch 버전 확인
print(torch.cuda.is_available())      # True 나오면 GPU 사용 가능
print(torch.cuda.get_device_name(0))  # "NVIDIA GeForce RTX 4090"
