import torch
import numpy as np
import os

def generate_data(nums_sample = 5000, steps_len = 60, dim  = 6):
    
    print("开始生成假数据")
    
    starts = torch.rand(size=(nums_sample,1,dim))*2 -1
    end = torch.rand(size=(nums_sample,1,dim))*2 -1
    
    steps = torch.linspace(0,1,steps=steps_len).view(1,steps_len,1)
    tarjectorys = starts + ((end - starts)*steps)
    
    noise = torch.rand(nums_sample,steps_len,dim)*0.01
    
    tarjectorys = tarjectorys + noise
    
    save_path = os.path.join(os.getcwd(),"data/tardata.npy")
    
    np.save(save_path, tarjectorys.numpy().astype(np.float32))
    
    print(f"数据已保存，shape为{tarjectorys.shape}")
    
if __name__ == "__main__":
    generate_data()

