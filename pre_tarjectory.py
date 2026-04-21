import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import numpy as np
from torch.utils.data import Dataset,DataLoader


device = 'cuda' if torch.cuda.is_available else 'cpu'

file_path = os.getcwd()
data_path = os.path.join(file_path,"data/tardata.npy")

if not os.path.exists(data_path):
    raise FileNotFoundError("找不到数据集")

data = np.load (data_path)
data = torch.tensor(data,dtype=torch.float32,device=device)

print(f"数据类型为{data.shape}")


data_len = len(data)
train_data = data[:int(0.9*data_len)]
val_data = data[int(0.9*data_len):]

class TrajectoryData(Dataset):
    
    def __init__(self,data,block_size):
        self.data = data
        self.block_size = block_size
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        trajectory = self.data[index]
        
        endpos = trajectory[-1]
        
        start_idx = torch.randint(0,len(trajectory) - block_size -1,(1,))
        
        x = trajectory[start_idx : start_idx + block_size]
        
        y = trajectory[start_idx + 1 : start_idx + block_size +1]
        
        x = torch.cat((endpos.unsqueeze(0) , x))
        
        return x,y
    
    
    
block_size = 16
batch_size = 32
val_iter = 15


train_dataset = TrajectoryData(train_data,block_size)
val_dataset= TrajectoryData(val_data,block_size)

train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle= True)
val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle= True)

    

@torch.no_grad()
def estimate_loss():
    model.eval()
    outputs = {}
    val_loder = iter(val_loader)
    train_loder = iter(train_loader)
    
    for spilt in ['train','val']:
        
        losses = torch.zeros(val_iter)
        loader_iter = train_loder if spilt=='train' else val_loder
        
        
        for i in range(val_iter):
            
            try:
                x,y = next(loader_iter)
                
            except StopIteration:
                
                loader_iter = iter(train_loader if spilt == 'train' else val_loader)
                    
                x, y = next(loader_iter)

            logit,loss = model(x,y)
            
            losses[i] = loss
            
        losses = torch.cat((torch.tensor(torch.mean(losses).unsqueeze(0)),losses))
        
        outputs[spilt] = losses
        
            
    model.train()
    return outputs

action_dim = 6
n_embed= 512
head_size = 256
dropout = 0.1
embedding_dim = 512
head_nums = 8
n_layer = 5
learn_rate = 3e-4
train_iter = 300
max_pre = 500


class Head(nn.Module):
    def __init__(self,head_size):
      
        super().__init__()

        self.key = nn.Linear(n_embed,head_size,bias=None)
        self.query = nn.Linear(n_embed,head_size,bias=None)
        self.value = nn.Linear(n_embed,head_size,bias=None)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer('tril',torch.tril(torch.ones(block_size+1,block_size+1)))


    def forward(self,idx):

        B,T,C = idx.shape
        
        keys = self.key(idx)
        query = self.query(idx)
        value = self.value(idx)

        weight = query @ keys.transpose(1,2) *(keys.shape[-1]**-0.5)

        maske_weight = weight.masked_fill(self.tril[:T,:T] == 0,float('-inf'))

        maske_weight = F.softmax(maske_weight,dim=-1)

        maske_weight = self.dropout(maske_weight)
        
        attention  = maske_weight @ value

        return attention
    

class MultiHeadAttention(nn.Module):
    def __init__(self,head_size,nums_head):
        super().__init__()
        self.multihead = nn.ModuleList(Head(head_size) for _ in range(nums_head))
        self.proj = nn.Linear(head_size*nums_head, n_embed)

    def forward(self,x):
        
        x = torch.cat([head(x) for head in self.multihead],dim=-1)
        x = self.proj(x)
        
        return x

class Feedforward(nn.Module):
    def __init__(self,n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed,4*n_embed),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4*n_embed,n_embed)
        )
        

    def forward(self,x):
        return self.net(x)
    

class Block(nn.Module):
    def __init__(self,head_nums,n_embed):
        super().__init__()

        self.multiahead = MultiHeadAttention(int(n_embed/head_nums),head_nums)
        self.feedforward = Feedforward(n_embed)
        self.layern1 = nn.LayerNorm(n_embed)
        self.layern2 = nn.LayerNorm(n_embed)


    def forward(self,x):
        x = self.multiahead(self.layern1(x)) + x

        x = self.feedforward(self.layern2(x)) + x

        return x
    

class PreTrajectmodel(nn.Module):
    def __init__(self):
        super().__init__()
        
        
        self.projectdim2embedding = nn.Linear(action_dim,embedding_dim)
        self.position_embedding = nn.Embedding(block_size+1,embedding_dim)
        self.block = nn.Sequential(*[Block(head_nums,embedding_dim) for _ in range(n_layer)])
        self.linerfinal = nn.Linear(embedding_dim,embedding_dim)
        
        self.projectembedding2action = nn.Linear(embedding_dim,action_dim)

        

    def forward(self,x,target= None):
        B,T,C = x.shape
        
        x = self.projectdim2embedding(x)
        embedding = x+self.position_embedding(torch.arange(0,T,device=device))

        x = self.block(embedding)
        x = self.linerfinal(x)
        pre_y = self.projectembedding2action(x)
        
        preyloss = pre_y[:,1:,:]
        
        
        if target==None:
            loss = None
            return preyloss,loss
        else:

            loss = F.mse_loss(preyloss,target)
            return preyloss,loss
        
    def generate(self,x,max_pre,threshold = 0.1):

        for i in range(max_pre):
            idxes = x[:,-block_size:]
            logits,_ = self(idxes)
            logits = logits[:,1:,:]
            x = torch.cat((x,logits),dim=1)
            if(torch.abs(logits[:,:1,:]-logits[:,-1:,:]) < threshold):
                print("已到达目标点")
                break;

        return x        
    
    
model = PreTrajectmodel().to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=learn_rate)

def train():
    train_loder_iter  = iter(train_loader)
    for i in range(train_iter):
        optimizer.zero_grad()
        try:
            x,y = next(train_loder_iter)
        except StopIteration:
            train_loder_iter  = iter(train_loader)
            x,y = next(train_loder_iter)
            
        logit,loss = model(x,y)
        
        loss.backward()
        
        optimizer.step()
        
        if i % 30 == 0:
            # 打印当前进度
            val_loss = estimate_loss()['val'][0].item()
            train_loss = estimate_loss()['train'][0].item()
            print(f"Step {i} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    torch.save(model.state_dict(),os.path.join(file_path,'TrajectorGenerate.pth'))
            
            
def test():
    pass

import argparse

def main():

    parser = argparse.ArgumentParser(description="GPT的模型训练与测试")
    parser.add_argument('--mode',type = str,default = 'test',help="选择运行模型")
    args = parser.parse_args()
    
    if(args.mode == 'train'):
        train()
    elif args.mode == 'test':
        test()
                
        
if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", device)
    main()