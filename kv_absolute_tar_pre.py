import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import numpy as np
from torch.utils.data import Dataset,DataLoader


"""
为什么训练的时候不需要这个kvcache？

generate为什么要shape-2 这个
                    不想思考了，名后天再考虑

"""
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
        
        return x,y,start_idx
    
    
    
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
                x,y,start_idx = next(loader_iter)
                
            except StopIteration:
                
                loader_iter = iter(train_loader if spilt == 'train' else val_loader)
                    
                x, y,start_idx = next(loader_iter)

            logit,loss = model(x,y,use_cache = 0,start_idx= start_idx)
            
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
n_layer = 3
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

        self.k_cache = None
        self.v_cache = None

        self.register_buffer('tril',torch.tril(torch.ones(block_size+1,block_size+1)))


    def clear_cache(self):

        self.k_cache = None
        self.v_cache = None

    def forward(self,idx,use_cache=False):
# """
# 如果用kvcahce 之后 idx的输入是一致的
# """

        key = self.key(idx)
        query = self.query(idx)
        value = self.value(idx)
        
        B,T_input,C = idx.shape

        if(not use_cache ):

            weight = query @ key.transpose(1,2) *(key.shape[-1]**-0.5)

            maske_weight = weight.masked_fill(self.tril[:T_input,:T_input] == 0,float('-inf'))

            maske_weight = F.softmax(maske_weight,dim=-1)

            maske_weight = self.dropout(maske_weight)
            
            attention  = maske_weight @ value
            
            return attention
        
        else:
            if self.k_cache == None:
                self.k_cache = key
                self.v_cache = value  
 
            else:             
                B,T,C = self.k_cache.shape

                if(T<block_size+1):
                    self.k_cache = torch.cat((self.k_cache,key),dim=1)
                    self.v_cache = torch.cat((self.v_cache,value),dim=1)
                else:
                    self.k_cache = torch.cat((self.k_cache[:,:1,:],self.k_cache[:,2:,:],key),dim=1)
                    self.v_cache = torch.cat((self.v_cache[:,:1,:],self.v_cache[:,2:,:],value),dim=1)                
            

            weight = query @ self.k_cache.transpose(1,2) *(self.k_cache.shape[-1]**-0.5)
            

            if T_input > 1:
                mask = self.tril[:T_input, :self.k_cache.shape[1]] == 0
                weight = weight.masked_fill(mask, float('-inf'))

            weight = F.softmax(weight,dim=-1)
            attention  = weight @ self.v_cache

            return attention

                

class MultiHeadAttention(nn.Module):
    def __init__(self,head_size,nums_head):
        super().__init__()
        self.multihead = nn.ModuleList(Head(head_size) for _ in range(nums_head))
        self.proj = nn.Linear(head_size*nums_head, n_embed)

    def forward(self,x,use_cache = 0):
        
        x = torch.cat([head(x,use_cache) for head in self.multihead],dim=-1)
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


    def forward(self,x,use_cache = 0):
        x = self.multiahead(self.layern1(x),use_cache) + x

        x = self.feedforward(self.layern2(x)) + x

        return x
    

class PreTrajectmodel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.max_sqe_stpes = 100
        self.projectdim2embedding = nn.Linear(action_dim,embedding_dim)
        self.position_embedding = nn.Embedding(self.max_sqe_stpes,embedding_dim)
        # self.block = nn.Sequential(*[Block(head_nums,embedding_dim) for _ in range(n_layer)])
        "Sequential 不能输入两个参数"
        self.blocks = nn.ModuleList([Block(head_nums,embedding_dim) for _ in range(n_layer)] )
        self.linerfinal = nn.Linear(embedding_dim,embedding_dim)
        
        self.projectembedding2action = nn.Linear(embedding_dim,action_dim)

        
    def reset_cache(self):
        for block in self.blocks:
            for head in block.multiahead.multihead:
                head.clear_cache()


    def forward(self,x,target= None,use_cache = 0,start_idx = 0):
        B,T_input,C = x.shape
        
        x = self.projectdim2embedding(x)
        if (T_input>1):
            position_embe = self.position_embedding(torch.arange(0,T_input-1,device=device).unsqueeze(0) + torch.as_tensor(start_idx,device=device)) 
            position_embe = torch.cat((torch.zeros((B,1,embedding_dim),device=device),position_embe),dim = 1)

            embedding = x+ position_embe

        else:
            # KV Cache 的自回归 Decode 阶段
            # 此时传入的 T=1 纯粹是当前步的轨迹点，必须叠加正确的时间步
            pos_seq = torch.arange(0, 1, device=device).unsqueeze(0) + torch.as_tensor(start_idx, device=device)
            embedding = x + self.position_embedding(pos_seq)        
        # x = self.block(embedding,use_cache)

        for block in self.blocks:
            embedding = block(embedding,use_cache)

        x = self.linerfinal(embedding)
        pre_y = self.projectembedding2action(x)
        
        if target is None:
            preyloss = pre_y
            loss = None
            return preyloss,loss
        else:
            preyloss =  pre_y[:,1:,:]
            loss = F.mse_loss(preyloss,target)
            return preyloss,loss
        
    def generate(self,x,max_pre,threshold = 0.1,use_cache = 0):
        
        target = x[:,0,:]
        if not use_cache :
            for i in range(max_pre):
                if x.shape[1]>block_size+1:
                    idxes = torch.cat([x[:, :1, :], x[:, -block_size:, :]], dim=1)
                else:
                    idxes = x

                logits,_ = self(idxes,use_cache = use_cache,start_idx = 0)
                current = logits[:,-1:,:]

                print(current)

                x = torch.cat((x,current),dim=1)
                if((torch.norm(target-current.squeeze(1),p=2,dim=-1) < threshold)).all():
                    print("已到达目标点")
                    break
        
        else:
            self.reset_cache()
            idxes = x[:,-block_size:,:]
            logits,_ = self(idxes,use_cache = use_cache,start_idx = 0)
            current = logits[:,-1:,:]
            x = torch.cat((x,current),dim=1)

            for i in range(max_pre-1):
                logits,_ = self(current,use_cache = use_cache,start_idx = x.shape[1]-2)
                current = logits[:,-1:,:]
                x = torch.cat((x,current),dim=1)
                print(f"当下位置为{current}")
                distance = torch.norm(target - current.squeeze(1), p=2, dim=-1)
                if (distance < threshold).all():
                    print(f"Step {i}: (KV Cache) 已到达目标点, 误差距离 {distance.item():.4f}")
                    break
        print(f"目标位置为{target}")
        return x        
    
    
model = PreTrajectmodel().to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=learn_rate)

def train():
    train_loder_iter  = iter(train_loader)
    for i in range(train_iter):
        optimizer.zero_grad()
        try:
            x,y,start_idx = next(train_loder_iter)
        except StopIteration:
            train_loder_iter  = iter(train_loader)
            x,y,start_idx = next(train_loder_iter)
            
        logit,loss = model(x,y,use_cache = 0,start_idx = start_idx)
        
        loss.backward()
        
        optimizer.step()
        
        if (i+1) % 30 == 0:
            # 打印当前进度
            val_loss = estimate_loss()['val'][0].item()
            train_loss = estimate_loss()['train'][0].item()
            print(f"Step {i} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(),os.path.join(file_path,'TrajectorGenerate.pth'))
            
            
def test():

    weight_path = os.path.join(file_path,'TrajectorGenerate.pth')
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    """
    start:  [ 0.83339864, -0.5571846,  -0.34901676,  0.70504457, -0.51074445,  0.48726186]
    end  : [ 0.91539425 -0.7622603  -0.41337305  0.8251397  -0.48477328  0.43935925]

    """
    start = torch.tensor([ 0.83339864, -0.5571846,  -0.34901676,  0.70504457, -0.51074445,  0.48726186],dtype= torch.float32,device=device)
    end = torch.tensor([ 0.91539425 ,-0.7622603 , -0.41337305 , 0.8251397 , -0.48477328 , 0.43935925],dtype= torch.float32,device=device)
    
    idx = torch.stack((end,start),dim=0).unsqueeze(0)
    model.generate(idx,max_pre = 50,threshold=0.01,use_cache=1)

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