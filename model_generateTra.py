import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import numpy as np
from torch.utils.data import Dataset,DataLoader

    
block_size = 16
batch_size = 32
val_iter = 15
action_dim = 6
n_embed= 512
head_size = 256
dropout = 0.1
embedding_dim = 512
head_nums = 8
n_layer = 5
learn_rate = 3e-4
train_iter = 300



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


    def forward(self,idx,target= None,use_cache = 0,start_idx = 0):
        B,T_input,C = idx.shape
        x = idx
        x = self.projectdim2embedding(x)
        if (T_input>1):
            position_embe = self.position_embedding(torch.arange(0,T_input-1,device=idx.device).unsqueeze(0) ) 

            position_embe = position_embe.expand(B,-1,-1)

            position_embe = torch.cat((torch.zeros((B,1,embedding_dim),device=idx.device),position_embe),dim = 1)

            embedding = x+ position_embe

        else:
            # KV Cache 的自回归 Decode 阶段
            # 如果为1 那一定是kvcache的后续操作，
            pos_seq  = self.blocks[0].multiahead.multihead[0].k_cache.shape[1] - 1
            embedding = x + self.position_embedding((torch.tensor([[pos_seq]],dtype=torch.int32,device=idx.device)))      

        # x = self.block(embedding,use_cache)

        for block in self.blocks:
            embedding = block(embedding,use_cache)

        x = self.linerfinal(embedding)
        pre_y = self.projectembedding2action(x)
        
        if target is None:

            loss = None
            return pre_y,loss
        
        else:
            pred_delta = pre_y[:, 1:, :]
            
            true_delta = target - idx[:, 1:, :]
            
            # 3. 算欧氏距离 Loss (或者用你现在的 smooth_l1_loss，但对象换成 delta)
            loss = F.smooth_l1_loss(pred_delta, true_delta)
            
            # 注意：返回的依然是 pred_delta 和 loss
            return pred_delta, loss
        
    def generate(self,x,max_pre,threshold = 0.1,use_cache = 0):

        """
        max_pre 最大就为16吧 ，不要多预测了
        """
        
        target = x[:,0,:]
        if not use_cache :
            for i in range(max_pre):
                if x.shape[1]>block_size+1:
                    idxes = torch.cat([x[:, :1, :], x[:, -block_size:, :]], dim=1)
                else:
                    idxes = x

                logits,_ = self(idxes,use_cache = use_cache,start_idx = 0)
                pred_delta = logits[:,-1:,:]
                current_absolute = x[:, -1:, :] + pred_delta

                print(current_absolute)

                x = torch.cat((x,current_absolute),dim=1)
                if((torch.norm(target-current_absolute.squeeze(1),p=2,dim=-1) < threshold)).all():
                    print("已到达目标点")
                    break
        
        else:
            self.reset_cache()
            idxes = x[:,-block_size:,:]
            logits,_ = self(idxes,use_cache = use_cache,start_idx = 0)
            pred_delta = logits[:,-1:,:]
            current_absolute = x[:, -1:, :] + pred_delta

            print(f"当下位置为{current_absolute}")

            x = torch.cat((x,current_absolute),dim=1)

            for i in range(max_pre-1):
                logits,_ = self(current_absolute,use_cache = use_cache,start_idx = 0 )
                pred_delta = logits[:,-1:,:]
                current_absolute = x[:, -1:, :] + pred_delta

                x = torch.cat((x,current_absolute),dim=1)
                print(f"当下位置为{current_absolute}")
                distance = torch.norm(target - current_absolute.squeeze(1), p=2, dim=-1)
                if (distance < threshold).all():
                    print(f"Step {i}: (KV Cache) 已到达目标点, 误差距离 {distance.item():.4f}")
                    break
        print(f"目标位置为{target}")

        return x        
    
    