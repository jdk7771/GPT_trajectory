import torch
import torch.nn as nn
from torch.nn import functional as F
import os
"""
最多就是按照blcoksize进行attetntion 之后输出token 然后
滑动窗口进行
这并不好，会遗忘阿
所以会有CVcahce？
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
file_path = os.getcwd()

with open(os.path.join(file_path,"input.txt"),'r') as f:
    content = f.read()
    print("数据读取成功")

data_len = len(content)

## set每次都是乱序的 所以需要排序一下

char = sorted(set(content))
len_char = len(char)

stoi = {char:i for i,char in enumerate(char)}
itos = {i:char for i,char in enumerate(char)}

encode = lambda str: [stoi[cha] for cha in str]
#合并为字符串
decode = lambda inter: ''.join([itos[inte] for inte in inter])

text = encode(content)

train_data = text[:int(0.9*data_len)]
val_data = text[int(0.9*data_len):]

#转为tensor在device中
train_data = torch.tensor(train_data).to(device)
val_data = torch.tensor(val_data).to(device)



def get_batch(spilt):
    if spilt=='train':
        curddata = train_data
    elif spilt=='val':
        curddata = val_data
    else:
        print("get_batch 输入错误")
    idxs = torch.randint(len(curddata)-block_size-1,(batch_size,))

    x = torch.stack([curddata[idx:idx+block_size] for idx in idxs])
    y = torch.stack([curddata[idx+1:idx+block_size+1] for idx in idxs])

    x = x.to(device)
    y = y.to(device)

    return x,y


@torch.no_grad()
def estimate_loss():
    model.eval()
    outputs = {}
    losses = torch.zeros(val_iter)
    for spilt in ['train','val']:
        for i in range(val_iter):
            x,y = get_batch(spilt)
            logit,loss = model(x,y)
            losses[i] = loss.item()
        outputs[spilt] = torch.cat((torch.tensor(losses.mean()).unsqueeze(0),losses))
    model.train()
    return outputs



block_size = 16
batch_size = 32

val_iter = 15
n_embed= 512
head_size = 256
dropout = 0.1
embedding_dim = 512
head_nums = 8
n_layer = 5
learn_rate = 3e-4
train_iter = 1000
max_pre = 500


class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        """
        是否需要加bias  ！！！！！！
        """
        self.key = nn.Linear(n_embed,head_size,bias=None)
        self.query = nn.Linear(n_embed,head_size,bias=None)
        self.value = nn.Linear(n_embed,head_size,bias=None)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))


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
    

class Gptmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len_char,embedding_dim)
        self.position_embedding = nn.Embedding(block_size,embedding_dim)
        self.block = nn.Sequential(*[Block(head_nums,embedding_dim) for _ in range(n_layer)])
        self.linerfinal = nn.Linear(embedding_dim,embedding_dim)
        self.linerpro = nn.Linear(embedding_dim,len_char)

    def forward(self,idx,target= None):
        B,T = idx.shape
        
        embedding = self.embedding(idx)
        embedding = embedding+self.position_embedding(torch.arange(0,T,device=device))

        x = self.block(embedding)
        x = self.linerfinal(x)
        logits = self.linerpro(x)
        if target==None:
            loss = None
            return logits,loss
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            target = target.view(B*T)

            loss = F.cross_entropy(logits,target)
            return logits,loss
        
    def generate(self,idx,max_pre):
        for i in range(max_pre):
            
            idxes = idx[:,-block_size:]
            logits,_ = self(idxes)
            logits = logits[:,-1,:]
            logits = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(logits, num_samples=1)
            idx = torch.cat((idx, idx_next),1)

        return idx        
    
    
model = Gptmodel().to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=learn_rate)

def train():
    
    for i in range(train_iter):
        optimizer.zero_grad()
        
        x,y = get_batch('train')
        
        logit,loss = model(x,y)
        
        loss.backward()
        
        optimizer.step()
        
        if(i%30 ==0):
            print(f"第{i}次训练的loss为{estimate_loss()['train'][0]}")
            
    torch.save(model.state_dict(),os.path.join(file_path,'gpt.pth'))
            
            
def test():
    
    weight = torch.load('gpt.pth',weights_only=1)
    
    model.load_state_dict(weight)
    
    model.eval()
    
    context = torch.zeros((1,1),dtype = torch.long,device=device)
    
    print(decode(model.generate(context,max_pre)[0].tolist()))
    
    with open('output.txt','w') as f:
        f.write(decode(model.generate(context,max_pre)[0].tolist()))
        
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
    input('')
    main()