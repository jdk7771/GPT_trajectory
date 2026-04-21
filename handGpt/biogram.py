import torch
import torch.nn as nn 
from torch.nn import functional as F
import os
#先读到训练数据

device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open("F:\Learnrobot\learn_pytorch\handGpt\input.txt", 'r', encoding='utf-8') as f:
    content = f.read()

current_dir = os.getcwd()

chars = sorted(set(content))
len_chars = len(chars)

#构建encode 和 decode


stoi = {char:i  for i ,char in enumerate(chars)}
itos = {i:char  for i ,char in enumerate(chars)}
encode = lambda a :[stoi[s] for s in a]
decode = lambda a :''.join([itos[s] for s in a])

content = encode(content)

train_data = content[:int(0.9*len(content))]
val_data = content[int(0.9*len(content)):]


train_data = torch.tensor(train_data)
val_data = torch.tensor(val_data)

#构建batch和block大小
batch_size = 4
block_size = 8

max_pre = 60
def get_batch(spilt):
    data = train_data if spilt == 'train' else val_data
    begin =  torch.randint(high=( int(len(train_data)-block_size-1)),size=(batch_size,))
    x = torch.stack([data[st:st+block_size] for st in begin] )
    y = torch.stack([data[st+1:st+block_size+1] for st in begin] )

    x,y = x.to(device),y.to(device)

    return x,y

class BigramLanuageModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)
        print(self.token_embedding_table)
    def forward(self,idx,target = None):
        logits = self.token_embedding_table(idx)
        loss = {}
        if target == None:
            loss = None
            return logits,loss
        
        B,T,C = logits.shape

        logits = logits.view((B*T,C))

        target = target.view(B*T)

        loss =  F.cross_entropy(logits,target)

        return logits,loss
    
    def generate(self,idx, max_nums):
        for i in range(max_nums):
            logits,_ = self(idx)
            logits = logits[:,-1,:]

            logits = F.softmax(logits,dim=-1)

            idx_next = torch.multinomial(logits, num_samples=1)

            idx = torch.cat((idx,idx_next),dim=1)

        return idx
    
@torch.no_grad()
def estimate_loss(val_iter):
    output = {}
    model.eval()

    losses = torch.zeros(val_iter)
    for i in range(val_iter):
        x,y = get_batch('val')

        loss,logit = model(x,y)

        losses[i] = loss.item()

    output = torch.cat((torch.tensor(losses.mean()).unsqueeze(0),losses))
    return output

iter_nums = 3000
vocab_size = len_chars


model = BigramLanuageModel(vocab_size)
model = model.to(device)

optimizer =  torch.optim.Adam(model.parameters(),3e-4)
x,y = get_batch('train')

def train():
    for i in range(iter_nums):
        optimizer.zero_grad()
        x,y = get_batch('train')
        logits, loss = model(x,y)
        loss.backward()
        optimizer.step()
        if((i%30) == 0):
            print(f"第{i}次训练的损失为{loss}")

    torch.save(model.state_dict(),os.path.join(current_dir,'bigram_model.pth'))
    print("已保存模型")
def test():

    model.load_state_dict(torch.load('bigram_model.pth'))
    model.eval()
    context = torch.zeros((1,1),dtype= torch.long, device=device)

    print(decode(model.generate(context,max_pre)[0].tolist()))


if __name__ == "__main__":

    train()

    










