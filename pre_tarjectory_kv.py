import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import numpy as np
from torch.utils.data import Dataset,DataLoader


device = 'cuda' if torch.cuda.is_available else 'cpu'


def denormalize(norm_data, d_min, d_max):
    """将 [-1, 1] 的网络输出还原为真实的物理弧度"""
    d_range = d_max - d_min + 1e-8
    return (norm_data + 1.0) / 2.0 * d_range + d_min



file_path = os.getcwd()
data_path = os.path.join(file_path,"data/tardata.npy")

if not os.path.exists(data_path):
    raise FileNotFoundError("找不到数据集")

data = np.load (data_path)
##归一化 数据处理   
norm_params = np.load('data/norm_params.npy', allow_pickle=True).item()
data_min = norm_params['min']
data_max = norm_params['max']
data_range = data_max - data_min + 1e-8
data_normalized = (data - data_min) / data_range * 2 - 1

data = torch.tensor(data_normalized,dtype=torch.float32,device=device)

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
            position_embe = self.position_embedding(torch.arange(0,T_input-1,device=device).unsqueeze(0) ) 

            position_embe = position_embe.expand(B,-1,-1)

            position_embe = torch.cat((torch.zeros((B,1,embedding_dim),device=device),position_embe),dim = 1)

            embedding = x+ position_embe

        else:
            # KV Cache 的自回归 Decode 阶段
            # 如果为1 那一定是kvcache的后续操作，
            pos_seq  = self.blocks[0].multiahead.multihead[0].k_cache.shape[1] - 1
            embedding = x + self.position_embedding((torch.tensor([[pos_seq]],dtype=torch.int32,device=device)))      

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
            
        logit,loss = model(x,y,use_cache = 0,start_idx = 0)
        
        loss.backward()
        
        optimizer.step()
        
        if (i+1) % 30 == 0:
            # 打印当前进度
            val_loss = estimate_loss()['val'][0].item()
            train_loss = estimate_loss()['train'][0].item()
            print(f"Step {i} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(),os.path.join(file_path,'TrajectorGenerate.pth'))
            
            
def test1():

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
    model.generate(idx,max_pre = 10,threshold=0.01,use_cache=1)

def test():
    weight_path = os.path.join(file_path,'TrajectorGenerate.pth')
    if not os.path.exists(weight_path):
        print(f"找不到权重文件: {weight_path}，请先运行 --mode train")
        return
        
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    
    print("\n" + "="*60)
    print("=== 单步预测对齐测试 (物理增量 vs 绝对坐标) ===")
    print("="*60)

    # 1. 从验证集中取出一个 Batch 的真实数据
    val_loader_iter = iter(val_loader)
    x, y, start_idx = next(val_loader_iter)
    
    # 2. 像训练时一样执行前向传播
    with torch.no_grad():
        # 注意：因为你的 forward 已经改成了返回增量，这里的 pre_y 实际上是 pred_delta
        pred_delta_batch, loss = model(x, target=y, use_cache=0, start_idx=start_idx)
    
    print(f"\n当前 Batch 的测试 Loss (Smooth L1): {loss.item():.6f}")
    
    # 3. 抽取 Batch 中的第 0 条轨迹观察
    b_idx = 0
    target_pos = x[b_idx, 0, :].cpu().numpy()
    
    print(f"\n🎯 全局 Target (终点目标):\n{target_pos}")
    print("-" * 60)
    
    # 4. 遍历这 16 步，逐帧对比
    for t in range(block_size):
        # 当前所在的真实状态 (绝对坐标)
        current_input = x[b_idx, t+1, :].cpu().numpy() 
        
        # 真实的下一步坐标 (绝对坐标)
        true_next_step = y[b_idx, t, :].cpu().numpy()
        
        # 模型的预测输出 (现在它是增量 Delta！)
        pred_delta = pred_delta_batch[b_idx, t, :].cpu().numpy()
        
        # ----------------------------------------------------
        # 【核心还原工序】：计算真实增量，并将预测增量积分为绝对坐标
        # ----------------------------------------------------
        true_delta = true_next_step - current_input           # 真实该走的距离
        pred_absolute = current_input + pred_delta            # 预测走到哪里
        
        # 计算 6-DoF 绝对空间位置的 L2 误差
        error = np.linalg.norm(true_next_step - pred_absolute)
        
        print(f"\n[第 {t+1} 步推演]")
        print(f"👉 当前输入状态 (x_t) : {np.round(current_input, 4)}")
        print(f"🔍 真实微小位移 (Δ_y) : {np.round(true_delta, 4)}")
        print(f"🤖 预测微小位移 (Δ_p) : {np.round(pred_delta, 4)}")
        print(f"✅ 真实下一步   (y_t) : {np.round(true_next_step, 4)}")
        print(f"🚀 模型还原坐标 (pre) : {np.round(pred_absolute, 4)}")
        print(f"📉 绝对坐标 L2 误差   : {error:.6f}")
        
        # 照妖镜升级：如果网络预测的增量 Δ 极其接近 0，说明它还在偷懒不肯动
        delta_magnitude = np.linalg.norm(pred_delta)
        true_magnitude = np.linalg.norm(true_delta)
        if delta_magnitude < 0.001 and true_magnitude > 0.005:
            print("   ⚠️ 警告: 网络输出的增量几乎为 0，依然在试图用【恒等映射】作弊！")

    print("\n" + "="*60)
    print("测试完毕。请重点观察【预测微小位移 (Δ_p)】是否在努力向【真实微小位移 (Δ_y)】靠拢。")

def test_generate():
    weight_path = os.path.join(file_path,'TrajectorGenerate.pth')
    if not os.path.exists(weight_path):
        print(f"找不到权重文件: {weight_path}，请先运行 --mode train")
        return
        
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    
    print("\n" + "="*60)
    print("=== 自回归 Generate 测试 (use_cache=1 开启) ===")
    print("="*60)

    # 1. 从验证集中取出一个 Batch 的真实数据
    val_loader_iter = iter(val_loader)
    x, y, start_idx = next(val_loader_iter)
    
    # 取第一条轨迹 (Batch index 0)
    b_idx = 0
    # x 的结构是 [Target, 历史16帧]
    target_pos = x[b_idx, 0:1, :]  # 取出 Target (shape: [1, 6])
    start_pos  = x[b_idx, 1:2, :]  # 取出 Start 第一帧 (shape: [1, 6])
    
    # 2. 拼装纯粹的初始输入: [Target, Start] -> shape: [1, 2, 6]
    # 这就是我们在真实环境中告诉机械臂的: "终点在哪，你现在在哪"
    gen_input = torch.cat((target_pos, start_pos), dim=0).unsqueeze(0)
    
    print(f"🎯 终点目标 (Target): {np.round(target_pos.cpu().numpy().flatten(), 4)}")
    print(f"🏁 起点位置 (Start) : {np.round(start_pos.cpu().numpy().flatten(), 4)}")
    print("-" * 60)
    print("模型开始闭眼推演 (自回归)......\n")
    
    # 3. 开启自回归推演！我们让它推演 15 步 (因为真实的轨迹长度也是 16)
    with torch.no_grad():
        # 这里调用你写的 generate，严格开启 KV Cache
        generated_traj = model.generate(gen_input, max_pre=15, threshold=0.01, use_cache=1)
    
    # 4. 评估生成的轨迹
    # generated_traj 的 shape 将会是 [1, 2+15, 6] (Target + Start + 15步预测)
    print("\n" + "="*60)
    print("=== 生成轨迹 vs 真实轨迹 对比 ===")
    print("="*60)
    
    for i in range(15):
        # 1. 提取模型自己推演出的坐标 (此时在 [-1, 1] 归一化空间)
        gen_step_norm = generated_traj[0, 2+i, :].cpu().numpy()
        true_step_norm = x[b_idx, 2+i, :].cpu().numpy() 
        
        # 2. 核心还原：转换回真实的物理弧度 (Radian)
        # 注意：这里的 data_min 和 data_max 是你在文件开头加载的全局变量
        gen_step_phys = denormalize(gen_step_norm, data_min, data_max)
        true_step_phys = denormalize(true_step_norm, data_min, data_max)
        
        # 3. 计算真实的物理偏离误差 (欧氏距离，单位: Radian)
        dist_phys = np.linalg.norm(gen_step_phys - true_step_phys)
        
        print(f"[Step {i+1}]")
        print(f"  ✅ 真实物理位姿: {np.round(true_step_phys, 4)}")
        print(f"  🤖 模型自回归到: {np.round(gen_step_phys, 4)}")
        print(f"  📉 物理漂移误差: {dist_phys:.6f} rad")
        
    # 计算最后一步的真实物理距离
    final_pos_norm = generated_traj[0, -1, :].cpu().numpy()
    final_pos_phys = denormalize(final_pos_norm, data_min, data_max)
    target_phys = denormalize(target_pos.cpu().numpy().flatten(), data_min, data_max)
    
    final_dist_phys = np.linalg.norm(target_phys - final_pos_phys)
    print("-" * 60)
    print(f"🏁 最终停止位置离目标的物理距离: {final_dist_phys:.6f} rad")
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