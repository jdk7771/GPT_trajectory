import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import torch
import sys

if torch.cuda.is_available():
    torch.cuda.synchronize()

# 导入你的网络模型图纸
sys.path.append('/home/jiang/snap/GenerateTra/')
from model_generateTra import PreTrajectmodel

# 打印配置
np.set_printoptions(suppress=True, precision=4)

#归一化与反归一化

def normalize(data_np, d_min, d_max):
    d_range = d_max - d_min + 1e-8
    return (data_np - d_min) / d_range * 2.0 - 1.0

def denormalize(norm_data_np, d_min, d_max):
    d_range = d_max - d_min + 1e-8
    return (norm_data_np + 1.0) / 2.0 * d_range + d_min


device = 'cuda' if torch.cuda.is_available() else "cpu"
print(f"🚀 使用设备: {device} 加载神经网络...")

# 加载保存的极值 (确保路径和训练时一致)
norm_path = "/home/jiang/snap/GenerateTra/data/norm_params.npy"
norm_params = np.load(norm_path, allow_pickle=True).item()
data_min = norm_params['min']
data_max = norm_params['max']

# 加载模型权重
model = PreTrajectmodel().to(device)
model.load_state_dict(torch.load("/home/jiang/snap/GenerateTra/TrajectorGenerate.pth", map_location=device))
model.eval() # 极其重要：关闭 Dropout！


start = np.array([0.7739, -0.4180, -0.3156, 0.6180, -0.5206, 0.5154])
end   = np.array([0.9153, -0.7622, -0.4133, 0.8251, -0.4847, 0.4393])

# 转换为归一化 Tensor，并拼装成模型需要的输入形状 [B=1, Seq=2, Dim=6] -> [Target, Start]
start_norm = normalize(start, data_min, data_max)
end_norm = normalize(end, data_min, data_max)

input_tensor = torch.cat((
    torch.tensor(end_norm, dtype=torch.float32, device=device).view(1, 1, 6),
    torch.tensor(start_norm, dtype=torch.float32, device=device).view(1, 1, 6)
), dim=1)


start_time = time.perf_counter()
with torch.no_grad():
    # 预测 60 步，和采集数据的步数对齐
    generated_norm = model.generate(input_tensor, max_pre=12, threshold=0.01, use_cache=1)

# 将预测出的整条轨迹反归一化回物理弧度
# generated_norm 的形状是 [1, 2+60, 6]，第0个是Target，第1个是Start

generated_phys = denormalize(generated_norm[0].cpu().numpy(), data_min.squeeze(0), data_max.squeeze(0))

pred_trajectory = generated_phys[1:] # 抠出 Start 和后续的预测轨迹 (共 61 帧)

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"🚀 轨迹生成与后处理总耗时: {elapsed_time:.4f} 秒")

model_path = "/home/jiang/tools/mujoco_menagerie/universal_robots_ur5e/ur5e.xml"
mj_model = mujoco.MjModel.from_xml_path(model_path)

mj_model.opt.gravity[:] = [0, 0, 0] 

mj_data = mujoco.MjData(mj_model)

control_dt = 0.02 # 神经网络的输出频率 50Hz
physics_steps_per_control = int(control_dt / mj_model.opt.timestep)

print(f"⚙️ 物理底层频率: {1/mj_model.opt.timestep}Hz | 神经网络控制频率: {1/control_dt}Hz")


smooth_q = start.copy()
alpha = 0.4


print("准备执行轨迹，请在仿真器中观察...")
mj_data.qpos[:6] = start
mj_data.qvel[:6] = 0.0
mujoco.mj_forward(mj_model, mj_data)

step_idx = 0
total_steps = len(pred_trajectory)
slow_motion_factor = 5.0


with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    # 预留 2 秒时间让你看清起始姿态
    time.sleep(1.0)
    
    while viewer.is_running():
        step_start = time.time()
        

        if step_idx < total_steps:
            target_q = pred_trajectory[step_idx]
            step_idx += 1
        else:
            target_q = pred_trajectory[-1] # 走完了就保持在终点
        

        for _ in range(physics_steps_per_control):

            smooth_q = (1 - alpha) * smooth_q + alpha * target_q
            print("----------------------------------")
            print(smooth_q)
            # 直接将角度指令下发给位置伺服器 (不需要 IK！)
            mj_data.ctrl[:6] = smooth_q
            mujoco.mj_step(mj_model, mj_data)
        
        # 3. 画面同步与真实时间对齐
        viewer.sync()
        
        time_until_next_step = control_dt*slow_motion_factor - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
            
        # 打印执行状态
        if step_idx == total_steps:
            final_err = np.linalg.norm(mj_data.qpos[:6] - end)
            print(f"轨迹执行完毕！最终关节位置误差: {final_err:.4f} rad")
            break # 执行完可选择退出或保持