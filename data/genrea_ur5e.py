import mujoco
import numpy as np
import os
import torch

# 优化打印格式
np.set_printoptions(suppress=True, precision=5)

def generate_minimum_jerk_trajectory(start, end, steps):
    """
    生成 Minimum Jerk (最小加加速度) 轨迹。
    保证起止点的速度、加速度均为 0，物理轨迹极其平滑。
    """
    t = np.linspace(0, 1, steps)
    # Minimum Jerk 多项式: tau = 10*t^3 - 15*t^4 + 6*t^5
    tau = 10 * (t**3) - 15 * (t**4) + 6 * (t**5)
    
    tau = tau[:, np.newaxis] # [steps, 1]
    trajectory = start + (end - start) * tau
    return trajectory

def generate_mujoco_data(nums_sample=5000, steps_len=60, dim=6):
    print("🚀 开始基于官方 UR5e 模型生成物理动力学数据...")
    
    # 1. 加载官方 UR5e XML 模型 (替换为你电脑上的绝对路径)
    model_path = "/home/jiang/tools/mujoco_menagerie/universal_robots_ur5e/ur5e.xml"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到 UR5e 模型文件，请检查路径: {model_path}")
        
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # 初始化数据容器
    all_trajectories = np.zeros((nums_sample, steps_len, dim), dtype=np.float32)
    
    # 2. 获取 UR5e 的 6 个关节的物理角度极限 (避免折断机械臂)
    lower_bounds = model.jnt_range[:6, 0]
    upper_bounds = model.jnt_range[:6, 1]
    # 3. 极其关键的频率设置！
    # 仿真器底层 timestep 通常是 0.002s (500Hz)
    # 如果我们的控制频率是 60 步，单纯跑 60 个 timestep 只有 0.12 秒，机械臂会因为加速度过大而飞车。
    # 我们设定整个动作持续 1.2 秒 (每个控制 step 占 0.02 秒)
    control_dt = 0.02
    physics_steps_per_control = int(control_dt / model.opt.timestep)

    print(f"⚙️ 物理 timestep: {model.opt.timestep}s | 控制周期: {control_dt}s")

    for i in range(nums_sample):
        # 随机采样合法的起点和终点 (在关节极限范围内)
        # 提示：为了避免打到地板，你可以人为缩小采样范围，例如 lower_bounds * 0.8
        q_start = np.random.uniform(lower_bounds, upper_bounds)
        q_end = np.random.uniform(lower_bounds, upper_bounds)
        
        # 规划平滑参考轨迹
        ref_trajectory = generate_minimum_jerk_trajectory(q_start, q_end, steps_len)
        
        # 重置物理引擎状态到起点
        mujoco.mj_resetData(model, data)
        data.qpos[:6] = q_start
        data.qvel[:6] = 0.0
        # 必须调一次 forward 更新内部坐标系
        mujoco.mj_forward(model, data) 
        
        simulated_trajectory = np.zeros((steps_len, dim))
        
        # 开始物理伺服控制循环
        for step in range(steps_len):
            # 下发当前 step 的目标位置给位置控制器
            data.ctrl[:6] = ref_trajectory[step]
            
            # 让物理引擎跑足够长的时间来追随这个指令 (微观物理推演)
            for _ in range(physics_steps_per_control):
                mujoco.mj_step(model, data)
                
            # 记录此时真实的物理状态 (包含了伺服延迟、惯性阻力和重力影响)
            simulated_trajectory[step] = data.qpos[:6].copy()
            
        all_trajectories[i] = simulated_trajectory
        
        if (i + 1) % 500 == 0:
            print(f"✅ 已生成 {i + 1}/{nums_sample} 条轨迹...")

    # 4. 保存为你的网络需要的格式
    save_dir = "/home/jiang/snap/GenerateTra/data"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "tardata.npy")
    
    np.save(save_path, all_trajectories)
    print(f"🎉 UR5e 真实动力学数据生成完毕！保存于: {save_path} | Shape: {all_trajectories.shape}")


    data_min = all_trajectories.min(axis=(0,1), keepdims=True)
    data_max = all_trajectories.max(axis=(0,1), keepdims=True)
    norm_params = {
        'min': data_min,
        'max': data_max,
    }
    np.save(os.path.join(save_dir, 'norm_params.npy'), norm_params)
    
if __name__ == "__main__":
    generate_mujoco_data()