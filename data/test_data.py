import mujoco
import mujoco.viewer
import numpy as np
import time
import os

def verify_trajectory_data(data_path, model_path, num_samples_to_view=5, slow_motion_factor=10.0):
    print(f"🔍 开始验证数据: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到数据文件: {data_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型文件: {model_path}")

    # 1. 加载数据
    trajectories = np.load(data_path)
    nums_sample, steps_len, dim = trajectories.shape
    print(f"📊 数据集总览: 共 {nums_sample} 条轨迹，每条 {steps_len} 步，维度 {dim}")

    # 2. 加载模型
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # 原始的控制频率是 0.02s (50Hz)
    original_dt = 0.02 
    # 慢放后的每帧停留时间 (如果你觉得还是快，可以调大 slow_motion_factor)
    playback_dt = original_dt * slow_motion_factor

    print(f"🐢 开启慢放模式: 原定帧间隔 {original_dt}s -> 慢放帧间隔 {playback_dt:.3f}s")

    # 3. 启动可视化器
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 预留点时间让视窗加载出来
        time.sleep(2.0)
        
        for i in range(min(num_samples_to_view, nums_sample)):
            print(f"\n▶️ 正在回放第 {i+1}/{num_samples_to_view} 条轨迹...")
            traj = trajectories[i]
            
            # 初始化到起点位置
            data.qpos[:6] = traj[0]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            viewer.sync()
            
            print("  [起点准备完毕，停顿 1 秒以便观察]")
            time.sleep(1.0)
            
            # 逐帧重现矩阵中的真实数据
            for step in range(steps_len):
                if not viewer.is_running():
                    print("窗口已关闭。")
                    return
                
                # 【核心】：直接用数据覆盖物理状态
                data.qpos[:6] = traj[step]
                
                # 【关键】：这里绝对不能用 mj_step()！
                # mj_step 会引发物理前推，破坏回放的保真度。
                # mj_forward 只负责根据你强行设置的 qpos 计算所有的几何和空间坐标（例如手爪位置），用于渲染。
                mujoco.mj_forward(model, data)
                viewer.sync()
                
                # 慢放延时
                time.sleep(playback_dt)
            
            print("  [轨迹结束，停顿 1 秒]")
            time.sleep(1.0)
            
    print("\n✅ 回放验证结束。")

if __name__ == "__main__":
    # 替换为你实际的路径
    DATA_PATH = "/home/jiang/snap/GenerateTra/data/tardata.npy"
    MODEL_PATH = "/home/jiang/tools/mujoco_menagerie/universal_robots_ur5e/ur5e.xml"
    
    # 默认看前5条，慢放10倍
    verify_trajectory_data(DATA_PATH, MODEL_PATH, num_samples_to_view=5, slow_motion_factor=10.0)