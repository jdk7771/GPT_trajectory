import numpy as np
import os

path_file = os.getcwd()
data = np.load(os.path.join(path_file,"data/tardata.npy"))

# data shape: (nums_sample, steps_len, 6)

# 同时沿着 第0维(样本) 和 第1维(时间步) 求最大/最小值
max_joints = np.max(data, axis=(0, 1))
min_joints = np.min(data, axis=(0, 1))

print("=== 6个关节的物理极限范围 ===")
# 遍历打印，格式化输出更清晰
for i in range(data.shape[-1]):
    print(f"关节 {i+1}: 最小值 = {min_joints[i]:.5f}, 最大值 = {max_joints[i]:.5f}")

