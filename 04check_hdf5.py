import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import random

def verify_data(folder_path):
    # 随机选择一个处理好的文件
    files = [f for f in os.listdir(folder_path) if f.endswith('.hdf5')]
    if not files:
        print("未找到处理后的文件。")
        return
    
    target_file = random.choice(files)
    file_path = os.path.join(folder_path, target_file)
    print(f"正在检查文件: {file_path}")

    with h5py.File(file_path, 'r') as f:
        qpos = f['observations/qpos'][:]
        # 假设最后一列是夹爪宽度，前三列是 x, y, z
        gripper_width = qpos[:, -1]
        tcp_xyz = qpos[:, :3]
        
        # 如果有触觉数据，也读取一下
        has_tactile = 'observations/tactile' in f
        if has_tactile:
            tactile = f['observations/tactile'][:]

    # 创建可视化窗口
    fig = plt.figure(figsize=(15, 5))

    # 1. 绘制夹爪宽度
    ax1 = fig.add_subplot(131)
    ax1.plot(gripper_width, color='blue', label='Normalized Width')
    ax1.set_title(f'Gripper Width ({target_file})')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Width (0-1)')
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True)
    ax1.legend()

    # 2. 绘制 3D 轨迹
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot(tcp_xyz[:, 0], tcp_xyz[:, 1], tcp_xyz[:, 2], label='TCP Path', color='red')
    ax2.scatter(tcp_xyz[0, 0], tcp_xyz[0, 1], tcp_xyz[0, 2], color='green', label='Start') # 起点
    ax2.set_title('TCP Trajectory (Base Frame)')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.legend()

    # 3. 绘制触觉数据 (如果存在)
    ax3 = fig.add_subplot(133)
    if has_tactile:
        # 假设触觉数据是多维的，取其均值或前几维观察趋势
        ax3.plot(np.mean(tactile, axis=(1, 2)) if tactile.ndim > 2 else tactile)
        ax3.set_title('Tactile Signals (Mean)')
    else:
        ax3.text(0.5, 0.5, 'No Tactile Data', ha='center')
    ax3.set_title('Tactile Data')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 指向你生成后的文件夹
    PROCESSED_DIR = "./dataset/wooden_block_tcp_with_gripper"
    verify_data(PROCESSED_DIR)