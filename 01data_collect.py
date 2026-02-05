import os
import json
import cv2
import h5py
import argparse
from tqdm import tqdm
from time import sleep
import numpy as np
import rospy
import csv
import threading
from collections import deque
import pandas as pd
import sys

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

# ================= 1. 配置与环境初始化 =================

# Load configuration
config_path = 'config/config.json'
if not os.path.exists(config_path):
    print(f"Error: Config file not found at {config_path}")
    sys.exit(1)

with open(config_path, 'r') as f:
    config = json.load(f)

# Task setup
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default="wooden_block")
parser.add_argument('--num_episodes', type=int, default=50)
args = parser.parse_args()

task = args.task
num_episodes = args.num_episodes
cfg = config['task_config']

# Paths Setup
data_path = os.path.join(config['device_settings']["data_dir"], "dataset" ,str(task))
os.makedirs(data_path, exist_ok=True)

IMAGE_PATH = os.path.join(data_path, 'camera/')
os.makedirs(IMAGE_PATH, exist_ok=True)

CSV_PATH = os.path.join(data_path, 'csv/')
os.makedirs(CSV_PATH, exist_ok=True)

STATE_PATH = os.path.join(data_path, 'states.csv')

if not os.path.exists(STATE_PATH):
    with open(STATE_PATH, 'w', newline='') as f:
        csv.writer(f).writerow([
            'Index', 'Start Time', 'Trajectory Timestamp', 'Frame Timestamp', \
            'Pos X', 'Pos Y', 'Pos Z', 'Q_X', 'Q_Y', 'Q_Z', 'Q_W',
        ])

VIDEO_PATH_TEMP = os.path.join(data_path, 'camera', 'temp_video_n.mp4')
TRAJECTORY_PATH_TEMP = os.path.join(data_path, 'csv', 'temp_trajectory.csv')
TIMESTAMP_PATH_TEMP = os.path.join(data_path, 'csv', 'temp_video_timestamps.csv')
FRAME_TIMESTAMP_PATH_TEMP = os.path.join(data_path, 'csv', 'frame_timestamps.csv')
TACTILE_PATH_TEMP = os.path.join(data_path, 'csv', 'temp_tactile.csv') 

# Global Variables
video_buffer = deque()
trajectory_buffer = deque()
tactile_buffer = deque()

buffer_lock = threading.Lock()
video_lock = threading.Lock()
trajectory_lock = threading.Lock()
tactile_lock = threading.Lock()

is_recording = False
start_time = 0.0
cv_bridge = CvBridge()

first_frame_timestamp = None
first_time_judger = False

# Live Monitoring Variables (UI)
current_pose = [0.0] * 7 # X, Y, Z, Qx, Qy, Qz, Qw
traj_data_received = False

# ================= 2. ROS Callbacks (统一使用 ROS 时间) =================

def video_callback(msg):
    global first_frame_timestamp, first_time_judger
    if not is_recording: return 
    
    # 【统一时钟】使用 ROS 接收时间
    timestamp = rospy.Time.now().to_sec()
    
    try:
        frame = cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with buffer_lock:
            if timestamp > start_time:
                video_buffer.append((frame, timestamp))
                if first_time_judger:
                    first_frame_timestamp = timestamp
                    first_time_judger = False
    except Exception as e:
        print(f"Video Error: {e}")

def trajectory_callback(msg):
    global current_pose, traj_data_received
    
    # 1. 解析数据 (兼容 Odometry 和 PoseStamped)
    if isinstance(msg, Odometry):
        pose = msg.pose.pose
    elif isinstance(msg, PoseStamped):
        pose = msg.pose
    else:
        return

    # 2. 更新 UI 显示用的实时变量
    current_pose = [
        pose.position.x, pose.position.y, pose.position.z,
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
    ]
    traj_data_received = True

    if not is_recording: return

    # 【统一时钟】忽略硬件时间戳，使用 ROS 接收时间
    timestamp = rospy.Time.now().to_sec()

    with buffer_lock:
        if timestamp > start_time:
            # Buffer 存储: [ROS_Time, X, Y, Z, Qx, Qy, Qz, Qw]
            trajectory_buffer.append([timestamp] + current_pose)

def tactile_callback(msg):
    # msg.data 原始结构: [Sender_Time, V1, V2 ... V12] (共13个float)
    if not is_recording: return
    
    # 【统一时钟】使用 ROS 接收时间
    timestamp = rospy.Time.now().to_sec()
    
    with buffer_lock:
        if timestamp > start_time:
            # Buffer 存储: [ROS_Time, Sender_Time, V1 ... V12]
            # msg.data 转为 list 后拼接
            tactile_buffer.append([timestamp] + list(msg.data))

# ================= 3. 数据写入线程 =================

def write_video(writer, ts_writer):
    frame_index = 0
    while not rospy.is_shutdown():
        with buffer_lock:
            if video_buffer:
                frame, timestamp = video_buffer.popleft()
                writer.write(frame)
                ts_writer.writerow([frame_index, timestamp])
                frame_index += 1
            elif not is_recording and not video_buffer:
                break
        sleep(0.001)

def write_trajectory(writer):
    while not rospy.is_shutdown():
        with buffer_lock:
            if trajectory_buffer:
                data = trajectory_buffer.popleft()
                # data: [ROS_Time, X, Y, Z, Qx, Qy, Qz, Qw]
                writer.writerow(data)
            elif not is_recording and not trajectory_buffer:
                break
        sleep(0.001)

def write_tactile(writer):
    while not rospy.is_shutdown():
        with buffer_lock:
            if tactile_buffer:
                data = tactile_buffer.popleft()
                # data: [ROS_Time, Sender_Time, V1 ... V12]
                writer.writerow(data)
            elif not is_recording and not tactile_buffer:
                break
        sleep(0.001)

# ================= 4. 控制逻辑 (UI & Loop) =================

def control_loop(episode_idx):
    global is_recording, start_time, first_time_judger
    
    window_name = "Data Collection Control"
    cv2.namedWindow(window_name)

    print(f"\n=== Episode {episode_idx + 1}/{num_episodes} Ready ===")
    
    # --- 阶段 1: 等待开始 (Ready) ---
    while not rospy.is_shutdown():
        info_img = np.zeros((300, 600, 3), dtype=np.uint8)
        
        cv2.putText(info_img, f"Episode {episode_idx+1}: READY", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(info_img, "Press [SPACE] to START", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # 实时位姿状态检查
        c_p = current_pose
        if traj_data_received:
            cv2.putText(info_img, "Pose Data: OK", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        else:
            cv2.putText(info_img, "Pose Data: NO DATA!", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        cv2.putText(info_img, f"Pos: {c_p[0]:.3f}, {c_p[1]:.3f}, {c_p[2]:.3f}", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.imshow(window_name, info_img)
        
        key = cv2.waitKey(10) & 0xFF
        if key == 32: # Space bar
            break
            
    if rospy.is_shutdown(): 
        cv2.destroyWindow(window_name)
        return False

    # --- 阶段 2: 录制中 (Recording) ---
    is_recording = True
    start_time = rospy.Time.now().to_sec()
    first_time_judger = True
    print(f"Episode {episode_idx + 1} STARTED.")

    # 准备文件
    video_writer = cv2.VideoWriter(VIDEO_PATH_TEMP.replace("_n", f"_{episode_idx}"), 
                                   cv2.VideoWriter_fourcc(*'mp4v'), 60, 
                                   (cfg['cam_width'], cfg['cam_height']))
    
    f_traj = open(TRAJECTORY_PATH_TEMP, 'w', newline='')
    f_time = open(TIMESTAMP_PATH_TEMP, 'w', newline='')
    f_tact = open(TACTILE_PATH_TEMP, 'w', newline='')

    writer_traj = csv.writer(f_traj)
    writer_time = csv.writer(f_time)
    writer_tact = csv.writer(f_tact)

    # Header 仅供 Pandas 读取参考，不影响写入逻辑
    # Traj CSV: [ROS_Time, X, Y, Z, Qx, Qy, Qz, Qw]
    # Tactile CSV: [ROS_Time, Sender_Time, V1...V12]
    # 我们不在 CSV 中写 header，直接写数据，方便后续 Pandas read_csv(header=None)

    v_thread = threading.Thread(target=write_video, args=(video_writer, writer_time))
    t_thread = threading.Thread(target=write_trajectory, args=(writer_traj,))
    tac_thread = threading.Thread(target=write_tactile, args=(writer_tact,))
    
    v_thread.start()
    t_thread.start()
    tac_thread.start()

    aborted = False
    while not rospy.is_shutdown():
        info_img = np.zeros((200, 600, 3), dtype=np.uint8)
        cv2.putText(info_img, f"Episode {episode_idx+1}: RECORDING...", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(info_img, "Press [SPACE] to STOP", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.imshow(window_name, info_img)
        
        if cv2.waitKey(10) & 0xFF == 32: # Space bar to Stop
            is_recording = False
            break
    
    if rospy.is_shutdown(): 
        is_recording = False
        aborted = True

    # --- 阶段 3: 清理 ---
    v_thread.join()
    t_thread.join()
    tac_thread.join()

    video_writer.release()
    f_traj.close()
    f_time.close()
    f_tact.close()
    cv2.destroyWindow(window_name)

    return not aborted

# ================= 5. 主程序 =================

if __name__ == "__main__":
    rospy.init_node('data_recorder', anonymous=True)

    # Subscribers
    rospy.Subscriber(cfg['ros']['video_topic'], Image, video_callback)
    
    # 尝试订阅两种位姿类型
    rospy.Subscriber(cfg['ros']['trajectory_topic'], Odometry, trajectory_callback)
    rospy.Subscriber(cfg['ros']['trajectory_topic'], PoseStamped, trajectory_callback)
    
    # 触觉 Topic
    rospy.Subscriber('/tactile_info', Float64MultiArray, tactile_callback)

    # 记录每个Episode的起始时间戳
    if not os.path.exists(FRAME_TIMESTAMP_PATH_TEMP):
         with open(FRAME_TIMESTAMP_PATH_TEMP, "w", newline='') as f:
             csv.writer(f).writerow(['Episode Index', 'Timestamp'])

    try:
        for episode in range(num_episodes):
            if rospy.is_shutdown(): break

            # Reset Buffers
            video_buffer.clear()
            trajectory_buffer.clear()
            tactile_buffer.clear()

            # --- Start Control Loop ---
            success = control_loop(episode)

            if not success:
                print("Recording aborted.")
                # 清理产生的垃圾文件
                for f in [TRAJECTORY_PATH_TEMP, TIMESTAMP_PATH_TEMP, TACTILE_PATH_TEMP]:
                    if os.path.exists(f): os.remove(f)
                break

            # --- Post Processing (Alignment) ---
            print("Processing and Aligning data...")
            
            # 1. 记录起始时间
            with open(FRAME_TIMESTAMP_PATH_TEMP, "a", newline='') as fts:
                csv.writer(fts).writerow([episode, first_frame_timestamp])

            if not os.path.exists(TIMESTAMP_PATH_TEMP) or os.path.getsize(TIMESTAMP_PATH_TEMP) == 0:
                print("No video data recorded, skipping.")
                continue

            # 2. 读取视频时间戳 & 降采样 (60Hz -> 20Hz)
            timestamps = pd.read_csv(TIMESTAMP_PATH_TEMP, header=None) # [FrameIdx, Timestamp]
            timestamps.columns = ['Frame Index', 'Timestamp']
            downsampled_timestamps = timestamps.iloc[::3].reset_index(drop=True)

            data_dict = {
                '/observations/qpos': [],
                '/action': [],
                '/observations/tactile': []
            }
            for cam_name in cfg['camera_names']:
                data_dict[f'/observations/images/{cam_name}'] = []

            # 3. 提取图像
            vid_path = VIDEO_PATH_TEMP.replace("_n", f"_{episode}")
            cap = cv2.VideoCapture(vid_path)
            for idx, row in tqdm(downsampled_timestamps.iterrows(), total=len(downsampled_timestamps), desc="Images"):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(row['Frame Index']))
                ret, frame = cap.read()
                if ret:
                    cv2.imwrite(os.path.join(IMAGE_PATH, f"ep{episode}_fr{idx}.jpg"), frame)
                    for cam_name in cfg['camera_names']:
                        data_dict[f'/observations/images/{cam_name}'].append(frame)
            cap.release()

            # 4. 读取 CSV (Trajectory & Tactile)
            
            # Trajectory: [ROS_Time, X, Y, Z, Qx, Qy, Qz, Qw]
            if os.path.exists(TRAJECTORY_PATH_TEMP) and os.path.getsize(TRAJECTORY_PATH_TEMP) > 0:
                traj_df = pd.read_csv(TRAJECTORY_PATH_TEMP, header=None)
                # 命名方便查找
                traj_df.columns = ['Timestamp', 'Pos X', 'Pos Y', 'Pos Z', 'Q_X', 'Q_Y', 'Q_Z', 'Q_W']
            else:
                print("Warning: Trajectory CSV is empty!")
                traj_df = pd.DataFrame(columns=['Timestamp', 'Pos X', 'Pos Y', 'Pos Z', 'Q_X', 'Q_Y', 'Q_Z', 'Q_W'])

            # Tactile: [ROS_Time, Sender_Time, V1 ... V12]
            if os.path.exists(TACTILE_PATH_TEMP) and os.path.getsize(TACTILE_PATH_TEMP) > 0:
                tac_df = pd.read_csv(TACTILE_PATH_TEMP, header=None)
                # 命名第0列为 Timestamp 用于对齐
                tac_df.rename(columns={0: 'Timestamp'}, inplace=True)
            else:
                print("没有触觉数据")
                exit(0)
                

            # 5. 对齐循环
            for idx, row in tqdm(downsampled_timestamps.iterrows(), total=len(downsampled_timestamps), desc="Aligning"):
                target_time = row['Timestamp']

                # --- 轨迹对齐 ---
                if not traj_df.empty:
                    # 最近邻查找
                    closest_idx = (np.abs(traj_df['Timestamp'] - target_time)).argmin()
                    closest_row = traj_df.iloc[closest_idx]
                    
                    # [关键]: 仅提取数据列 (X,Y,Z,Qx,Qy,Qz,Qw)，剔除时间戳
                    qpos = closest_row[['Pos X', 'Pos Y', 'Pos Z', 'Q_X', 'Q_Y', 'Q_Z', 'Q_W']].values.tolist()
                else:
                    qpos = [0.0] * 7
                
                data_dict['/observations/qpos'].append(qpos)
                data_dict['/action'].append(qpos)

                # --- 触觉对齐 ---
                if not tac_df.empty:
                    closest_t_idx = (np.abs(tac_df['Timestamp'] - target_time)).argmin()
                    closest_t_row = tac_df.iloc[closest_t_idx]
                    
                    # [关键]: 提取纯触觉数据
                    # CSV结构: 0:ROS_Time, 1:Sender_Time, 2-13:Data
                    # 取下标 2 之后的所有数据
                    tac_data = closest_t_row.values[2:].astype(float)
                    
                    # 维度保护
                    if len(tac_data) != 12:
                        # 如果维度不对，补零或截断，防止报错
                        temp = np.zeros(12)
                        l = min(len(tac_data), 12)
                        temp[:l] = tac_data[:l]
                        tac_data = temp
                    
                    data_dict['/observations/tactile'].append(tac_data)
                else:
                    data_dict['/observations/tactile'].append(np.zeros(12))

            # 6. 保存 HDF5
            idx_hdf5 = len([n for n in os.listdir(data_path) if n.endswith('.hdf5')])
            h5_path = os.path.join(data_path, f'episode_{idx_hdf5}.hdf5')
            
            with h5py.File(h5_path, 'w', rdcc_nbytes=2 * 1024 ** 2) as root:
                root.attrs['sim'] = False
                obs = root.create_group('observations')
                
                # Images
                img_grp = obs.create_group('images')
                for cam_name in cfg['camera_names']:
                    img_grp.create_dataset(cam_name, data=np.array(data_dict[f'/observations/images/{cam_name}'], dtype=np.uint8), compression='gzip')
                
                # Qpos (N, 7) - 无时间戳
                root.create_dataset('observations/qpos', data=np.array(data_dict['/observations/qpos'], dtype=np.float32))
                
                # Tactile (N, 12) - 无时间戳
                if len(data_dict['/observations/tactile']) > 0:
                    root.create_dataset('observations/tactile', data=np.array(data_dict['/observations/tactile'], dtype=np.float32))
                
                # Action (N, 7) - 无时间戳
                root.create_dataset('action', data=np.array(data_dict['/action'], dtype=np.float32))

            print(f"Saved successfully: {h5_path}")

            # 7. 清理中间文件
            print("Cleaning up temporary CSVs...")
            for f in [TRAJECTORY_PATH_TEMP, TIMESTAMP_PATH_TEMP, TACTILE_PATH_TEMP]:
                if os.path.exists(f): os.remove(f)
            print("Done.")

    except KeyboardInterrupt:
        print("\nForce Exit by User.")
    finally:
        cv2.destroyAllWindows()
        sys.exit(0)