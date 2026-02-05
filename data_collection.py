import os
import json
import cv2
import h5py
import argparse
from tqdm import tqdm
import numpy as np
import rospy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
import csv
import queue
import threading
import pandas as pd


# Load configuration from config.json
with open('config/config.json', 'r') as f:
    config = json.load(f)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default="wooden_block")
parser.add_argument('--num_episodes', type=int, default=2)
args = parser.parse_args()

task = args.task
num_episodes = args.num_episodes
cfg = config['task_config']

data_path = os.path.join(config['device_settings']["data_dir"], "dataset" ,str(task))
os.makedirs(data_path, exist_ok=True)

IMAGE_PATH = os.path.join(data_path, 'camera/')
os.makedirs(IMAGE_PATH, exist_ok=True)

CSV_PATH = os.path.join(data_path, 'csv/')
os.makedirs(CSV_PATH, exist_ok=True)

STATE_PATH = os.path.join(data_path, 'states.csv')

if not os.path.exists(STATE_PATH):
    with open(STATE_PATH, 'w') as csv_file2:
        csv_writer2 = csv.writer(csv_file2)
        csv_writer2.writerow([
            'Index', 'Start Time', 'Trajectory Timestamp', 'Frame Timestamp', 'Tactile Timestamp',
            'Pos X', 'Pos Y', 'Pos Z', 'Q_X', 'Q_Y', 'Q_Z', 'Q_W',
            'nf_l0', 'nf_l1', 'nf_r0', 'nf_r1',
            'tf_l0', 'tf_l1', 'tf_r0', 'tf_r1',
            'tfDir_l0', 'tfDir_l1', 'tfDir_r0', 'tfDir_r1',
        ])

VIDEO_PATH_TEMP = os.path.join(data_path, 'camera', 'temp_video_n.mp4')
TRAJECTORY_PATH_TEMP = os.path.join(data_path, 'csv', 'temp_trajectory.csv')
TIMESTAMP_PATH_TEMP = os.path.join(data_path, 'csv', 'temp_video_timestamps.csv')
FRAME_TIMESTAMP_PATH_TEMP = os.path.join(data_path, 'csv', 'frame_timestamps.csv')
TACTILE_PATH_TEMP = os.path.join(data_path, 'csv', 'temp_tactile.csv')

# Buffers for storing incoming data (Queue is thread-safe)
video_buffer = queue.Queue()
trajectory_buffer = queue.Queue()
tactile_buffer = queue.Queue()

# Initialize CvBridge for image conversion
cv_bridge = CvBridge()

# Variable to store the first frame's timestamp
first_frame_timestamp = None
first_time_judger = False

# Global flag to indicate recording status
is_recording = False


# Callback for video frames (60 Hz expected)
def video_callback(msg: Image):
    global first_frame_timestamp, first_time_judger

    frame = cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    timestamp = msg.header.stamp.to_sec()

    if start_time < timestamp:
        video_buffer.put((frame, timestamp))

        if first_time_judger:
            first_frame_timestamp = timestamp
            first_time_judger = False

# Callback for trajectory data (e.g., T265 at 200 Hz)
def trajectory_callback(msg: Odometry):
    timestamp = msg.header.stamp.to_sec()  # Ensure timestamp is in Unix format (float)

    if start_time < timestamp:
        pose = msg.pose.pose
        trajectory_buffer.put((
            timestamp, pose.position.x, pose.position.y, pose.position.z,
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w,
        ))

# Callback for tactile sensor data
def tactile_callback(msg: Float64MultiArray):
    timestamp = msg.data[0]  # Assuming the first element is the timestamp
    if start_time < timestamp:
        tactile_buffer.put(tuple(msg.data))  # Store the rest of the data


# Thread for writing video frames and timestamps
def write_video(video_writer, timestamp_writer):
    global is_recording
    frame_index = 0
    pbar = tqdm(desc='Recording Video Frames', unit=' frames')

    while is_recording or not video_buffer.empty():
        try:
            frame, timestamp = video_buffer.get(timeout=0.1)
            video_writer.write(frame)

            # Write timestamp for each frame to CSV
            timestamp_writer.writerow([frame_index, timestamp])
            frame_index += 1
            pbar.update(1)

        except queue.Empty:
            continue
    
    pbar.close()
    print(f"Video Done! Total frames: {frame_index}")


# Thread for writing trajectory data to CSV
def write_trajectory(trajectory_writer):
    global is_recording
    counter = 0

    while is_recording or not trajectory_buffer.empty():
        try:
            # Timestamp, PosX, PosY, PosZ, Q_X, Q_Y, Q_Z, Q_W = trajectory_buffer.get(timeout=0.1)
            # trajectory_writer.writerow([
            #     Timestamp, PosX, PosY, PosZ, Q_X, Q_Y, Q_Z, Q_W,
            # ])
            data = trajectory_buffer.get(timeout=0.1)
            trajectory_writer.writerow(data)
            counter += 1
        except queue.Empty:
            continue
    print(f"Trajectory Done! Total points: {counter}")


def write_tactile(tactile_writer):
    global is_recording
    counter = 0
    while is_recording or not tactile_buffer.empty():
        try:
            data = tactile_buffer.get(timeout=0.1)
            tactile_writer.writerow(data)
            counter += 1
        except queue.Empty:
            continue
    print(f"Tactile Done! Total points: {counter}")


# Main function to start recording
def start_recording(video_writer, trajectory_writer, timestamp_writer, tactile_writer):
    # Reset buffers
    global is_recording
    is_recording = True

    # Start separate threads for writing video and trajectory data
    video_thread = threading.Thread(target=write_video, args=(video_writer, timestamp_writer))
    trajectory_thread = threading.Thread(target=write_trajectory, args=(trajectory_writer,))
    tactile_thread = threading.Thread(target=write_tactile, args=(tactile_writer,))

    video_thread.start()
    trajectory_thread.start()
    tactile_thread.start()

    return video_thread, trajectory_thread, tactile_thread


if __name__ == "__main__":

    rospy.init_node('data_collection_node', anonymous=True)

    # Initialize subscribers
    start_time = float("inf")
    video_subscriber = rospy.Subscriber(
        config['task_config']['ros']['video_topic'],
        Image,
        video_callback,
        queue_size=config['task_config']['ros']['queue_size'],
    )
    trajectory_subscriber = rospy.Subscriber(
        config['task_config']['ros']['trajectory_topic'],
        Odometry, 
        trajectory_callback,
        queue_size=config['task_config']['ros']['queue_size'],
    )
    tactile_subscriber = rospy.Subscriber(
        config['task_config']['ros']['tactile_topic'],
        Float64MultiArray,
        tactile_callback,
        queue_size=config['task_config']['ros']['queue_size'],
    )

    # Initialize frame timestamp file
    with open(FRAME_TIMESTAMP_PATH_TEMP, "a", newline='') as frame_timestamp_file:
        frame_timestamp_writer = csv.writer(frame_timestamp_file)
        frame_timestamp_writer.writerow(['Episode Index', 'Timestamp'])

        for episode in range(num_episodes):
            # Video writer parameters for 60 Hz recording
            video_writer = cv2.VideoWriter(
                VIDEO_PATH_TEMP.replace("_n", f"_{episode}"),
                cv2.VideoWriter_fourcc(*'mp4v'),
                cfg['cam_fps'],
                (cfg['cam_width'], cfg['cam_height']),
            )

            # CSV for trajectory data and video timestamps
            with open(TRAJECTORY_PATH_TEMP, 'w', newline='') as trajectory_file, \
                 open(TIMESTAMP_PATH_TEMP, 'w', newline='') as timestamp_file, \
                 open(TACTILE_PATH_TEMP, 'w', newline='') as tactile_file:

                trajectory_writer = csv.writer(trajectory_file)
                timestamp_writer = csv.writer(timestamp_file)
                tactile_writer = csv.writer(tactile_file)

                # Write headers to CSV files
                trajectory_writer.writerow([
                    'Timestamp', 'Pos X', 'Pos Y', 'Pos Z', 'Q_X', 'Q_Y', 'Q_Z', 'Q_W',
                ])
                timestamp_writer.writerow(['Frame Index', 'Timestamp'])
                tactile_writer.writerow([
                    'Timestamp',
                    'nf_l0', 'nf_l1', 'nf_r0', 'nf_r1',
                    'tf_l0', 'tf_l1', 'tf_r0', 'tf_r1',
                    'tfDir_l0', 'tfDir_l1', 'tfDir_r0', 'tfDir_r1',
                ])

                first_time_judger = False

                input(f"Episode {episode + 1}/{num_episodes} ready. Press Enter to start...")

                video_buffer.queue.clear()
                trajectory_buffer.queue.clear()
                tactile_buffer.queue.clear()

                start_time = rospy.Time.now().to_sec()  # Start time
                first_time_judger = True

                video_thread, trajectory_thread, tactile_thread = start_recording(
                    video_writer, trajectory_writer, timestamp_writer, tactile_writer
                )

                input(f"Episode {episode + 1}/{num_episodes} started! Press Enter to stop...")

                is_recording = False

                # Wait for threads to finish
                video_thread.join()
                trajectory_thread.join()
                tactile_thread.join()

                frame_timestamp_writer.writerow([episode, first_frame_timestamp])
                video_writer.release()
                print(f"Episode {episode + 1}/{num_episodes} recording stopped.")
                

                # Data list preparation
                data_dict = {
                    '/observations/qpos': [],
                    '/action': [],
                    '/observations/tactile': []
                }

                for cam_name in cfg['camera_names']:
                    data_dict[f'/observations/images/{cam_name}'] = []

                # process timestamps
                timestamp_file.close() # Ensure file is written
                timestamps = pd.read_csv(TIMESTAMP_PATH_TEMP)
                downsampled_timestamps = timestamps.iloc[::3].reset_index(drop=True) # Downsample from 60Hz to 20 Hz  

                # Process video frames
                cap = cv2.VideoCapture(VIDEO_PATH_TEMP.replace("_n", f"_{episode}"))
                for idx, row in tqdm(downsampled_timestamps.iterrows(), desc='Extracting Images'):
                    frame_idx = row['Frame Index']
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if ret:
                        # filename = f"{int(frame_idx / 3)}.jpg"
                        # cv2.imwrite(os.path.join(IMAGE_PATH, filename), frame)
                        for cam_name in cfg['camera_names']:
                            data_dict[f'/observations/images/{cam_name}'].append(frame)
                cap.release()

                # Process trajectory and tactile data
                trajectory_file.close()
                tactile_file.close()
                
                trajectory = pd.read_csv(TRAJECTORY_PATH_TEMP)
                trajectory['Timestamp'] = trajectory['Timestamp'].astype(float)
                
                tactile = pd.read_csv(TACTILE_PATH_TEMP)
                tactile['Timestamp'] = tactile['Timestamp'].astype(float)

                with open(STATE_PATH, 'a', newline='') as csv_file2:
                    csv_writer2 = csv.writer(csv_file2)

                    for idx, row in tqdm(downsampled_timestamps.iterrows(), desc='Extracting States and Tactile'):
                        tj_idx = (np.abs(trajectory['Timestamp'] - row['Timestamp'])).argmin()
                        tj_row = trajectory.iloc[tj_idx]
                        pos_quat = [
                            tj_row['Pos X'], tj_row['Pos Y'], tj_row['Pos Z'],
                            tj_row['Q_X'], tj_row['Q_Y'], tj_row['Q_Z'], tj_row['Q_W']
                        ]

                        tc_idx = (np.abs(tactile['Timestamp'] - row['Timestamp'])).argmin()
                        tc_row = tactile.iloc[tc_idx]
                        tactile_data = [
                            tc_row['nf_l0'], tc_row['nf_l1'], tc_row['nf_r0'], tc_row['nf_r1'],
                            tc_row['tf_l0'], tc_row['tf_l1'], tc_row['tf_r0'], tc_row['tf_r1'],
                            tc_row['tfDir_l0'], tc_row['tfDir_l1'], tc_row['tfDir_r0'], tc_row['tfDir_r1'],
                        ]

                        data_dict['/observations/qpos'].append(pos_quat)
                        data_dict['/action'].append(pos_quat)
                        data_dict['/observations/tactile'].append(tactile_data)

                        csv_writer2.writerow([
                            idx, start_time, tj_row['Timestamp'], row['Timestamp'], tc_row['Timestamp']] + pos_quat + tactile_data
                        )

                max_timesteps = len(data_dict['/observations/qpos'])

                # idx = len([name for name in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, name))])
                # dataset_path = os.path.join(data_path, f'episode_{idx}.hdf5')
                hdf5_count = len([name for name in os.listdir(data_path) if name.endswith('.hdf5')])
                dataset_path = os.path.join(data_path, f'episode_{hdf5_count}.hdf5')
                os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

                # Save the data
                with h5py.File(dataset_path, 'w', rdcc_nbytes=2 * 1024 ** 2) as root:
                    root.attrs['sim'] = False
                    obs = root.create_group('observations')
                    image_grp = obs.create_group('images')

                    for cam_name in cfg['camera_names']:
                        image_grp.create_dataset(
                            cam_name,
                            data=np.array(data_dict[f'/observations/images/{cam_name}'], dtype=np.uint8),
                            compression='gzip',
                            compression_opts=4
                        )
                    root.create_dataset(
                        'observations/qpos',
                        data=np.array(data_dict['/observations/qpos']),
                        dtype=np.float32,
                    )
                    root.create_dataset(
                        'action',
                        data=np.array(data_dict['/action']),
                        dtype=np.float32,
                    )
                    root.create_dataset(
                        'observations/tactile',
                        data=np.array(data_dict['/observations/tactile']),
                        dtype=np.float32,  
                    )

    print("All episodes completed successfully!")