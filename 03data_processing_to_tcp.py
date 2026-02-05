import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import cv2
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import json

# Load the configuration from the config.json file
with open('config/config.json', 'r') as config_file:
    full_config = json.load(config_file)
config = full_config["data_process_config"]

# --- ArUco 初始化修改 ---
# 获取字典
aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config["aruco_dict"]))
# 创建检测参数
parameters = cv2.aruco.DetectorParameters()
parameters.adaptiveThreshWinSizeMin = 3
parameters.adaptiveThreshWinSizeMax = 23
parameters.adaptiveThreshWinSizeStep = 10

# 创建检测器对象 (适配新版 OpenCV)
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

def get_gripper_width(img_list):
    """
    使用全局线性插值计算夹爪宽度，比原始的手动插值更稳定。
    """
    total_frames = img_list.shape[0]
    distances = []
    valid_indices = []

    m_min = config["distances"]["marker_min"]
    m_max = config["distances"]["marker_max"]
    g_max = config["distances"]["gripper_max"]

    # 1. 遍历所有帧进行检测
    for i in range(total_frames):
        if img_list[i].shape[2] == 3:
            gray = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2GRAY)
        else:
            gray = img_list[i]

        # --- 修改处：使用新版 detector.detectMarkers ---
        corners, ids, _ = detector.detectMarkers(gray)

        current_dist = None
        if ids is not None:
            marker_centers = {}
            for idx, marker_id in enumerate(ids.flatten()):
                if marker_id in [config["marker_id_0"], config["marker_id_1"]]:
                    c = np.mean(corners[idx][0], axis=0)
                    marker_centers[marker_id] = c
            
            if config["marker_id_0"] in marker_centers and config["marker_id_1"] in marker_centers:
                p0 = marker_centers[config["marker_id_0"]]
                p1 = marker_centers[config["marker_id_1"]]
                current_dist = np.linalg.norm(p0 - p1)
        
        if current_dist is not None:
            distances.append(current_dist)
            valid_indices.append(i)

    # 2. 异常处理
    if not valid_indices:
        print("Warning: No ArUco markers detected in entire episode! Defaulting to 0.")
        return np.zeros((total_frames, 1))

    # 3. 全局线性插值
    all_indices = np.arange(total_frames)
    interpolated_distances = np.interp(all_indices, valid_indices, distances)

    # 4. 映射到物理单位并归一化
    mapped_width = (interpolated_distances - m_min) / (m_max - m_min) * g_max
    normalized_width = mapped_width / g_max
    
    # 5. 截断范围 [0, 1]
    normalized_width = np.clip(normalized_width, 0.0, 1.0)

    return normalized_width

def transform_to_base_quat(x, y, z, qx, qy, qz, qw, T_base_to_local):
    rotation_local = R.from_quat([qx, qy, qz, qw]).as_matrix()
    T_local = np.eye(4)
    T_local[:3, :3] = rotation_local
    T_local[:3, 3] = [x, y, z]
    T_base_r = np.dot(T_local[:3, :3] , T_base_to_local[:3, :3] )
    x_base, y_base, z_base = T_base_to_local[:3, 3] + T_local[:3, 3]
    rotation_base = R.from_matrix(T_base_r)
    roll_base, pitch_base, yaw_base = rotation_base.as_euler('xyz', degrees=False)
    qx_base, qy_base, qz_base, qw_base = rotation_base.as_quat()
    
    return x_base, y_base, z_base, qx_base, qy_base, qz_base, qw_base, roll_base, pitch_base, yaw_base

def normalize_and_save_base_tcp_hdf5(args):
    input_file, output_file = args
    base_x, base_y, base_z = config["base_position"]["x"], config["base_position"]["y"], config["base_position"]["z"] 
    base_roll, base_pitch, base_yaw = np.deg2rad([config["base_orientation"]["roll"], config["base_orientation"]["pitch"], config["base_orientation"]["yaw"]])
    rotation_base_to_local = R.from_euler('xyz', [base_roll, base_pitch, base_yaw]).as_matrix()
    
    T_base_to_local = np.eye(4)
    T_base_to_local[:3, :3] = rotation_base_to_local
    T_base_to_local[:3, 3] = [base_x, base_y, base_z]
    
    try:
        with h5py.File(input_file, 'r') as f_in:
            action_data = f_in['action'][:]
            qpos_data = f_in['observations/qpos'][:]
            image_data = f_in['observations/images/front'][:]             
            
            tactile_data = None
            if 'observations/tactile' in f_in:
                tactile_data = f_in['observations/tactile'][:]
                if tactile_data.shape[0] != qpos_data.shape[0]:
                    #长度不一致的话可以选择报错或者截断数据，这里选择报错
                    raise KeyError("位姿数据和触觉数据长度不一致")
                    # print(f"Warning: Tactile shape mismatch {tactile_data.shape} vs {qpos_data.shape}, padding.")
                    # min_len = min(tactile_data.shape[0], qpos_data.shape[0])
                    # tactile_data = tactile_data[:min_len]
            else:
                raise KeyError(f"没有找到触觉数据: 'observations/tactile' missing in {input_file}")


            normalized_qpos = np.copy(qpos_data)

            for i in range(normalized_qpos.shape[0]):
                x, y, z, qx, qy, qz, qw = normalized_qpos[i, 0:7]
                x -= config["offset"]["x"]
                z += config["offset"]["z"]

                x_base, y_base, z_base, qx_base, qy_base, qz_base, qw_base, _, _, _ = transform_to_base_quat(x, y, z, qx, qy, qz, qw, T_base_to_local)
                ori = R.from_quat([qx_base, qy_base, qz_base, qw_base]).as_matrix()
                pos = np.array([x_base, y_base, z_base])
                pos += config["offset"]["x"] * ori[:, 2] 
                pos -= config["offset"]["z"] * ori[:, 0]
                x_base, y_base, z_base = pos
                normalized_qpos[i, :] = [x_base, y_base, z_base, qx_base, qy_base, qz_base, qw_base]

            image_data = np.array(image_data)
            normalized_width = get_gripper_width(image_data)
            
            gripper_width = normalized_width.reshape(-1, 1)
            normalized_qpos_with_gripper = np.concatenate((normalized_qpos, gripper_width), axis=1)
            
            normalized_action_with_gripper = np.copy(normalized_qpos_with_gripper)

            with h5py.File(output_file, 'w') as f_out:
                f_out.create_dataset('action', data=normalized_action_with_gripper)
                observations_group = f_out.create_group('observations')
                images_group = observations_group.create_group('images')
                
                max_timesteps = f_in['observations/images/front'].shape[0]
                cam_hight, cam_width = f_in['observations/images/front'].shape[1], f_in['observations/images/front'].shape[2]

                images_group.create_dataset(
                    'front',
                    (max_timesteps, cam_hight, cam_width, 3),
                    dtype='uint8',
                    chunks=(1, cam_hight, cam_width, 3),
                    compression='gzip',
                    compression_opts=4
                )
                images_group['front'][:] = f_in['observations/images/front'][:]
                observations_group.create_dataset('qpos', data=normalized_qpos_with_gripper)
                
                if tactile_data is not None:
                    observations_group.create_dataset('tactile', data=tactile_data)
                                
                print(f"Normalized data saved to: {output_file}")
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    input_dir = config["input_dir"]
    output_dir = config["output_tcp_dir"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_list = [
        filename for filename in os.listdir(input_dir)
        if filename.endswith('.hdf5')
    ] 
    args_list = []
    for filename in file_list:
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename)
        args_list.append((input_file, output_file))

    print(f"Starting parallel processing on {len(args_list)} files...")

    num_processes = cpu_count()
    with Pool(num_processes) as pool:
        list(
            tqdm(pool.imap_unordered(normalize_and_save_base_tcp_hdf5, args_list),
                    total=len(args_list),
                    desc="Processing files"))

    print("Processing completed.")