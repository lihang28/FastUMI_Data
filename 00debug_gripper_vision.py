# 检查夹爪tag的距离，生成一个视频
import h5py
import cv2
import numpy as np
import json
import os
from tqdm import tqdm

# ================= 配置 =================
HDF5_PATH = "./dataset/03after_tcp_with_gripper/episode_0.hdf5" 
CONFIG_PATH = "config/config.json"
OUTPUT_VIDEO = "debug_gripper_view.mp4"
# =======================================

def debug_gripper():
    # 1. 读取配置
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: 配置文件不存在 {CONFIG_PATH}")
        return
        
    with open(CONFIG_PATH, 'r') as f:
        cfg = json.load(f)
    
    dp_cfg = cfg['data_process_config']
    aruco_dict_name = dp_cfg["aruco_dict"]
    
    # --- 新版 OpenCV ArUco API 初始化 ---
    # 获取字典
    dictionary_id = getattr(cv2.aruco, aruco_dict_name)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
    # 创建检测器参数
    parameters = cv2.aruco.DetectorParameters()
    # 创建检测器对象 (这是解决 AttributeError 的关键)
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    # 获取关键参数
    id0 = dp_cfg["marker_id_0"]
    id1 = dp_cfg["marker_id_1"]
    min_px = dp_cfg["distances"]["marker_min"]
    max_px = dp_cfg["distances"]["marker_max"]
    
    print(f"DEBUG INFO: Looking for IDs {id0} & {id1}")
    print(f"DEBUG INFO: Config Pixel Range [{min_px}, {max_px}]")

    # 2. 读取 HDF5
    if not os.path.exists(HDF5_PATH):
        print(f"Error: 文件不存在 {HDF5_PATH}")
        return

    with h5py.File(HDF5_PATH, 'r') as f:
        if 'observations/images/front' in f:
            images = f['observations/images/front'][:]
        else:
            print("Error: HDF5中未找到 observations/images/front")
            return
            
    # 3. 准备视频写入
    h, w, c = images[0].shape
    writer = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))
    
    raw_distances = []

    print("开始处理帧并生成调试视频...")
    for i in tqdm(range(len(images))):
        img = images[i].copy() 
        # 确保是 BGR 格式用于绘图，灰度用于检测
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # --- 使用新版 API 进行检测 ---
        corners, ids, rejected = detector.detectMarkers(gray)
        
        # 画出所有检测到的标记
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(img_bgr, corners, ids)
        
        dist_px = 0.0
        status_text = "No Marker"
        color = (0, 0, 255) # Red

        marker_centers = {}
        
        if ids is not None:
            ids_flat = ids.flatten()
            for idx, marker_id in enumerate(ids_flat):
                if marker_id in [id0, id1]:
                    c_corners = corners[idx][0]
                    center = np.mean(c_corners, axis=0).astype(int)
                    marker_centers[marker_id] = center
            
            if id0 in marker_centers and id1 in marker_centers:
                p1 = marker_centers[id0]
                p2 = marker_centers[id1]
                dist_px = np.linalg.norm(p1 - p2)
                
                cv2.line(img_bgr, tuple(p1), tuple(p2), (0, 255, 0), 2)
                status_text = f"TWO Markers: {dist_px:.1f}px"
                color = (0, 255, 0)
                
            elif len(marker_centers) == 1:
                mid = list(marker_centers.keys())[0]
                p1 = marker_centers[mid]
                img_center_x = w / 2
                dist_px = abs(img_center_x - p1[0]) * 2
                
                cv2.line(img_bgr, tuple(p1), (int(img_center_x), p1[1]), (0, 255, 255), 2)
                status_text = f"ONE Marker: {dist_px:.1f}px (Fallback)"
                color = (0, 255, 255)
        
        raw_distances.append(dist_px)

        # 绘制文本信息
        cv2.putText(img_bgr, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img_bgr, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(img_bgr, f"Range: {min_px}-{max_px}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        if dist_px > max_px:
             cv2.putText(img_bgr, "OVER MAX!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif dist_px < min_px and dist_px > 0:
             cv2.putText(img_bgr, "UNDER MIN!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        writer.write(img_bgr)

    writer.release()
    print(f"\n调试视频已保存至: {OUTPUT_VIDEO}")
    
    valid_dists = [d for d in raw_distances if d > 0]
    if valid_dists:
        print(f"\n统计信息 (Raw Pixels):")
        print(f"  Min: {np.min(valid_dists):.2f} | Max: {np.max(valid_dists):.2f} | Avg: {np.mean(valid_dists):.2f}")
    else:
        print("\n未检测到任何有效距离！请检查 ID 设置。")

if __name__ == "__main__":
    debug_gripper()