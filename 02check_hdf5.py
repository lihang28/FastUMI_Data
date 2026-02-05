import h5py
import cv2
import numpy as np
import argparse
import time

def playback_hdf5(file_path, scale=0.5):
    with h5py.File(file_path, 'r') as root:
        # --- [新增] 维度检查部分 ---
        print("\n" + "="*50)
        print(f"HDF5 数据结构检查: {file_path}")
        print("-"*50)
        
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name:40} | Shape: {str(obj.shape):20} | Type: {obj.dtype}")
        
        root.visititems(print_structure)
        print("="*50 + "\n")

        # --- 初始化变量 ---
        cam_names = list(root['/observations/images'].keys())
        qpos = root['/observations/qpos'][:]
        tactile = root['/observations/tactile'][:] if '/observations/tactile' in root else None
        num_frames = len(root[f'/observations/images/{cam_names[0]}'])

        # 性能统计
        last_time = time.time()
        fps = 0

        for i in range(num_frames):
            start_frame_time = time.time()
            
            # 1. 图像处理
            frames = []
            for cam in cam_names:
                # 注意：直接读取 root[...][i] 是最耗时的部分（解压）
                img = root[f'/observations/images/{cam}'][i]
                # 如果采集时是 RGB，OpenCV 预览建议转 BGR，否则颜色会反
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                frames.append(img_bgr)
            
            combined_frame = np.hstack(frames)
            
            # 缩放
            width = int(combined_frame.shape[1] * scale)
            height = int(combined_frame.shape[0] * scale)
            small_frame = cv2.resize(combined_frame, (width, height), interpolation=cv2.INTER_NEAREST)

            # 2. 信息叠加
            # 绘制信息底色
            cv2.rectangle(small_frame, (0, 0), (280, 130), (0, 0, 0), -1)
            
            curr_q = qpos[i]
            font = cv2.FONT_HERSHEY_SIMPLEX
            f_scale = 0.45 
            
            # 计算实际 FPS
            if i % 5 == 0: # 每 5 帧更新一次 FPS 显示
                fps = 1.0 / (time.time() - last_time + 1e-6)
            last_time = time.time()

            # 文字显示
            cv2.putText(small_frame, f"Frame: {i}/{num_frames} | Actual FPS: {fps:.1f}", (10, 20), font, f_scale, (255, 255, 255), 1)
            cv2.putText(small_frame, f"X:{curr_q[0]:.3f} Y:{curr_q[1]:.3f} Z:{curr_q[2]:.3f}", (10, 45), font, f_scale, (0, 255, 255), 1)
            cv2.putText(small_frame, f"Q:{curr_q[3]:.2f},{curr_q[4]:.2f},{curr_q[5]:.2f},{curr_q[6]:.2f}", (10, 65), font, f_scale, (0, 255, 255), 1)

            if tactile is not None:
                # 显示触觉维度和首个值示例
                tac_val = tactile[i]
                cv2.putText(small_frame, f"Tac Shape: {tac_val.shape}", (10, 90), font, f_scale, (0, 255, 0), 1)
                cv2.putText(small_frame, f"Tac[0]: {tac_val[0]:.3f}", (10, 110), font, f_scale, (0, 255, 0), 1)

            # 3. 显示
            cv2.imshow("HDF5 Inspector", small_frame)
            
            # waitKey(1) 尽可能减少程序人为延迟
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key == 32: cv2.waitKey(0) 

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--scale', type=float, default=0.5)
    args = parser.parse_args()
    
    playback_hdf5(args.file, args.scale)