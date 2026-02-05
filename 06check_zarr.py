import zarr
import cv2
import numpy as np
import os
import sys
from imagecodecs_numcodecs import register_codecs
register_codecs()

def visualize_zarr(zarr_path):
    try:
        if zarr_path.endswith('.zip'):
            store = zarr.ZipStore(zarr_path, mode='r')
            root = zarr.open(store)
        else:
            root = zarr.open(zarr_path, mode='r')
    except Exception as e:
        print(f"无法打开 Zarr 文件: {e}")
        return

    data_group = root['data']
    episode_ends = root['meta']['episode_ends'][:]
    
    cv2.namedWindow('FastUMI Monitor', cv2.WINDOW_NORMAL)
    
    start_idx = 0
    for ep_idx, end_idx in enumerate(episode_ends):
        print(f"\n\n>>> 正在播放 Episode {ep_idx} | 总帧数: {end_idx - start_idx}")
        print("-" * 80)
        # 终端表头
        header = f"{'Frame':<8} | {'Gripper':<8} | {'Pos (x, y, z)':<25} | {'Tactile (first 3)':<20}"
        print(header)
        print("-" * 80)

        for i in range(start_idx, end_idx):
            # 1. 提取图像
            img = data_group['camera0_rgb'][i]
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # 2. 提取数据
            pos = data_group['robot0_eef_pos'][i]
            gripper = data_group['robot0_gripper_width'][i][0]
            
            tactile_str = "N/A"
            if 'robot0_tactile' in data_group:
                # 为了终端简洁，这里只取触觉数据的前3个维度演示
                t_val = data_group['robot0_tactile'][i]
                tactile_str = np.array2string(t_val[:3], precision=2, separator=',')

            # 3. 终端动态刷新显示
            # \r 让光标回到行首，实现动态更新
            sys.stdout.write(
                f"\r{i:<8} | {gripper:<8.4f} | {np.array2string(pos, precision=3):<25} | {tactile_str:<20}"
            )
            sys.stdout.flush()

            # 4. 显示纯净画面
            cv2.imshow('FastUMI Monitor', img_bgr)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                print("\n\n用户退出。")
                return
            elif key == ord('s'):
                print(f"\n跳过 Episode {ep_idx}")
                break
        
        start_idx = end_idx
        print("\n" + "-" * 80)

    cv2.destroyAllWindows()
    print("\n所有数据播放完毕。")

if __name__ == "__main__":
    ZARR_PATH = "./dataset/wooden_block_train_data.zarr.zip"
    visualize_zarr(ZARR_PATH)