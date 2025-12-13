import os
import numpy as np
import cv2
import argparse
from multiprocessing import Pool
from pathlib import Path


# def image_write(path_A, path_B, path_AB):
#     im_A = cv2.imread(str(path_A), 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
#     im_B = cv2.imread(str(path_B), 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
#     im_AB = np.concatenate([im_A, im_B], 1)
#     cv2.imwrite(str(path_AB), im_AB)


# ---------------- SAR2OPT的拼接函数 ----------------
def image_write(path_A, path_B, path_AB):
    # 1. 读取 A (SAR) 图像
    # 使用 cv2.IMREAD_UNCHANGED (或 -1) 以读取原始通道数
    im_A = cv2.imread(str(path_A), cv2.IMREAD_UNCHANGED) 
    
    # 2. 读取 B (Optical) 图像
    # 使用 cv2.IMREAD_COLOR (或 1) 确保 B 图像是 3 通道
    im_B = cv2.imread(str(path_B), cv2.IMREAD_COLOR) 
    
    # --- 通道数统一处理：将 im_A 转换为 3 通道 ---
    im_A_fixed = None
    
    if im_A is None or im_B is None:
        print(f"Warning: Could not read one or both files: A={path_A}, B={path_B}. Skipping.")
        return

    # 检查 im_A 的维度
    if len(im_A.shape) == 2:
        # 1 通道灰度图 (H, W) -> 转换为 3 通道 BGR (H, W, 3)
        im_A_fixed = cv2.cvtColor(im_A, cv2.COLOR_GRAY2BGR)
    elif im_A.shape[2] == 2:
        # 2 通道双极化图 (H, W, 2) -> 简单复制一个通道以创建 3 通道
        # 注意：这只是为了满足拼接需求。在实际训练中，如果需要 2 通道输入，
        # data_loader 需要处理这个 3 通道拼接图，然后只提取前 2 个通道。
        im_A_fixed = np.dstack((im_A[:, :, 0], im_A[:, :, 1], im_A[:, :, 0]))
    elif im_A.shape[2] == 3:
        # 已经是 3 通道 (H, W, 3)
        im_A_fixed = im_A
    else:
        # 其他不常见的通道数
        print(f"Warning: SAR image {path_A} has unsupported channels: {im_A.shape[2]}. Skipping.")
        return
    
    # 3. 检查尺寸一致性
    if im_A_fixed.shape[0] != im_B.shape[0] or im_A_fixed.shape[2] != im_B.shape[2]:
        print(f"Warning: Image dimensions mismatch (H or C): A={im_A_fixed.shape}, B={im_B.shape}. Skipping {path_AB}.")
        return

    # 4. 水平拼接 (维度 1)
    im_AB = np.concatenate([im_A_fixed, im_B], 1)
    
    # 5. 写入结果
    cv2.imwrite(str(path_AB), im_AB)
# ----------------------  拼接函数结束 ----------------------


parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str, default='../dataset/50kshoes_edges')
parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B', type=str, default='../dataset/50kshoes_jpg')
parser.add_argument('--fold_AB', dest='fold_AB', help='output directory', type=str, default='../dataset/test_AB')
parser.add_argument('--num_imgs', dest='num_imgs', help='number of images', type=int, default=1000000)
parser.add_argument('--use_AB', dest='use_AB', help='if true: (0001_A, 0001_B) to (0001_AB)', action='store_true')
parser.add_argument('--no_multiprocessing', dest='no_multiprocessing', help='If used, chooses single CPU execution instead of parallel execution', action='store_true',default=False)
args = parser.parse_args()

for arg in vars(args):
    print('[%s] = ' % arg, getattr(args, arg))

fold_A = Path(args.fold_A)
fold_B = Path(args.fold_B)
fold_AB = Path(args.fold_AB)

# 1. 检查是否存在子目录（即是否为分级结构）
splits = [p.name for p in fold_A.iterdir() if p.is_dir()]

# 2. 如果存在子目录，则正常处理分级结构
if splits:
    print(f"检测到分级目录：{splits}")
    dirs_to_process = [(sp, fold_A / sp, fold_B / sp) for sp in splits]
else:
    # 3. 如果没有子目录，则直接处理根目录（扁平结构）
    print("未检测到子目录，处理根目录中的文件。")
    # 使用一个占位符 'root' 作为 split name
    dirs_to_process = [('root', fold_A, fold_B)]

# 初始化多进程池
if not args.no_multiprocessing:
    pool=Pool()

# 遍历需要处理的目录/根目录
for sp, img_fold_A, img_fold_B in dirs_to_process:
    
    # 核心文件查找逻辑
    img_list = [p.name for p in img_fold_A.iterdir() if p.is_file()]
    if args.use_AB:
        img_list = [img_path for img_path in img_list if '_A.' in img_path]

    num_imgs = min(args.num_imgs, len(img_list))
    print('split = %s, use %d/%d images' % (sp, num_imgs, len(img_list)))
    
    # 设置输出目录
    img_fold_AB = fold_AB / sp if sp != 'root' else fold_AB
    img_fold_AB.mkdir(parents=True, exist_ok=True)
    
    print('split = %s, number of images = %d' % (sp, num_imgs))
    
    for n in range(num_imgs):
        name_A = img_list[n]
        path_A = img_fold_A / name_A
        if args.use_AB:
            name_B = name_A.replace('_A.', '_B.')
        else:
            name_B = name_A
        path_B = img_fold_B / name_B
        if path_A.is_file() and path_B.is_file():
            name_AB = name_A
            if args.use_AB:
                name_AB = name_AB.replace('_A.', '.')  # remove _A
            path_AB = img_fold_AB / name_AB
            
            if not args.no_multiprocessing:
                pool.apply_async(image_write, args=(path_A, path_B, path_AB))
            else:
                image_write(path_A, path_B, path_AB)

# 关闭并等待进程
if not args.no_multiprocessing:
    pool.close()
    pool.join()
    