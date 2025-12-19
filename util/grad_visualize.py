import numpy as np
import cv2
import matplotlib.pyplot as plt

def visualize_edge_comparison(image_path, downsample_factor=1, thresholds=[(50, 150), (100, 200), (30, 100)]):
    """
    对比原始 Sobel 方法与 改进的 Canny 去噪方法。
    """
    # 1. 加载图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：无法加载图像 {image_path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 降采样处理
    if downsample_factor > 1:
        h, w = img_gray.shape
        img_gray = cv2.resize(img_gray, (w // downsample_factor, h // downsample_factor), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.resize(img_rgb, (w // downsample_factor, h // downsample_factor), interpolation=cv2.INTER_LINEAR)

    img_float = img_gray.astype(np.float32) / 255.0

    # 设置画布：第一行原始图，后面每行是一个参数下的对比
    rows = len(thresholds) + 1
    plt.figure(figsize=(12, 4 * rows))

    # --- 第一行：原图 ---
    plt.subplot(rows, 3, 1)
    plt.title("Original Image")
    plt.imshow(img_rgb)
    plt.axis('off')

    for i, (low, high) in enumerate(thresholds):
        row_idx = i + 2
        
        # --- 方法 A: 您的 Sobel 二次提取 (未显式去噪) ---
        # 第一次梯度
        sobelx = cv2.Sobel(img_float, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_float, cv2.CV_32F, 0, 1, ksize=3)
        mag1 = np.sqrt(sobelx**2 + sobely**2)
        mag1 = mag1 / (mag1.max() + 1e-8)
        # 第二次梯度 (您提到的思路)
        sobelx2 = cv2.Sobel(mag1, cv2.CV_32F, 1, 0, ksize=3)
        sobely2 = cv2.Sobel(mag1, cv2.CV_32F, 0, 1, ksize=3)
        mag2 = np.sqrt(sobelx2**2 + sobely2**2)
        mag2 = mag2 / (mag2.max() + 1e-8)
        
        # 使用 high/255 作为阈值模拟
        edge_sobel = (mag2 > (high / 255.0)).astype(np.uint8)

        # --- 方法 B: 改进的去噪 Canny (推荐) ---
        # 1. 中值滤波去噪 (这对遥感图像的斑点噪声非常有效)
        denoised = cv2.medianBlur(img_gray, 5) 
        # 2. Canny 算子 (内置非极大值抑制和滞后阈值)
        edge_canny = cv2.Canny(denoised, low, high)

        # --- 可视化 ---
        # 原始 Sobel 结果
        plt.subplot(rows, 3, 3 * i + 4)
        plt.title(f"Double Sobel (No Denoise)")
        plt.imshow(edge_sobel, cmap='gray')
        plt.axis('off')

        # Canny 去噪结果
        plt.subplot(rows, 3, 3 * i + 5)
        plt.title(f"MedianBlur + Canny (Low:{low} High:{high})")
        plt.imshow(edge_canny, cmap='gray')
        plt.axis('off')

        # 叠加对比 (将边缘叠在原图上，看是否对齐)
        overlay = img_rgb.copy()
        overlay[edge_canny > 0] = [255, 0, 0] # 红色边缘
        plt.subplot(rows, 3, 3 * i + 6)
        plt.title("Canny Overlay on Image")
        plt.imshow(overlay)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# 示例调用
image_file = 'datasets/sar2opt/val/ROIs1868_summer_s1_59_p425.png' 
visualize_edge_comparison(
    image_file, 
    downsample_factor=1, 
    thresholds=[(50, 150)]
    # thresholds=[(30, 100), (50, 150), (80, 200)] # 测试不同的 Canny 阈值
)