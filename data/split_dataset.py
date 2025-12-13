import os
import shutil
import random
import argparse
from typing import List, Tuple, Dict

def split_dataset(
    source_dir: str, 
    output_root_dir: str, 
    train_ratio: float, 
    val_ratio: float, 
    copy_files: bool = True
) -> None:
    """
    将源目录中的文件按比例随机划分到 train, val, test 目录中。

    :param source_dir: 包含所有待划分文件的源目录路径。
    :param output_root_dir: 数据集划分后的根目录，例如 'data/my_dataset_splits'。
    :param train_ratio: 训练集所占比例 (0.0 到 1.0)。
    :param val_ratio: 验证集所占比例 (0.0 到 1.0)。
    :param copy_files: True 表示复制文件，False 表示移动文件 (默认: 复制)。
    """
    
    # 确保比例总和合理
    if not (0.0 < train_ratio + val_ratio < 1.0):
        raise ValueError("训练集和验证集比例之和必须小于 1.0。")
    
    test_ratio = 1.0 - train_ratio - val_ratio
    split_ratios = {'train': train_ratio, 'val': val_ratio, 'test': test_ratio}

    print(f"--- 数据集划分比例 ---")
    print(f"训练集 (Train): {train_ratio:.2f}")
    print(f"验证集 (Val): {val_ratio:.2f}")
    print(f"测试集 (Test): {test_ratio:.2f}")
    print("-" * 25)

    # 获取所有文件的列表（排除目录）
    all_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    random.shuffle(all_files)
    
    total_files = len(all_files)
    if total_files == 0:
        print("源目录中没有找到任何文件，操作终止。")
        return

    # 计算划分的索引
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    
    train_files = all_files[:train_count]
    val_files = all_files[train_count : train_count + val_count]
    test_files = all_files[train_count + val_count :]

    splits: Dict[str, List[str]] = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    print(f"文件总数: {total_files}")
    print(f"Train 数量: {len(train_files)} | Val 数量: {len(val_files)} | Test 数量: {len(test_files)}")
    print(f"目标操作: {'复制' if copy_files else '移动'}")
    print("-" * 25)

    op_func = shutil.copy2 if copy_files else shutil.move
    op_name = '复制' if copy_files else '移动'
    
    # 执行文件操作
    for split_name, file_list in splits.items():
        dest_dir = os.path.join(output_root_dir, split_name)
        os.makedirs(dest_dir, exist_ok=True)
        
        print(f"开始 {op_name} {split_name} ({len(file_list)} 个文件) 到 {dest_dir}...")
        
        for i, filename in enumerate(file_list):
            source_path = os.path.join(source_dir, filename)
            dest_path = os.path.join(dest_dir, filename)
            
            op_func(source_path, dest_path)
            
            if (i + 1) % 1000 == 0 or (i + 1) == len(file_list):
                print(f"\t已完成 {i + 1}/{len(file_list)}", end='\r')
        
        print(f"\n{split_name} 划分完成。")

def main():
    parser = argparse.ArgumentParser(description="随机划分数据集为训练集、验证集和测试集。")
    parser.add_argument('source_dir', type=str, 
                        help="包含所有待划分文件的源目录路径。")
    parser.add_argument('output_root_dir', type=str,
                        help="数据集划分后的根目录，例如：'./dataset_splits'。")
    parser.add_argument('--train', type=float, default=0.8,
                        help="训练集比例 (默认: 0.8)")
    parser.add_argument('--val', type=float, default=0.1,
                        help="验证集比例 (默认: 0.1)")
    parser.add_argument('--move', action='store_true',
                        help="使用移动 (move) 操作而非默认的复制 (copy) 操作。")

    args = parser.parse_args()

    # 检查源目录是否存在
    if not os.path.isdir(args.source_dir):
        print(f"错误：源目录不存在或不是一个目录: {args.source_dir}")
        return

    try:
        split_dataset(
            source_dir=args.source_dir,
            output_root_dir=args.output_root_dir,
            train_ratio=args.train,
            val_ratio=args.val,
            copy_files=not args.move
        )
        print("\n所有数据集划分完成！")
    except ValueError as e:
        print(f"\n错误: {e}")
    except Exception as e:
        print(f"\n发生未知错误: {e}")

if __name__ == '__main__':
    main()