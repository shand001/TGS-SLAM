#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python3 /home/shen/DSLAM/convert_to_nyu40.py
import os
import numpy as np
import pandas as pd
from PIL import Image

# —— 配置区域 —— #
# 输入标签图所在目录
INPUT_DIR = "/home/handong/scannet/scans/scene0181_00/label-filt"
# 输出映射后标签图目录
OUTPUT_DIR = "/home/handong/scannet/scans/scene0181_00/label_40"
# scannetv2-labels.combined.tsv 的路径
MAPPING_TSV = "/home/handong/TGS-SLAM/preprocess/scannet/scannetv2-labels.combined.tsv"
# —————— #

def load_mapping(tsv_path):
    """
    读取 TSV 文件，建立 从 原始 id -> NYU-40 id 的映射字典
    """
    # 如果 TSV 文件里包含注释行（以 # 开头），pandas 可以用 comment 参数跳过
    df = pd.read_csv(tsv_path, sep='\t', comment='#', 
                     usecols=['id', 'nyu40id'])
    # 构造 dict： key = 原始 id , value = nyu40id
    mapping = dict(zip(df['id'], df['nyu40id']))
    return mapping

def remap_label_image(img_array, mapping, unk_val=0):
    """
    将 img_array 中的每个标签值，用 mapping 映射到新的 NYU-40 类别。
    mapping 中不存在的标签，赋值为 unk_val（默认背景 0）
    """
    # 方法一：用 numpy.vectorize
    mapper = np.vectorize(lambda x: mapping.get(int(x), unk_val))
    return mapper(img_array).astype(np.uint8)

def main():
    # 读取映射表
    mapping = load_mapping(MAPPING_TSV)
    print(f"Loaded mapping: {len(mapping)} entries.")
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 遍历所有标签图
    for fname in sorted(os.listdir(INPUT_DIR)):
        if not (fname.lower().endswith('.png') or fname.lower().endswith('.tif')):
            continue
        
        in_path  = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(OUTPUT_DIR, fname)
        
        # 以灰度方式打开 -> numpy 数组
        img = Image.open(in_path)
        arr = np.array(img)
        
        # remap
        remapped = remap_label_image(arr, mapping, unk_val=0)
        
        # 保存
        Image.fromarray(remapped).save(out_path)
        print(f"Processed {fname} → {out_path}")
    
    print("All done.")

if __name__ == "__main__":
    main()
