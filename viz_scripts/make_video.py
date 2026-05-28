#!/usr/bin/env python3
import os
import cv2
import argparse
from pathlib import Path
import re


def numeric_key(path: Path):
    """
    从文件名里提取数字用于排序，例如：
    0.png, 1.png, 10.png
    或 frame_0.png, frame_10.png
    都会按数字大小排序。
    """
    name = path.stem  # 不带后缀的文件名
    # 提取里面所有连续数字，比如 "frame_0012" -> "0012"
    nums = re.findall(r"\d+", name)
    if not nums:
        # 没数字就按原字符串排序
        return float("inf"), name
    # 用最后一段数字作为序号
    return int(nums[-1]), name


def make_video(img_dir, num_frames, fps, output_name):
    img_dir = Path(img_dir)

    if not img_dir.exists():
        raise FileNotFoundError(f"图片目录不存在: {img_dir}")

    # 找到所有 png 图片，并按“数字从小到大”排序
    images = [p for p in img_dir.glob("*.png")]
    if not images:
        raise RuntimeError(f"目录中没有找到 png 图片: {img_dir}")

    images = sorted(images, key=numeric_key)

    # 只取前 num_frames 张
    images = images[:num_frames]
    if not images:
        raise RuntimeError(f"前 {num_frames} 张图片为空，请检查参数或目录内容。")

    print("排序后的前几张图片示例：")
    for p in images[:10]:
        print(" ", p.name)
    print(f"共使用 {len(images)} 张图片生成视频。")

    # 读取第一张图片来获取尺寸
    first_img = cv2.imread(str(images[0]))
    if first_img is None:
        raise RuntimeError(f"无法读取图片: {images[0]}")
    height, width, _ = first_img.shape

    # 输出视频路径
    output_path = img_dir / output_name

    # 定义视频编码器和 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 生成 mp4
    video_writer = cv2.VideoWriter(
        str(output_path), fourcc, fps, (width, height))

    if not video_writer.isOpened():
        raise RuntimeError("VideoWriter 打开失败，可能是编码器不支持。")

    # 逐张写入
    for i, img_path in enumerate(images, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"警告: 无法读取图片 {img_path}，跳过。")
            continue

        # 如果有尺寸不一致，调整到第一张的大小
        if img.shape[0] != height or img.shape[1] != width:
            img = cv2.resize(img, (width, height))

        video_writer.write(img)
        print(f"[{i}/{len(images)}] 已写入: {img_path.name}")

    video_writer.release()
    print(f"完成！输出视频文件: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="将指定目录中的前 N 张 png 图片按数字顺序拼成视频"
    )
    parser.add_argument(
        "num_frames",
        type=int,
        help="使用前多少张图片（整数）"
    )
    parser.add_argument(
        "fps",
        type=float,
        help="视频帧率（如 25、30 等）"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output.mp4",
        help="输出视频文件名（默认：output.mp4）"
    )
    parser.add_argument(
        "-d", "--dir",
        type=str,
        default="/home/handong/TGS-SLAM/experiments/Replica/room0_2008_20251117_0044/3d_vis/rgb/source",
        help="图片目录（默认为题目给的目录）"
    )

    args = parser.parse_args()

    make_video(
        img_dir=args.dir,
        num_frames=args.num_frames,
        fps=args.fps,
        output_name=args.output
    )


if __name__ == "__main__":
    main()
