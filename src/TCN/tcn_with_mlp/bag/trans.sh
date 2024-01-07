#!/bin/bash

# 设置 ROSBAG 文件名和输出的 CSV 文件名
bagfile="$1"
output_csv="${bagfile%.bag}_pre.csv"

# 设置要包含的主题列表
topics="/cmd_vel /odometry/imu"

# 创建一个临时目录以保存单独的 CSV 文件
temp_dir=$(mktemp -d)
trap 'rm -rf "$temp_dir"' EXIT

# 循环遍历每个主题并将其保存到单独的 CSV 文件中
for topic in $topics; do
    echo "Converting topic: $topic"
    rostopic echo -b "$bagfile" -p "$topic" > "$temp_dir/${topic//\//_}.csv"
done

# 使用 paste 命令将所有 CSV 文件的内容合并到一个文件中
paste -d "," $temp_dir/*.csv > "$output_csv"

