#!/bin/bash

# 输出结果的文件名
output_file="sft_results.txt"

# 清空文件（如果已存在）
> $output_file

# 循环生成命令并执行
for step in 999 1999 2999 3999 4999 5999 6999 7999 8999 9999
do
  # 生成命令
  command="python generate.py --prompt \"Write a story about a girl named Lily.\" --path /root/autodl-tmp/llama/saves/sft-$step.pt"
  
  # 写入分隔符
  echo "============================================" >> $output_file
  echo "Running model: sft-$step.pt" >> $output_file
  echo "============================================" >> $output_file
  
  # 写入当前命令到结果文件
  echo "Command: $command" >> $output_file
  echo "--------------------------------------------" >> $output_file
  
  # 执行命令并捕获输出
  output=$(eval $command)
  
  # 写入命令输出
  echo "Output:" >> $output_file
  echo "$output" >> $output_file
  echo "--------------------------------------------" >> $output_file
done

echo "All commands executed. Results saved in $output_file"
