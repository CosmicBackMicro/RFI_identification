#!/bin/bash

echo "=== CUDA控制功能测试 ==="
echo

echo "1. 测试帮助信息（应显示--enableCuda选项）："
./build/ReadFASTData --help 2>&1 | grep -A1 -B1 enableCuda
echo

echo "2. 测试CUDA启用（-c 1）："
timeout 5s ./build/ReadFASTData -i /dev/null -n 1 -s 0 -t 1 -f 1 -M 0 -p /tmp -P 0 -c 1 2>&1 | grep -E "(CUDA|acceleration)" | head -5
echo

echo "3. 测试CUDA禁用（-c 0）："
timeout 5s ./build/ReadFASTData -i /dev/null -n 1 -s 0 -t 1 -f 1 -M 0 -p /tmp -P 0 -c 0 2>&1 | grep -E "(CUDA|acceleration|disabled)" | head -5
echo

echo "4. 测试默认行为（应启用CUDA）："
timeout 5s ./build/ReadFASTData -i /dev/null -n 1 -s 0 -t 1 -f 1 -M 0 -p /tmp -P 0 2>&1 | grep -E "(CUDA|acceleration)" | head -5
echo

echo "=== 测试完成 ==="
