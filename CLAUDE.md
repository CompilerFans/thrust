# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

Thrust 是 NVIDIA 的 C++ 并行算法库，专为 GPU 计算设计。这是一个头文件库，包含三个主要子项目：
- **Thrust**: 主要的并行算法库
- **CUB**: CUDA 基础算法库（低级块级和设备级原语）
- **libcudacxx**: CUDA C++ 标准库
- **moderngpu**: 现代 GPU 编程库

## 构建系统和常用命令

### CMake 构建（推荐）
```bash
# 创建构建目录
mkdir build && cd build

# 配置
cmake .. [选项]

# 构建
cmake --build . -j ${NUM_JOBS}

# 运行测试
ctest
```

### 快速构建脚本
```bash
# 使用提供的构建脚本
./build.sh
```

### 重要的 CMake 选项
- `-DTHRUST_ENABLE_TESTING=ON/OFF`: 启用/禁用测试
- `-DTHRUST_ENABLE_EXAMPLES=ON/OFF`: 启用/禁用示例
- `-DTHRUST_INCLUDE_CUB_CMAKE=ON`: 包含 CUB 测试和示例
- `-DTHRUST_AUTO_DETECT_COMPUTE_ARCHS=ON`: 自动检测 GPU 架构

## 代码架构

### 目录结构
- `thrust/`: 主要头文件库
  - `system/`: 不同后端系统实现（CUDA、CPP、OMP、TBB）
  - `detail/`: 内部实现细节
  - `iterator/`: 迭代器实现
  - `async/`: 异步算法接口
  - `mr/`: 内存资源管理

- `examples/`: 示例代码
- `testing/`: 单元测试
- `cub/`: CUB 库源码
- `libcudacxx/`: libcudacxx 库源码
- `moderngpu/`: moderngpu 库源码

### 核心设计概念
1. **执行策略**: 通过模板参数控制在 CPU 或 GPU 上执行
2. **迭代器**: 提供丰富的迭代器类型（变换、计数、常量等）
3. **系统标签**: 用于分发算法到不同后端（thrust::cuda, thrust::cpp 等）
4. **异步接口**: 支持非阻塞的 GPU 操作

### 主要算法类别
- **变换**: transform, for_each
- **约简**: reduce, transform_reduce
- **扫描**: scan, transform_scan
- **排序**: sort, stable_sort, sort_by_key
- **搜索**: binary_search, find, count
- **集合操作**: set_union, set_intersection 等

## 测试相关

### 运行测试
```bash
# 运行所有测试
ctest

# 运行特定测试
ctest -R <测试名称模式>
```

### 示例编译
示例可以直接用 nvcc 编译：
```bash
nvcc examples/norm.cu -o norm
```

## 系统要求

- CUDA Toolkit (如使用 CUDA 后端)
- C++14 或更高版本
- CMake 3.15+ (构建)，3.17+ (测试)

## 依赖项

项目在 `dependencies/` 目录包含：
- `cub/`: CUB 库
- `libcudacxx/`: libcudacxx 库
- `moderngpu/`: moderngpu 库

这些是作为子模块或独立目录包含的，确保所有组件版本兼容。

## 开发注意事项

- 这是头文件库，主要开发涉及模板和内联函数
- 支持多种后端：CUDA、OpenMP、TBB、串行 C++
- 代码需要在主机和设备上都能编译
- 遵循现有的命名约定和代码风格
- 新功能应包含适当的测试用例