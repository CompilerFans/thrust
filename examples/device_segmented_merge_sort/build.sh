#!/bin/bash

# CUB DeviceSegmentedMergeSort Build Script
# =========================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if nvcc is available
if ! command -v nvcc &> /dev/null; then
    print_error "nvcc not found. Please ensure CUDA toolkit is installed and in PATH."
    exit 1
fi

# Get CUDA version
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
print_status "Using CUDA version: $CUDA_VERSION"

# Compilation flags
NVCC_FLAGS="--extended-lambda --expt-relaxed-constexpr -DTHRUST_IGNORE_CUB_VERSION_CHECK"
INCLUDE_DIRS="-I../../cub -I../../moderngpu/src"

# Build directory
BUILD_DIR="build"
mkdir -p $BUILD_DIR

print_status "Building CUB DeviceSegmentedMergeSort examples..."

# Build functionality test
print_status "Compiling functionality test..."
if nvcc $NVCC_FLAGS $INCLUDE_DIRS functionality_test.cu -o $BUILD_DIR/functionality_test; then
    print_success "Functionality test compiled successfully"
else
    print_error "Failed to compile functionality test"
    exit 1
fi

# Build performance test
print_status "Compiling performance test..."
if nvcc $NVCC_FLAGS $INCLUDE_DIRS performance_test.cu -o $BUILD_DIR/performance_test; then
    print_success "Performance test compiled successfully"
else
    print_error "Failed to compile performance test"
    exit 1
fi

# Build basic example
print_status "Compiling basic example..."
if nvcc $NVCC_FLAGS $INCLUDE_DIRS device_segmented_merge_sort_example.cu -o $BUILD_DIR/device_segmented_merge_sort_example; then
    print_success "Basic example compiled successfully"
else
    print_error "Failed to compile basic example"
    exit 1
fi

print_success "All builds completed successfully!"
print_status ""
print_status "Available executables in $BUILD_DIR/:"
print_status "  - functionality_test          : Comprehensive functionality tests"
print_status "  - performance_test            : Performance benchmarks and analysis"
print_status "  - device_segmented_merge_sort_example : Basic usage example"
print_status ""
print_status "To run tests:"
print_status "  cd $BUILD_DIR"
print_status "  ./functionality_test          # Run functionality tests"
print_status "  ./performance_test            # Run performance benchmarks"
print_status "  ./device_segmented_merge_sort_example  # Run basic example"