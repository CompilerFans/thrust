#!/bin/bash

# CUB DeviceSegmentedMergeSort Build Script
# =========================================

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
INCLUDE_DIRS="-I../../cub"

# Build directory
BUILD_DIR="build"
mkdir -p $BUILD_DIR

print_status "Building CUB DeviceSegmentedMergeSort examples..."

# Build test_segsort (main test without Thrust dependencies)
print_status "Compiling test_segsort..."
if nvcc $NVCC_FLAGS $INCLUDE_DIRS test_segsort.cu -o $BUILD_DIR/test_segsort; then
    print_success "test_segsort compiled successfully"
else
    print_error "Failed to compile test_segsort"
    exit 1
fi

print_success "Build completed successfully!"
print_status ""
print_status "Available executables in $BUILD_DIR/:"
print_status "  - test_segsort                : Comprehensive segmented merge sort test"
print_status ""
print_status "To run test:"
print_status "  cd $BUILD_DIR"
print_status "  ./test_segsort                # Run segmented merge sort test"
