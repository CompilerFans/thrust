#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip>
#include <cub/device/device_segmented_merge_sort.cuh>

#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
} while(0)

struct BenchmarkConfig {
    int num_items;
    int num_segments;
    int num_runs;
    std::string description;
};

double benchmark_sort_pairs(const BenchmarkConfig& config) {
    // Generate random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, config.num_items);
    
    std::vector<int> h_keys_in(config.num_items);
    std::vector<int> h_values_in(config.num_items);
    for (int i = 0; i < config.num_items; ++i) {
        h_keys_in[i] = dis(gen);
        h_values_in[i] = i;
    }
    
    // Generate segment offsets
    std::vector<int> h_segment_offsets(config.num_segments + 1);
    h_segment_offsets[0] = 0;
    h_segment_offsets[config.num_segments] = config.num_items;
    for (int i = 1; i < config.num_segments; ++i) {
        h_segment_offsets[i] = (i * config.num_items) / config.num_segments;
    }
    
    // Allocate device memory
    int *d_keys_in, *d_keys_out, *d_values_in, *d_values_out, *d_segment_offsets;
    CUDA_CHECK(cudaMalloc(&d_keys_in, sizeof(int) * config.num_items));
    CUDA_CHECK(cudaMalloc(&d_keys_out, sizeof(int) * config.num_items));
    CUDA_CHECK(cudaMalloc(&d_values_in, sizeof(int) * config.num_items));
    CUDA_CHECK(cudaMalloc(&d_values_out, sizeof(int) * config.num_items));
    CUDA_CHECK(cudaMalloc(&d_segment_offsets, sizeof(int) * (config.num_segments + 1)));
    
    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_keys_in, h_keys_in.data(), sizeof(int) * config.num_items, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values_in, h_values_in.data(), sizeof(int) * config.num_items, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_segment_offsets, h_segment_offsets.data(), sizeof(int) * (config.num_segments + 1), cudaMemcpyHostToDevice));
    
    // Allocate temporary storage
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    CUDA_CHECK(cub::DeviceSegmentedMergeSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out, d_values_in, d_values_out,
        config.num_items, config.num_segments, d_segment_offsets, d_segment_offsets + 1));
    
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
    // Warmup
    for (int i = 0; i < 3; ++i) {
        CUDA_CHECK(cub::DeviceSegmentedMergeSort::SortPairs(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out, d_values_in, d_values_out,
            config.num_items, config.num_segments, d_segment_offsets, d_segment_offsets + 1));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timing runs
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int run = 0; run < config.num_runs; ++run) {
        CUDA_CHECK(cub::DeviceSegmentedMergeSort::SortPairs(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out, d_values_in, d_values_out,
            config.num_items, config.num_segments, d_segment_offsets, d_segment_offsets + 1));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avg_time = duration.count() / (double)config.num_runs;
    
    // Clean up
    CUDA_CHECK(cudaFree(d_keys_in));
    CUDA_CHECK(cudaFree(d_keys_out));
    CUDA_CHECK(cudaFree(d_values_in));
    CUDA_CHECK(cudaFree(d_values_out));
    CUDA_CHECK(cudaFree(d_segment_offsets));
    CUDA_CHECK(cudaFree(d_temp_storage));
    
    return avg_time;
}

double benchmark_sort_keys(const BenchmarkConfig& config) {
    // Generate random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, config.num_items);
    
    std::vector<int> h_keys_in(config.num_items);
    for (int i = 0; i < config.num_items; ++i) {
        h_keys_in[i] = dis(gen);
    }
    
    // Generate segment offsets
    std::vector<int> h_segment_offsets(config.num_segments + 1);
    h_segment_offsets[0] = 0;
    h_segment_offsets[config.num_segments] = config.num_items;
    for (int i = 1; i < config.num_segments; ++i) {
        h_segment_offsets[i] = (i * config.num_items) / config.num_segments;
    }
    
    // Allocate device memory
    int *d_keys_in, *d_keys_out, *d_segment_offsets;
    CUDA_CHECK(cudaMalloc(&d_keys_in, sizeof(int) * config.num_items));
    CUDA_CHECK(cudaMalloc(&d_keys_out, sizeof(int) * config.num_items));
    CUDA_CHECK(cudaMalloc(&d_segment_offsets, sizeof(int) * (config.num_segments + 1)));
    
    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_keys_in, h_keys_in.data(), sizeof(int) * config.num_items, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_segment_offsets, h_segment_offsets.data(), sizeof(int) * (config.num_segments + 1), cudaMemcpyHostToDevice));
    
    // Allocate temporary storage
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    CUDA_CHECK(cub::DeviceSegmentedMergeSort::SortKeys(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out,
        config.num_items, config.num_segments, d_segment_offsets, d_segment_offsets + 1));
    
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
    // Warmup
    for (int i = 0; i < 3; ++i) {
        CUDA_CHECK(cub::DeviceSegmentedMergeSort::SortKeys(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out,
            config.num_items, config.num_segments, d_segment_offsets, d_segment_offsets + 1));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timing runs
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int run = 0; run < config.num_runs; ++run) {
        CUDA_CHECK(cub::DeviceSegmentedMergeSort::SortKeys(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out,
            config.num_items, config.num_segments, d_segment_offsets, d_segment_offsets + 1));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avg_time = duration.count() / (double)config.num_runs;
    
    // Clean up
    CUDA_CHECK(cudaFree(d_keys_in));
    CUDA_CHECK(cudaFree(d_keys_out));
    CUDA_CHECK(cudaFree(d_segment_offsets));
    CUDA_CHECK(cudaFree(d_temp_storage));
    
    return avg_time;
}

void run_scaling_benchmark() {
    std::cout << "=== Scaling Performance Benchmark ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    
    // Test different sizes
    std::vector<BenchmarkConfig> configs = {
        {1000, 10, 100, "Small: 1K items, 10 segments"},
        {10000, 100, 50, "Medium: 10K items, 100 segments"},
        {100000, 1000, 20, "Large: 100K items, 1K segments"},
        {1000000, 10000, 10, "X-Large: 1M items, 10K segments"},
        {10000000, 100000, 5, "XX-Large: 10M items, 100K segments"}
    };
    
    std::cout << "\n--- SortPairs Performance ---" << std::endl;
    std::cout << std::setw(40) << "Test Case" 
              << std::setw(15) << "Avg Time (μs)"
              << std::setw(20) << "Throughput (M items/s)"
              << std::setw(15) << "Memory BW (GB/s)" << std::endl;
    std::cout << std::string(90, '-') << std::endl;
    
    for (const auto& config : configs) {
        double avg_time = benchmark_sort_pairs(config);
        double throughput = (config.num_items / avg_time) * 1e6 / 1e6; // Million items per second
        double memory_bw = (config.num_items * sizeof(int) * 4 / avg_time) * 1e6 / 1e9; // GB/s (read keys, values + write keys, values)
        
        std::cout << std::setw(40) << config.description
                  << std::setw(15) << avg_time
                  << std::setw(20) << throughput
                  << std::setw(15) << memory_bw << std::endl;
    }
    
    std::cout << "\n--- SortKeys Performance ---" << std::endl;
    std::cout << std::setw(40) << "Test Case" 
              << std::setw(15) << "Avg Time (μs)"
              << std::setw(20) << "Throughput (M items/s)"
              << std::setw(15) << "Memory BW (GB/s)" << std::endl;
    std::cout << std::string(90, '-') << std::endl;
    
    for (const auto& config : configs) {
        double avg_time = benchmark_sort_keys(config);
        double throughput = (config.num_items / avg_time) * 1e6 / 1e6; // Million items per second
        double memory_bw = (config.num_items * sizeof(int) * 2 / avg_time) * 1e6 / 1e9; // GB/s (read + write keys)
        
        std::cout << std::setw(40) << config.description
                  << std::setw(15) << avg_time
                  << std::setw(20) << throughput
                  << std::setw(15) << memory_bw << std::endl;
    }
}

void run_segment_size_benchmark() {
    std::cout << "\n=== Segment Size Impact Benchmark ===" << std::endl;
    
    const int num_items = 1000000;
    const int num_runs = 10;
    
    // Test different segment sizes
    std::vector<int> segment_counts = {10, 100, 1000, 10000, 100000};
    
    std::cout << std::setw(20) << "Num Segments"
              << std::setw(20) << "Avg Segment Size"
              << std::setw(15) << "Time (μs)"
              << std::setw(20) << "Throughput (M items/s)" << std::endl;
    std::cout << std::string(75, '-') << std::endl;
    
    for (int num_segments : segment_counts) {
        BenchmarkConfig config = {num_items, num_segments, num_runs, ""};
        double avg_time = benchmark_sort_pairs(config);
        double throughput = (num_items / avg_time) * 1e6 / 1e6;
        int avg_segment_size = num_items / num_segments;
        
        std::cout << std::setw(20) << num_segments
                  << std::setw(20) << avg_segment_size
                  << std::setw(15) << std::fixed << std::setprecision(2) << avg_time
                  << std::setw(20) << throughput << std::endl;
    }
}

void run_memory_usage_analysis() {
    std::cout << "\n=== Memory Usage Analysis ===" << std::endl;
    
    std::vector<int> sizes = {1000, 10000, 100000, 1000000};
    
    std::cout << std::setw(15) << "Items"
              << std::setw(20) << "Input Memory (MB)"
              << std::setw(20) << "Temp Storage (bytes)"
              << std::setw(20) << "Memory Efficiency" << std::endl;
    std::cout << std::string(75, '-') << std::endl;
    
    for (int size : sizes) {
        const int num_segments = size / 100; // Average 100 items per segment
        
        // Calculate input memory
        size_t input_memory = size * sizeof(int) * 4; // keys_in, keys_out, values_in, values_out
        size_t segment_memory = (num_segments + 1) * sizeof(int);
        size_t total_input = input_memory + segment_memory;
        
        // Get temp storage requirement
        int *d_keys_in, *d_keys_out, *d_values_in, *d_values_out, *d_segment_offsets;
        CUDA_CHECK(cudaMalloc(&d_keys_in, sizeof(int) * size));
        CUDA_CHECK(cudaMalloc(&d_keys_out, sizeof(int) * size));
        CUDA_CHECK(cudaMalloc(&d_values_in, sizeof(int) * size));
        CUDA_CHECK(cudaMalloc(&d_values_out, sizeof(int) * size));
        CUDA_CHECK(cudaMalloc(&d_segment_offsets, sizeof(int) * (num_segments + 1)));
        
        void *d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        
        CUDA_CHECK(cub::DeviceSegmentedMergeSort::SortPairs(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out, d_values_in, d_values_out,
            size, num_segments, d_segment_offsets, d_segment_offsets + 1));
        
        double input_mb = total_input / (1024.0 * 1024.0);
        double efficiency = (double)total_input / (total_input + temp_storage_bytes) * 100.0;
        
        std::cout << std::setw(15) << size
                  << std::setw(20) << std::fixed << std::setprecision(2) << input_mb
                  << std::setw(20) << temp_storage_bytes
                  << std::setw(19) << efficiency << "%" << std::endl;
        
        CUDA_CHECK(cudaFree(d_keys_in));
        CUDA_CHECK(cudaFree(d_keys_out));
        CUDA_CHECK(cudaFree(d_values_in));
        CUDA_CHECK(cudaFree(d_values_out));
        CUDA_CHECK(cudaFree(d_segment_offsets));
    }
}

int main() {
    std::cout << "CUB DeviceSegmentedMergeSort Performance Benchmark" << std::endl;
    std::cout << "==================================================" << std::endl;
    
    // Get GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    std::cout << "SMs: " << prop.multiProcessorCount << std::endl;
    std::cout << "==================================================" << std::endl;
    
    run_scaling_benchmark();
    run_segment_size_benchmark(); 
    run_memory_usage_analysis();
    
    std::cout << "\n=== Performance Benchmark Completed ===" << std::endl;
    
    return 0;
}