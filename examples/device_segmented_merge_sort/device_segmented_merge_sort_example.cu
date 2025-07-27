#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

#include <cub/device/device_segmented_merge_sort.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>

#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
} while(0)

// CPU reference implementation for comparison
std::vector<int> cpu_segmented_sort(const std::vector<int>& data, const std::vector<int>& segments) {
    std::vector<int> copy = data;
    int cur = 0;
    for (int seg = 0; seg < segments.size() - 1; ++seg) {
        int next = segments[seg + 1];
        std::sort(copy.data() + cur, copy.data() + next);
        cur = next;
    }
    return copy;
}

// Enhanced validation function
bool validate_segmented_sort(const std::vector<int>& input, const std::vector<int>& output, 
                           const std::vector<int>& offsets, const char* test_name) {
    std::cout << "\n--- Validating " << test_name << " ---" << std::endl;
    
    // Check if output is all zeros (critical bug detection)
    bool all_zeros = true;
    for (int val : output) {
        if (val != 0) {
            all_zeros = false;
            break;
        }
    }
    
    if (all_zeros && !input.empty()) {
        std::cout << "ERROR: Output is all zeros but input is not empty!" << std::endl;
        return false;
    }
    
    // Get CPU reference result
    std::vector<int> cpu_result = cpu_segmented_sort(input, offsets);
    
    // Compare with CPU reference
    bool matches_cpu = (output == cpu_result);
    if (!matches_cpu) {
        std::cout << "ERROR: GPU result does not match CPU reference!" << std::endl;
        std::cout << "First few differences:" << std::endl;
        for (size_t i = 0; i < std::min(output.size(), size_t(20)); ++i) {
            if (output[i] != cpu_result[i]) {
                std::cout << "  Index " << i << ": GPU=" << output[i] << ", CPU=" << cpu_result[i] << std::endl;
            }
        }
        return false;
    }
    
    // Verify segment-wise sorting
    bool segment_sorted = true;
    for (size_t seg = 0; seg < offsets.size() - 1; ++seg) {
        int start = offsets[seg];
        int end = offsets[seg + 1];
        
        for (int j = start; j < end - 1; ++j) {
            if (output[j] > output[j + 1]) {
                std::cout << "ERROR: Segment " << seg << " not sorted at positions " 
                         << j << "," << j+1 << ": " << output[j] << " > " << output[j + 1] << std::endl;
                segment_sorted = false;
                break;
            }
        }
        if (!segment_sorted) break;
    }
    
    std::cout << "Validation result: " << (matches_cpu && segment_sorted ? "PASS" : "FAIL") << std::endl;
    return matches_cpu && segment_sorted;
}

void test_basic_functionality() {
    std::cout << "=== Testing Basic Functionality ===" << std::endl;
    
    const int num_items = 12;
    const int num_segments = 3;
    
    // Input data
    int h_keys_in[num_items] = {8, 6, 7, 5, 3, 0, 9, 2, 1, 4, 11, 10};
    int h_values_in[num_items] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    int h_segment_offsets[num_segments + 1] = {0, 4, 7, 12};
    
    std::cout << "Input keys: ";
    for (int i = 0; i < num_items; ++i) {
        std::cout << h_keys_in[i] << " ";
    }
    std::cout << std::endl;
    
    // Allocate device memory
    int *d_keys_in, *d_keys_out, *d_values_in, *d_values_out, *d_segment_offsets;
    CUDA_CHECK(cudaMalloc(&d_keys_in, sizeof(int) * num_items));
    CUDA_CHECK(cudaMalloc(&d_keys_out, sizeof(int) * num_items));
    CUDA_CHECK(cudaMalloc(&d_values_in, sizeof(int) * num_items));
    CUDA_CHECK(cudaMalloc(&d_values_out, sizeof(int) * num_items));
    CUDA_CHECK(cudaMalloc(&d_segment_offsets, sizeof(int) * (num_segments + 1)));
    
    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_keys_in, h_keys_in, sizeof(int) * num_items, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values_in, h_values_in, sizeof(int) * num_items, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_segment_offsets, h_segment_offsets, sizeof(int) * (num_segments + 1), cudaMemcpyHostToDevice));
    
    // Test 1: DeviceSegmentedMergeSort::SortPairs
    {
        std::cout << "\n--- Test 1: DeviceSegmentedMergeSort::SortPairs ---" << std::endl;
        
        void *d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        
        CUDA_CHECK(cub::DeviceSegmentedMergeSort::SortPairs(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out, d_values_in, d_values_out,
            num_items, num_segments, d_segment_offsets, d_segment_offsets + 1));
        
        CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
        
        CUDA_CHECK(cub::DeviceSegmentedMergeSort::SortPairs(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out, d_values_in, d_values_out,
            num_items, num_segments, d_segment_offsets, d_segment_offsets + 1));
        
        // Copy results back
        int h_keys_out[num_items], h_values_out[num_items];
        CUDA_CHECK(cudaMemcpy(h_keys_out, d_keys_out, sizeof(int) * num_items, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_values_out, d_values_out, sizeof(int) * num_items, cudaMemcpyDeviceToHost));
        
        std::cout << "MergeSort Output keys: ";
        for (int i = 0; i < num_items; ++i) std::cout << h_keys_out[i] << " ";
        std::cout << std::endl;
        
        // Validate results
        std::vector<int> input_vec(h_keys_in, h_keys_in + num_items);
        std::vector<int> output_vec(h_keys_out, h_keys_out + num_items);
        std::vector<int> offsets_vec(h_segment_offsets, h_segment_offsets + num_segments + 1);
        
        bool merge_correct = validate_segmented_sort(input_vec, output_vec, offsets_vec, "DeviceSegmentedMergeSort::SortPairs");
        
        CUDA_CHECK(cudaFree(d_temp_storage));
        
        // Compare with CUB DeviceSegmentedRadixSort
        std::cout << "\n--- Comparing with CUB DeviceSegmentedRadixSort ---" << std::endl;
        
        int *d_radix_keys_out, *d_radix_values_out;
        CUDA_CHECK(cudaMalloc(&d_radix_keys_out, sizeof(int) * num_items));
        CUDA_CHECK(cudaMalloc(&d_radix_values_out, sizeof(int) * num_items));
        
        void *d_radix_temp = nullptr;
        size_t radix_temp_bytes = 0;
        
        CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(
            d_radix_temp, radix_temp_bytes,
            d_keys_in, d_radix_keys_out, d_values_in, d_radix_values_out,
            num_items, num_segments, d_segment_offsets, d_segment_offsets + 1));
        
        CUDA_CHECK(cudaMalloc(&d_radix_temp, radix_temp_bytes));
        
        CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(
            d_radix_temp, radix_temp_bytes,
            d_keys_in, d_radix_keys_out, d_values_in, d_radix_values_out,
            num_items, num_segments, d_segment_offsets, d_segment_offsets + 1));
        
        int h_radix_keys[num_items], h_radix_values[num_items];
        CUDA_CHECK(cudaMemcpy(h_radix_keys, d_radix_keys_out, sizeof(int) * num_items, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_radix_values, d_radix_values_out, sizeof(int) * num_items, cudaMemcpyDeviceToHost));
        
        std::cout << "RadixSort  Output keys: ";
        for (int i = 0; i < num_items; ++i) std::cout << h_radix_keys[i] << " ";
        std::cout << std::endl;
        
        // Compare results
        bool keys_match = true, values_match = true;
        for (int i = 0; i < num_items; ++i) {
            if (h_keys_out[i] != h_radix_keys[i]) {
                keys_match = false;
                std::cout << "Key difference at index " << i << ": MergeSort=" << h_keys_out[i] 
                         << ", RadixSort=" << h_radix_keys[i] << std::endl;
                break;
            }
            if (h_values_out[i] != h_radix_values[i]) {
                values_match = false;
                std::cout << "Value difference at index " << i << ": MergeSort=" << h_values_out[i] 
                         << ", RadixSort=" << h_radix_values[i] << std::endl;
                break;
            }
        }
        
        std::cout << "Keys comparison: " << (keys_match ? "MATCH" : "DIFFER") << std::endl;
        std::cout << "Values comparison: " << (values_match ? "MATCH" : "DIFFER") << std::endl;
        std::cout << "Overall MergeSort vs RadixSort: " << (keys_match && values_match ? "IDENTICAL" : "DIFFERENT") << std::endl;
        
        CUDA_CHECK(cudaFree(d_radix_keys_out));
        CUDA_CHECK(cudaFree(d_radix_values_out));
        CUDA_CHECK(cudaFree(d_radix_temp));
        
        if (merge_correct && keys_match && values_match) {
            std::cout << "\n✓ DeviceSegmentedMergeSort::SortPairs PASSED all tests!" << std::endl;
        } else {
            std::cout << "\n✗ DeviceSegmentedMergeSort::SortPairs FAILED tests!" << std::endl;
        }
    }
    
    // Clean up
    CUDA_CHECK(cudaFree(d_keys_in));
    CUDA_CHECK(cudaFree(d_keys_out));
    CUDA_CHECK(cudaFree(d_values_in));
    CUDA_CHECK(cudaFree(d_values_out));
    CUDA_CHECK(cudaFree(d_segment_offsets));
}

int main() {
    std::cout << "Enhanced DeviceSegmentedMergeSort Test with CUB RadixSort Comparison" << std::endl;
    
    test_basic_functionality();
    
    std::cout << "\n=== All Tests Completed ===" << std::endl;
    
    return 0;
}