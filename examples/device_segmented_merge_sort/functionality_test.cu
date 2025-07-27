#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
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

// CPU reference implementation
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
                           const std::vector<int>& offsets, const char* test_name, bool verbose = true) {
    if (verbose) {
        std::cout << "\n--- Validating " << test_name << " ---" << std::endl;
    }
    
    // Critical bug detection: check if output is all zeros
    bool all_zeros = true;
    for (int val : output) {
        if (val != 0) {
            all_zeros = false;
            break;
        }
    }
    
    if (all_zeros && !input.empty()) {
        if (verbose) std::cout << "ERROR: Output is all zeros but input is not empty!" << std::endl;
        return false;
    }
    
    // Get CPU reference result
    std::vector<int> cpu_result = cpu_segmented_sort(input, offsets);
    
    // Compare with CPU reference
    bool matches_cpu = (output == cpu_result);
    if (!matches_cpu && verbose) {
        std::cout << "ERROR: GPU result does not match CPU reference!" << std::endl;
        std::cout << "First few differences:" << std::endl;
        for (size_t i = 0; i < std::min(output.size(), size_t(10)); ++i) {
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
                if (verbose) {
                    std::cout << "ERROR: Segment " << seg << " not sorted at positions " 
                             << j << "," << j+1 << ": " << output[j] << " > " << output[j + 1] << std::endl;
                }
                segment_sorted = false;
                break;
            }
        }
        if (!segment_sorted) break;
    }
    
    bool result = matches_cpu && segment_sorted;
    if (verbose) {
        std::cout << "Validation result: " << (result ? "PASS" : "FAIL") << std::endl;
    }
    return result;
}

void test_basic_functionality() {
    std::cout << "=== Test 1: Basic Functionality ===" << std::endl;
    
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
    
    // Test DeviceSegmentedMergeSort::SortPairs
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
    
    // Validate results
    std::vector<int> input_vec(h_keys_in, h_keys_in + num_items);
    std::vector<int> output_vec(h_keys_out, h_keys_out + num_items);
    std::vector<int> offsets_vec(h_segment_offsets, h_segment_offsets + num_segments + 1);
    
    bool correct = validate_segmented_sort(input_vec, output_vec, offsets_vec, "SortPairs");
    
    if (correct) {
        std::cout << "\n✓ Basic functionality test PASSED!" << std::endl;
    } else {
        std::cout << "\n✗ Basic functionality test FAILED!" << std::endl;
    }
    
    // Clean up
    CUDA_CHECK(cudaFree(d_keys_in));
    CUDA_CHECK(cudaFree(d_keys_out));
    CUDA_CHECK(cudaFree(d_values_in));
    CUDA_CHECK(cudaFree(d_values_out));
    CUDA_CHECK(cudaFree(d_segment_offsets));
    CUDA_CHECK(cudaFree(d_temp_storage));
}

void test_all_apis() {
    std::cout << "\n=== Test 2: All API Functions ===" << std::endl;
    
    const int num_items = 20;
    const int num_segments = 4;
    
    std::vector<int> h_keys_in = {15, 3, 8, 12, 7, 1, 9, 14, 6, 2, 11, 5, 13, 4, 10, 18, 16, 19, 17, 20};
    std::vector<int> h_values_in(num_items);
    for (int i = 0; i < num_items; ++i) h_values_in[i] = i;
    
    std::vector<int> h_segment_offsets = {0, 5, 10, 15, 20};
    
    // Allocate device memory
    int *d_keys_in, *d_keys_out, *d_values_in, *d_values_out, *d_segment_offsets;
    CUDA_CHECK(cudaMalloc(&d_keys_in, sizeof(int) * num_items));
    CUDA_CHECK(cudaMalloc(&d_keys_out, sizeof(int) * num_items));
    CUDA_CHECK(cudaMalloc(&d_values_in, sizeof(int) * num_items));
    CUDA_CHECK(cudaMalloc(&d_values_out, sizeof(int) * num_items));
    CUDA_CHECK(cudaMalloc(&d_segment_offsets, sizeof(int) * (num_segments + 1)));
    
    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_keys_in, h_keys_in.data(), sizeof(int) * num_items, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values_in, h_values_in.data(), sizeof(int) * num_items, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_segment_offsets, h_segment_offsets.data(), sizeof(int) * (num_segments + 1), cudaMemcpyHostToDevice));
    
    int total_passed = 0;
    const char* api_names[] = {"SortPairs", "SortPairsDescending", "SortKeys", "SortKeysDescending"};
    
    for (int api = 0; api < 4; ++api) {
        std::cout << "\n--- Testing " << api_names[api] << " ---" << std::endl;
        
        void *d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        
        // Call appropriate API
        if (api == 0) {
            CUDA_CHECK(cub::DeviceSegmentedMergeSort::SortPairs(
                d_temp_storage, temp_storage_bytes,
                d_keys_in, d_keys_out, d_values_in, d_values_out,
                num_items, num_segments, d_segment_offsets, d_segment_offsets + 1));
            CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
            CUDA_CHECK(cub::DeviceSegmentedMergeSort::SortPairs(
                d_temp_storage, temp_storage_bytes,
                d_keys_in, d_keys_out, d_values_in, d_values_out,
                num_items, num_segments, d_segment_offsets, d_segment_offsets + 1));
        } else if (api == 1) {
            CUDA_CHECK(cub::DeviceSegmentedMergeSort::SortPairsDescending(
                d_temp_storage, temp_storage_bytes,
                d_keys_in, d_keys_out, d_values_in, d_values_out,
                num_items, num_segments, d_segment_offsets, d_segment_offsets + 1));
            CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
            CUDA_CHECK(cub::DeviceSegmentedMergeSort::SortPairsDescending(
                d_temp_storage, temp_storage_bytes,
                d_keys_in, d_keys_out, d_values_in, d_values_out,
                num_items, num_segments, d_segment_offsets, d_segment_offsets + 1));
        } else if (api == 2) {
            CUDA_CHECK(cub::DeviceSegmentedMergeSort::SortKeys(
                d_temp_storage, temp_storage_bytes,
                d_keys_in, d_keys_out,
                num_items, num_segments, d_segment_offsets, d_segment_offsets + 1));
            CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
            CUDA_CHECK(cub::DeviceSegmentedMergeSort::SortKeys(
                d_temp_storage, temp_storage_bytes,
                d_keys_in, d_keys_out,
                num_items, num_segments, d_segment_offsets, d_segment_offsets + 1));
        } else {
            CUDA_CHECK(cub::DeviceSegmentedMergeSort::SortKeysDescending(
                d_temp_storage, temp_storage_bytes,
                d_keys_in, d_keys_out,
                num_items, num_segments, d_segment_offsets, d_segment_offsets + 1));
            CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
            CUDA_CHECK(cub::DeviceSegmentedMergeSort::SortKeysDescending(
                d_temp_storage, temp_storage_bytes,
                d_keys_in, d_keys_out,
                num_items, num_segments, d_segment_offsets, d_segment_offsets + 1));
        }
        
        std::vector<int> h_keys_out(num_items);
        CUDA_CHECK(cudaMemcpy(h_keys_out.data(), d_keys_out, sizeof(int) * num_items, cudaMemcpyDeviceToHost));
        
        // Validate based on sort type
        bool passed = false;
        if (api == 0 || api == 2) {
            // Ascending sort
            passed = validate_segmented_sort(h_keys_in, h_keys_out, h_segment_offsets, api_names[api], false);
        } else {
            // Descending sort
            std::vector<int> desc_expected = h_keys_in;
            int cur = 0;
            for (int seg = 0; seg < num_segments; ++seg) {
                int next = h_segment_offsets[seg + 1];
                std::sort(desc_expected.data() + cur, desc_expected.data() + next, std::greater<int>());
                cur = next;
            }
            passed = (h_keys_out == desc_expected);
        }
        
        std::cout << api_names[api] << ": " << (passed ? "PASS" : "FAIL") << std::endl;
        if (passed) total_passed++;
        
        CUDA_CHECK(cudaFree(d_temp_storage));
    }
    
    std::cout << "\nAPI Tests Summary: " << total_passed << "/4 tests passed" << std::endl;
    
    // Clean up
    CUDA_CHECK(cudaFree(d_keys_in));
    CUDA_CHECK(cudaFree(d_keys_out));
    CUDA_CHECK(cudaFree(d_values_in));
    CUDA_CHECK(cudaFree(d_values_out));
    CUDA_CHECK(cudaFree(d_segment_offsets));
}

void test_edge_cases() {
    std::cout << "\n=== Test 3: Edge Cases ===" << std::endl;
    
    // Test empty segments
    {
        std::cout << "\n--- Testing with empty segments ---" << std::endl;
        const int num_items = 6;
        const int num_segments = 4;
        
        int h_keys_in[num_items] = {3, 1, 2, 6, 4, 5};
        int h_segment_offsets[num_segments + 1] = {0, 0, 3, 3, 6}; // Empty segments at 0 and 2
        
        int *d_keys_in, *d_keys_out, *d_segment_offsets;
        CUDA_CHECK(cudaMalloc(&d_keys_in, sizeof(int) * num_items));
        CUDA_CHECK(cudaMalloc(&d_keys_out, sizeof(int) * num_items));
        CUDA_CHECK(cudaMalloc(&d_segment_offsets, sizeof(int) * (num_segments + 1)));
        
        CUDA_CHECK(cudaMemcpy(d_keys_in, h_keys_in, sizeof(int) * num_items, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_segment_offsets, h_segment_offsets, sizeof(int) * (num_segments + 1), cudaMemcpyHostToDevice));
        
        void *d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        
        CUDA_CHECK(cub::DeviceSegmentedMergeSort::SortKeys(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out,
            num_items, num_segments, d_segment_offsets, d_segment_offsets + 1));
        
        CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
        CUDA_CHECK(cub::DeviceSegmentedMergeSort::SortKeys(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out,
            num_items, num_segments, d_segment_offsets, d_segment_offsets + 1));
        
        int h_keys_out[num_items];
        CUDA_CHECK(cudaMemcpy(h_keys_out, d_keys_out, sizeof(int) * num_items, cudaMemcpyDeviceToHost));
        
        std::vector<int> input_vec(h_keys_in, h_keys_in + num_items);
        std::vector<int> output_vec(h_keys_out, h_keys_out + num_items);
        std::vector<int> offsets_vec(h_segment_offsets, h_segment_offsets + num_segments + 1);
        
        bool passed = validate_segmented_sort(input_vec, output_vec, offsets_vec, "Empty Segments", false);
        std::cout << "Empty segments test: " << (passed ? "PASS" : "FAIL") << std::endl;
        
        CUDA_CHECK(cudaFree(d_keys_in));
        CUDA_CHECK(cudaFree(d_keys_out));
        CUDA_CHECK(cudaFree(d_segment_offsets));
        CUDA_CHECK(cudaFree(d_temp_storage));
    }
    
    // Test single element segments
    {
        std::cout << "\n--- Testing with single element segments ---" << std::endl;
        const int num_items = 5;
        const int num_segments = 5;
        
        int h_keys_in[num_items] = {5, 3, 1, 4, 2};
        int h_segment_offsets[num_segments + 1] = {0, 1, 2, 3, 4, 5};
        
        int *d_keys_in, *d_keys_out, *d_segment_offsets;
        CUDA_CHECK(cudaMalloc(&d_keys_in, sizeof(int) * num_items));
        CUDA_CHECK(cudaMalloc(&d_keys_out, sizeof(int) * num_items));
        CUDA_CHECK(cudaMalloc(&d_segment_offsets, sizeof(int) * (num_segments + 1)));
        
        CUDA_CHECK(cudaMemcpy(d_keys_in, h_keys_in, sizeof(int) * num_items, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_segment_offsets, h_segment_offsets, sizeof(int) * (num_segments + 1), cudaMemcpyHostToDevice));
        
        void *d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        
        CUDA_CHECK(cub::DeviceSegmentedMergeSort::SortKeys(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out,
            num_items, num_segments, d_segment_offsets, d_segment_offsets + 1));
        
        CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
        CUDA_CHECK(cub::DeviceSegmentedMergeSort::SortKeys(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out,
            num_items, num_segments, d_segment_offsets, d_segment_offsets + 1));
        
        int h_keys_out[num_items];
        CUDA_CHECK(cudaMemcpy(h_keys_out, d_keys_out, sizeof(int) * num_items, cudaMemcpyDeviceToHost));
        
        // Single element segments should remain unchanged
        bool passed = true;
        for (int i = 0; i < num_items; ++i) {
            if (h_keys_out[i] != h_keys_in[i]) {
                passed = false;
                break;
            }
        }
        
        std::cout << "Single element segments test: " << (passed ? "PASS" : "FAIL") << std::endl;
        
        CUDA_CHECK(cudaFree(d_keys_in));
        CUDA_CHECK(cudaFree(d_keys_out));
        CUDA_CHECK(cudaFree(d_segment_offsets));
        CUDA_CHECK(cudaFree(d_temp_storage));
    }
}

int main() {
    std::cout << "CUB DeviceSegmentedMergeSort Functionality Tests" << std::endl;
    std::cout << "================================================" << std::endl;
    
    test_basic_functionality();
    test_all_apis();
    test_edge_cases();
    
    std::cout << "\n=== All Functionality Tests Completed ===" << std::endl;
    
    return 0;
}