#include <cub/device/device_segmented_merge_sort.cuh>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

using namespace cub;

// CPU reference implementation (similar to moderngpu style)
std::vector<int> cpu_segsort(const std::vector<int>& data,
  const std::vector<int>& segments) {

  std::vector<int> copy = data;
  int cur = 0;
  for(int seg = 0; seg < segments.size() - 1; ++seg) {
    int next = segments[seg + 1];
    std::sort(copy.data() + cur, copy.data() + next);
    cur = next;
  }
  return copy;
}

// Generate random data (moderngpu style)
std::vector<int> fill_random(int min_val, int max_val, int count, bool sorted) {
    std::vector<int> data(count);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(min_val, max_val);
    
    for(int i = 0; i < count; ++i) {
        data[i] = dis(gen);
    }
    
    if(sorted) {
        std::sort(data.begin(), data.end());
    }
    
    return data;
}

int main(int argc, char** argv) {

  for(int count = 1000; count < 23456789; count += count / 10) {

    for(int it = 1; it <= 10; ++it) {

      int num_segments = std::max(1, count / 100);
      
      // Generate segment offsets (moderngpu style)
      std::vector<int> segs_host = fill_random(0, count - 1, num_segments, true);
      segs_host.push_back(count); // Add final offset
      
      // Generate random data
      std::vector<int> host_data = fill_random(0, 100000, count, false);
      std::vector<int> host_values(count);
      
      // Initialize values as indices  
      for(int i = 0; i < count; ++i) {
          host_values[i] = i;
      }

      // Allocate device memory
      int *d_keys_in, *d_keys_out, *d_values_in, *d_values_out, *d_segments;
      cudaMalloc(&d_keys_in, sizeof(int) * count);
      cudaMalloc(&d_keys_out, sizeof(int) * count);
      cudaMalloc(&d_values_in, sizeof(int) * count);
      cudaMalloc(&d_values_out, sizeof(int) * count);
      cudaMalloc(&d_segments, sizeof(int) * (num_segments + 1));

      // Copy data to device
      cudaMemcpy(d_keys_in, host_data.data(), sizeof(int) * count, cudaMemcpyHostToDevice);
      cudaMemcpy(d_values_in, host_values.data(), sizeof(int) * count, cudaMemcpyHostToDevice);
      cudaMemcpy(d_segments, segs_host.data(), sizeof(int) * (num_segments + 1), cudaMemcpyHostToDevice);

      // Allocate temporary storage
      void *d_temp_storage = nullptr;
      size_t temp_storage_bytes = 0;
      
      DeviceSegmentedMergeSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out, d_values_in, d_values_out,
        count, num_segments, d_segments, d_segments + 1);
      
      cudaMalloc(&d_temp_storage, temp_storage_bytes);
      
      // Perform segmented sort
      DeviceSegmentedMergeSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out, d_values_in, d_values_out,
        count, num_segments, d_segments, d_segments + 1);

      // Copy results back
      std::vector<int> sorted(count);
      std::vector<int> host_indices(count);
      cudaMemcpy(sorted.data(), d_keys_out, sizeof(int) * count, cudaMemcpyDeviceToHost);
      cudaMemcpy(host_indices.data(), d_values_out, sizeof(int) * count, cudaMemcpyDeviceToHost);

      // Get CPU reference
      std::vector<int> ref = cpu_segsort(host_data, segs_host);

      // Check that the indices are correct.
      bool indices_correct = true;
      for(int i = 0; i < count; ++i) {
        if(sorted[i] != host_data[host_indices[i]]) {
          printf("count = %8d it = %3d KEY FAILURE at index %d\n", count, it, i);
          indices_correct = false;
          break;
        }
      }

      // Check that the keys are sorted.
      bool success = ref == sorted;
      printf("count = %8d it = %3d segments = %6d %s\n", count, it, num_segments,
        success && indices_correct ? "SUCCESS" : "FAILURE");

      if(!success || !indices_correct) {
        // Print some debug info for failures
        printf("  First few expected: ");
        for(int i = 0; i < std::min(10, count); ++i) {
          printf("%d ", ref[i]);
        }
        printf("\n");
        printf("  First few actual:   ");
        for(int i = 0; i < std::min(10, count); ++i) {
          printf("%d ", sorted[i]);
        }
        printf("\n");
        break;
      }

      // Cleanup
      cudaFree(d_keys_in);
      cudaFree(d_keys_out);
      cudaFree(d_values_in);
      cudaFree(d_values_out);
      cudaFree(d_segments);
      cudaFree(d_temp_storage);
    }
    
    if(count > 1000000) break; // Limit for demo
  }

  return 0;
}