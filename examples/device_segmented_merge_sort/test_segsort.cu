#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

#include <cub/device/device_segmented_merge_sort.cuh>

// Helper function to divide up with rounding up
static int div_up(int a, int b) {
  return (a + b - 1) / b;
}

// CPU reference implementation for segmented sort
static std::vector<int> cpu_segsort(const std::vector<int>& data,
                             const std::vector<int>& segments) {
  std::vector<int> copy = data;
  int cur = 0;
  for(int seg = 0; seg < segments.size(); ++seg) {
    int next = segments[seg];
    std::sort(copy.data() + cur, copy.data() + next);
    cur = next;
  }
  std::sort(copy.data() + cur, copy.data() + data.size());
  return copy;
}

// Helper function to generate random segments
std::vector<int> generate_random_segments(int count, int num_segments) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(1, count / num_segments);

  std::vector<int> segments(num_segments);
  int current_pos = 0;

  for(int i = 0; i < num_segments - 1; ++i) {
    int segment_size = dis(gen);
    current_pos += segment_size;
    if(current_pos >= count) {
      current_pos = count - 1;
    }
    segments[i] = current_pos;
  }
  segments[num_segments - 1] = count;

  return segments;
}

int main(int argc, char** argv) {
  // Initialize CUDA
  cudaError_t error = cudaSetDevice(0);
  if (error != cudaSuccess) {
    std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  for(int count = 1000; count < 23456789; count += count / 10) {
    for(int it = 1; it <= 10; ++it) {

      int num_segments = div_up(count, 100);

      // Generate test data on host
      std::vector<int> h_keys_in(count);
      std::vector<int> h_values_in(count);
      std::vector<int> h_segment_offsets_full(num_segments + 1);

      // Generate random keys on host
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<> key_dist(0, 100000);

      for(int i = 0; i < count; ++i) {
        h_keys_in[i] = key_dist(gen);
        h_values_in[i] = i;  // Use indices as values
      }

      // Generate segments on host
      std::vector<int> h_segment_offsets = generate_random_segments(count, num_segments);
      h_segment_offsets_full[0] = 0;
      for(int i = 0; i < num_segments; ++i) {
        h_segment_offsets_full[i + 1] = h_segment_offsets[i];
      }

      // Allocate device memory
      int *d_keys, *d_values, *d_segment_offsets;
      cudaMalloc(&d_keys, sizeof(int) * count);
      cudaMalloc(&d_values, sizeof(int) * count);
      cudaMalloc(&d_segment_offsets, sizeof(int) * (num_segments + 1));

      // Copy data to device
      cudaMemcpy(d_keys, h_keys_in.data(), sizeof(int) * count, cudaMemcpyHostToDevice);
      cudaMemcpy(d_values, h_values_in.data(), sizeof(int) * count, cudaMemcpyHostToDevice);
      cudaMemcpy(d_segment_offsets, h_segment_offsets_full.data(), sizeof(int) * (num_segments + 1), cudaMemcpyHostToDevice);

      // Call DeviceSegmentedMergeSort
      error = cub::DeviceSegmentedMergeSort::SortPairs(
        d_keys, d_values, count, num_segments, d_segment_offsets);

      if (error != cudaSuccess) {
        std::cerr << "DeviceSegmentedMergeSort failed: " << cudaGetErrorString(error) << std::endl;
        return -1;
      }

      // Copy results back to host
      std::vector<int> sorted_keys(count);
      std::vector<int> sorted_values(count);
      cudaMemcpy(sorted_keys.data(), d_keys, sizeof(int) * count, cudaMemcpyDeviceToHost);
      cudaMemcpy(sorted_values.data(), d_values, sizeof(int) * count, cudaMemcpyDeviceToHost);

      // Generate reference result
      std::vector<int> ref = cpu_segsort(h_keys_in, h_segment_offsets);

      // Check that the indices are correct
      for(int i = 0; i < count; ++i) {
        if(sorted_keys[i] != h_keys_in[sorted_values[i]]) {
          printf("count = %8d it = %3d KEY FAILURE\n", count, it);
          exit(0);
        }
      }

      // Check that the keys are sorted correctly
      bool success = (ref == sorted_keys);
      printf("count = %8d it = %3d %s\n", count, it,
             success ? "SUCCESS" : "FAILURE");

      if(!success) {
        exit(0);
      }

      // Clean up device memory
      cudaFree(d_keys);
      cudaFree(d_values);
      cudaFree(d_segment_offsets);
    }
  }

  return 0;
}