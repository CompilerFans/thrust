/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/******************************************************************************
 * Simple example of DeviceSegmentedMergeSort::SortPairs().
 *
 * Sorts an array of int keys paired with a corresponding array of int values,
 * organized into segments.
 *
 * To compile using the command line:
 *   nvcc --extended-lambda --expt-relaxed-constexpr -DTHRUST_IGNORE_CUB_VERSION_CHECK \
 *        -I../../cub -I../../moderngpu/src example_cub_style.cu -o example_cub_style
 *
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>
#include <algorithm>

#include <cub/util_allocator.cuh>
#include <cub/device/device_segmented_merge_sort.cuh>

using namespace cub;

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------

/**
 * Initialize problem
 */
void Initialize(
    int     *h_keys,
    int     *h_values,
    int     *h_segment_offsets,
    int     num_items,
    int     num_segments)
{
    for (int i = 0; i < num_items; ++i)
    {
        h_keys[i] = rand() % 1000;
        h_values[i] = i;
    }

    h_segment_offsets[0] = 0;
    for (int i = 1; i < num_segments; ++i)
    {
        h_segment_offsets[i] = (rand() % (num_items / num_segments)) + h_segment_offsets[i-1];
        if (h_segment_offsets[i] >= num_items)
            h_segment_offsets[i] = num_items - 1;
    }
    h_segment_offsets[num_segments] = num_items;

    // Ensure offsets are sorted
    std::sort(h_segment_offsets, h_segment_offsets + num_segments + 1);
}


/**
 * Solve problem
 */
void Solve(
    int     *h_keys,
    int     *h_values,
    int     *h_segment_offsets,
    int     *h_reference_keys,
    int     *h_reference_values,
    int     num_items,
    int     num_segments)
{
    for (int i = 0; i < num_items; ++i)
    {
        h_reference_keys[i] = h_keys[i];
        h_reference_values[i] = h_values[i];
    }

    // Sort each segment independently
    for (int seg = 0; seg < num_segments; ++seg)
    {
        int segment_begin = h_segment_offsets[seg];
        int segment_end = h_segment_offsets[seg + 1];
        int segment_size = segment_end - segment_begin;

        if (segment_size > 1)
        {
            // Create index array for this segment
            std::vector<int> indices(segment_size);
            for (int i = 0; i < segment_size; ++i)
                indices[i] = i;

            // Sort indices based on keys
            std::sort(indices.begin(), indices.end(), 
                [&](int a, int b) {
                    return h_keys[segment_begin + a] < h_keys[segment_begin + b];
                });

            // Apply sorted order to both keys and values
            for (int i = 0; i < segment_size; ++i)
            {
                h_reference_keys[segment_begin + i] = h_keys[segment_begin + indices[i]];
                h_reference_values[segment_begin + i] = h_values[segment_begin + indices[i]];
            }
        }
    }
}

//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

/**
 * Main
 */
int main(int argc, char** argv)
{
    int num_items       = 512;
    int num_segments    = 64;

    // Command line args
    if (argc > 1) num_items = atoi(argv[1]);
    if (argc > 2) num_segments = atoi(argv[2]);

    // Display test problem
    printf("cub::DeviceSegmentedMergeSort::SortPairs() %d items, %d segments\n", 
           num_items, num_segments);
    fflush(stdout);

    // Allocate host arrays
    int *h_keys           = (int*) malloc(sizeof(int) * num_items);
    int *h_values         = (int*) malloc(sizeof(int) * num_items);
    int *h_segment_offsets = (int*) malloc(sizeof(int) * (num_segments + 1));
    int *h_reference_keys = (int*) malloc(sizeof(int) * num_items);
    int *h_reference_values = (int*) malloc(sizeof(int) * num_items);
    int *h_keys_out       = (int*) malloc(sizeof(int) * num_items);
    int *h_values_out     = (int*) malloc(sizeof(int) * num_items);

    // Initialize problem and solution
    Initialize(h_keys, h_values, h_segment_offsets, num_items, num_segments);
    Solve(h_keys, h_values, h_segment_offsets, h_reference_keys, h_reference_values, num_items, num_segments);

    printf("Input keys: ");
    for (int i = 0; i < std::min(num_items, 20); ++i)
        printf("%d ", h_keys[i]);
    if (num_items > 20) printf("...");
    printf("\n");

    printf("Segment offsets: ");
    for (int i = 0; i <= num_segments; ++i)
        printf("%d ", h_segment_offsets[i]);
    printf("\n");

    // Allocate device arrays
    int *d_keys_in        = NULL;
    int *d_keys_out       = NULL;
    int *d_values_in      = NULL;
    int *d_values_out     = NULL;
    int *d_segment_offsets = NULL;

    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys_in, sizeof(int) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys_out, sizeof(int) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values_in, sizeof(int) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values_out, sizeof(int) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_segment_offsets, sizeof(int) * (num_segments + 1)));

    // Copy problem to device
    CubDebugExit(cudaMemcpy(d_keys_in, h_keys, sizeof(int) * num_items, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_values_in, h_values, sizeof(int) * num_items, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_segment_offsets, h_segment_offsets, sizeof(int) * (num_segments + 1), cudaMemcpyHostToDevice));

    // Allocate temporary storage
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    CubDebugExit(DeviceSegmentedMergeSort::SortPairs(d_temp_storage, temp_storage_bytes, 
        d_keys_in, d_keys_out, d_values_in, d_values_out, 
        num_items, num_segments, d_segment_offsets, d_segment_offsets + 1));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    printf("Temp storage required: %zu bytes\n", temp_storage_bytes);

    // Solve problem
    printf("Computing DeviceSegmentedMergeSort::SortPairs()...\n");
    CubDebugExit(DeviceSegmentedMergeSort::SortPairs(d_temp_storage, temp_storage_bytes, 
        d_keys_in, d_keys_out, d_values_in, d_values_out, 
        num_items, num_segments, d_segment_offsets, d_segment_offsets + 1));

    // Copy result back to host
    CubDebugExit(cudaMemcpy(h_keys_out, d_keys_out, sizeof(int) * num_items, cudaMemcpyDeviceToHost));
    CubDebugExit(cudaMemcpy(h_values_out, d_values_out, sizeof(int) * num_items, cudaMemcpyDeviceToHost));

    printf("Output keys: ");
    for (int i = 0; i < std::min(num_items, 20); ++i)
        printf("%d ", h_keys_out[i]);
    if (num_items > 20) printf("...");
    printf("\n");

    // Verify solution
    printf("Checking result...\n");
    bool correct = true;
    for (int i = 0; i < num_items; ++i)
    {
        if (h_keys_out[i] != h_reference_keys[i] || h_values_out[i] != h_reference_values[i])
        {
            printf("INCORRECT at index %d: got key=%d,value=%d, expected key=%d,value=%d\n", 
                   i, h_keys_out[i], h_values_out[i], h_reference_keys[i], h_reference_values[i]);
            correct = false;
            break;
        }
    }

    if (correct)
    {
        printf("CORRECT\n");
    }

    // Cleanup
    if (h_keys) free(h_keys);
    if (h_values) free(h_values);
    if (h_segment_offsets) free(h_segment_offsets);
    if (h_reference_keys) free(h_reference_keys);
    if (h_reference_values) free(h_reference_values);
    if (h_keys_out) free(h_keys_out);
    if (h_values_out) free(h_values_out);
    if (d_keys_in) CubDebugExit(g_allocator.DeviceFree(d_keys_in));
    if (d_keys_out) CubDebugExit(g_allocator.DeviceFree(d_keys_out));
    if (d_values_in) CubDebugExit(g_allocator.DeviceFree(d_values_in));
    if (d_values_out) CubDebugExit(g_allocator.DeviceFree(d_values_out));
    if (d_segment_offsets) CubDebugExit(g_allocator.DeviceFree(d_segment_offsets));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

    printf("\n");
    return 0;
}