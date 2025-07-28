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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <stdio.h>
#include <iterator>
#include <functional>
#include <memory>
#include <mutex>

#include <cub/config.cuh>
#include <cub/util_deprecated.cuh>
#include <cub/util_device.cuh>

// Include moderngpu segsort implementation
#include "../../../moderngpu/src/moderngpu/kernel_segsort.hxx"

CUB_NAMESPACE_BEGIN

struct DeviceSegmentedMergeSort
{
  static constexpr bool PRINT_PROP = false;

  // Static initialization function - thread safe using Meyer's Singleton
  static mgpu::standard_context_t& GetContext(cudaStream_t stream)
  {
    // Check if stream is supported (currently only stream 0 is supported)
    if (stream != 0) {
      fprintf(stderr, "DeviceSegmentedMergeSort: Only stream 0 is currently supported. Got stream: %p\n", stream);
      // For now, fall back to stream 0 context
      stream = 0;
    }

    static std::unique_ptr<mgpu::standard_context_t> context;
    static std::mutex context_mutex;

    if (!context) {
      std::lock_guard<std::mutex> lock(context_mutex);
      if (!context) {
        context = std::make_unique<mgpu::standard_context_t>(PRINT_PROP, stream);
      }
    }
    return *context;
  }
  /******************************************************************//**
   * @name Key-value pairs
   *********************************************************************/
  template <typename KeyT,
            typename ValueT,
            typename BeginOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortPairs(KeyT *d_keys,
            ValueT *d_values,
            int num_items,
            int num_segments,
            BeginOffsetIteratorT d_begin_offsets,
            cudaStream_t stream = 0)
  {
    cudaError_t error = cudaSuccess;

    // Return early if empty problem
    if (num_items == 0 || num_segments == 0) {
        return error;
    }

    // Get thread-safe context
    auto& context = GetContext(stream);

    // Call moderngpu segmented sort on output arrays
    mgpu::segmented_sort(d_keys, d_values, num_items,
                        d_begin_offsets, num_segments, mgpu::less_t<KeyT>(), context);

    // Synchronize after everything
    if (CubDebug(error = cudaStreamSynchronize(stream))) return error;

    return error;
  }

  /******************************************************************//**
   * @name Key-value pairs descending
   *********************************************************************/
  template <typename KeyT,
            typename ValueT,
            typename BeginOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortPairsDescending(
                      KeyT *d_keys,
                      ValueT *d_values,
                      int num_items,
                      int num_segments,
                      BeginOffsetIteratorT d_begin_offsets,
                      cudaStream_t stream = 0)
  {
    cudaError_t error = cudaSuccess;

    // Return early if empty problem
    if (num_items == 0 || num_segments == 0) {
        return error;
    }

    // Get thread-safe context
    auto& context = GetContext(stream);

    // Call moderngpu segmented sort with descending comparison
    mgpu::segmented_sort(d_keys, d_values, num_items,
                        d_begin_offsets, num_segments, mgpu::greater_t<KeyT>(), context);


    // Synchronize if needed
    if (CubDebug(error = cudaStreamSynchronize(stream))) return error;

    return error;
  }

  /******************************************************************//**
   * @name Keys-only
   *********************************************************************/
  template <typename KeyT,
            typename BeginOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeys(KeyT *d_keys,
           int num_items,
           int num_segments,
           BeginOffsetIteratorT d_begin_offsets,
           cudaStream_t stream = 0)
  {
    cudaError_t error = cudaSuccess;

    // Return early if empty problem
    if (num_items == 0 || num_segments == 0) {
        return error;
    }

    // Get thread-safe context
    auto& context = GetContext(stream);

    // Call moderngpu segmented sort (keys only)
    mgpu::segmented_sort(d_keys, num_items,
                        d_begin_offsets, num_segments, mgpu::less_t<KeyT>(), context);


    // Synchronize if needed
    if (CubDebug(error = cudaStreamSynchronize(stream))) return error;

    return error;
  }

  /******************************************************************//**
   * @name Keys-only descending
   *********************************************************************/
  template <typename KeyT,
            typename BeginOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeysDescending(KeyT *d_keys,
           int num_items,
           int num_segments,
           BeginOffsetIteratorT d_begin_offsets,
           cudaStream_t stream = 0)
  {
    cudaError_t error = cudaSuccess;

    // Return early if empty problem
    if (num_items == 0 || num_segments == 0) {
        return error;
    }

    // Get thread-safe context
    auto& context = GetContext(stream);

    // Call moderngpu segmented sort (keys only, descending)
    mgpu::segmented_sort(d_keys, num_items,
                        d_begin_offsets, num_segments, mgpu::greater_t<KeyT>(), context);


    // Synchronize if needed
    if (CubDebug(error = cudaStreamSynchronize(stream))) return error;

    return error;
  }

  //@}  end member group

};

CUB_NAMESPACE_END
