/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <mma.h>
#include <cuda_runtime.h>
#include "ptx_instructions.cuh"

// the number of template waves that are computed at once
#define N_BLOCK (NBLOCK * 2)

extern "C" {

constexpr int block_size = 96;
constexpr int warp_size = 32;
constexpr int num_warps = block_size / warp_size;

constexpr int m_mma = 16;
constexpr int n_mma = 8;
constexpr int k_mma = 8;

//FIXME: enable the computation for k > 256
constexpr int num_k_mmas = 32;
//const int num_mma_groups = (*nm - 1) / (num_mmas_per_group * k_mma) + 1;

constexpr int tpl_sh_size = (num_k_mmas * k_mma) * m_mma;
constexpr int obs_sh_size = (num_k_mmas * k_mma) + (num_warps * N_BLOCK * n_mma);
constexpr int result_sh_size = m_mma * (num_warps * N_BLOCK * n_mma);
constexpr int sh_size_in_bytes = 4 * (tpl_sh_size + obs_sh_size + result_sh_size);

__device__ __forceinline__ uint32_t _float_to_tf32(float f)
{
    uint32_t x = *reinterpret_cast<uint32_t*>(&f);
    if (isfinite(f))
    {
        x += 0x1000u;
    }
    return x;
}

__device__ __forceinline__ void _load_obs(
    uint32_t* obs_sh, const float* obs, const int nt, const int offset, const int tid)
{
    constexpr int num_elems = 16 / sizeof(int); // ensure 16B access
    float buffer[num_elems];
    uint32_t tmp[num_elems];
    for (int i_base = 0; i_base < obs_sh_size; i_base += block_size * num_elems)
    {
        const int i = i_base + tid * num_elems;
        if (i >= obs_sh_size)
        {
            break;
        }
        _global_load(buffer, &obs[offset + i]);
        #pragma unroll
        for (int j = 0; j < num_elems; j++)
        {
            tmp[j] = _float_to_tf32(buffer[j]);
        }
        _shared_store(&obs_sh[i], tmp);
    }
}

__device__ __forceinline__ int _swizzle_smem_pos_in(int x, int y)
{
    constexpr int vector_size = 4; // ensure 128bit access
    constexpr int num_vectors_per_row = (num_k_mmas * k_mma) / vector_size;
    const int new_x = ((x / vector_size) ^ (y % num_vectors_per_row)) * vector_size;
    return new_x;
}

__device__ __forceinline__ int _swizzle_smem_pos_out(int x, int y)
{
    constexpr int vector_size = 2; // ensure 64bit access
    constexpr int num_vectors_per_row = n_mma / vector_size;
    const int new_x = ((x / vector_size) ^ (y % num_vectors_per_row)) * vector_size;
    return new_x;
}

__device__ __forceinline__ void _load_tpl(
    float* tpl_sh, const float* tpl, const int nm, const int width, const int tid)
{
    constexpr int num_elems = 16 / sizeof(int);
    for (int i_base = 0; i_base < tpl_sh_size; i_base += block_size * num_elems)
    {
        const int i = i_base + tid * num_elems;
        if (i >= tpl_sh_size)
        {
            break;
        }
            
        const int gid_template = i / width;
        const int gid_time = i % width;
        const int gid = gid_time + nm * gid_template;

        const int sid_template = gid_template;
        const int sid_time = i % width;
        const int swizzled_sid_time = _swizzle_smem_pos_in(sid_time, sid_template);
        const int sid = swizzled_sid_time + width * sid_template;

        float buffer[num_elems];
        uint32_t tmp[num_elems];
        _global_load(buffer, &tpl[gid]);
        #pragma unroll
        for (int j = 0; j < num_elems; j++)
        {
            tmp[j] = _float_to_tf32(buffer[j]);
        }
        _shared_store(&tpl_sh[sid], tmp);
    }
}

__device__ __forceinline__ void _load_a_segment(
    uint32_t* av, float* tpl, const int nm, const int k_iter, const int k_mma, const int laneid)
{
    const int k_offset = k_iter * k_mma;

    const int row = laneid % 16;
    const int col = k_offset + (laneid / 16) * 4;
    const int swizzled_col = _swizzle_smem_pos_in(col, row);
    const int pos = swizzled_col + row * nm;

    _load_a_matrix(av, &tpl[pos]);
}

__device__ __forceinline__ void _load_b_segment(
    uint32_t* bv, const uint32_t* obs_sh, const int local_n_iter, const int n_mma, const int local_k_iter,
    const int k_mma, const int tid, const int laneid)
{
    const int n_iter = local_n_iter + (tid / warp_size) * N_BLOCK;
    const int col = laneid % 4;
    const int row = laneid / 4; 
    const int pos = (local_k_iter * k_mma) + (n_iter * n_mma) + col + row;

    bv[0] = obs_sh[pos];
    bv[1] = obs_sh[pos + 4];
}

__device__ __forceinline__ void store_results_to_smem(
    float* sh, float dv[NBLOCK][4], int warpid, int laneid)
{
    for (int local_n = 0; local_n < N_BLOCK; local_n++)
    {
        const int sh_offset = (warpid * N_BLOCK + local_n) * n_mma;

        const int col = (laneid % 4) * 2;

        const int row0 = laneid / 4;
        const int new_col0 = _swizzle_smem_pos_out(col, row0);
        const int sid0 = sh_offset + new_col0 + row0 * (n_mma * N_BLOCK * num_warps);

        const int row1 = row0 + 8;
        const int new_col1 = _swizzle_smem_pos_out(col, row1);
        const int sid1 = sh_offset + new_col1 + row1 * (n_mma * N_BLOCK * num_warps);

        *reinterpret_cast<float2*>(&sh[sid0]) = *reinterpret_cast<float2*>(&dv[local_n][0]);
        *reinterpret_cast<float2*>(&sh[sid1]) = *reinterpret_cast<float2*>(&dv[local_n][2]);
    } //local_n
}

__device__ __forceinline__ void store_results_to_gmem(
    float* co, float* sh, int nt, int tid)
{
    constexpr int num_elems = 8 / sizeof(int);
    for (int i_base = 0; i_base < m_mma * (n_mma * N_BLOCK * num_warps); i_base += block_size * num_elems)
    {
        const int i = i_base + tid * num_elems;
        const int width = n_mma * N_BLOCK * num_warps;
        const int gl_offset = blockIdx.x * width;
            
        const int gid_template = i / width;
        const int gid_time = gl_offset + i % width;
        if (gid_time >= nt) {
            continue;
        }
        const int gid = gid_time + nt * gid_template;

        const int sid_template = gid_template;
        const int sid_time = i % width;
        const int swizzled_sid_time = _swizzle_smem_pos_out((sid_time % 8), sid_template);
        const int sid = (sid_time / 8) * 8 + swizzled_sid_time + width * sid_template;

        *reinterpret_cast<float2*>(&co[gid]) = *reinterpret_cast<float2*>(&sh[sid]);
    }
}

__global__ void compdef_gputc_ (
    const int *nt_in, const int *nm_in, const int *nc_in, const float *obs, const float *tpl, float *co)
{
    const int nt = *nt_in;
    const int nm = *nm_in;
    const int nc = *nc_in;

    const int tid = threadIdx.x;
    const int laneid = tid % warp_size;
    const int warpid = tid / warp_size;

    extern __shared__ float sh[];
    float* tpl_sh = sh;
    uint32_t* obs_sh = reinterpret_cast<uint32_t*>(&tpl_sh[tpl_sh_size]);
    float* result_sh = reinterpret_cast<float*>(&obs_sh[obs_sh_size]);

    float dv[N_BLOCK][4] = {{ 0.0f }};

    // load all template wave components to shared memory.
    _load_tpl(tpl_sh, tpl, nm, (k_mma * num_k_mmas), tid);

    constexpr int n_per_block = num_warps * N_BLOCK * n_mma;
    #pragma unroll 1
    for (int obs_global_offset = blockIdx.x * n_per_block; obs_global_offset <= nc; obs_global_offset += gridDim.x * n_per_block)
    {
        // load all the observation wave to shared memory.
        _load_obs(obs_sh, obs, nt, obs_global_offset, tid);

        __syncthreads();

        uint32_t av[4], bv[2][2];

        #pragma unroll 16
        for (int k_mma_iter = 0; k_mma_iter < num_k_mmas; k_mma_iter++)
        {
            // load matrix A
            _load_a_segment(av, tpl_sh, (k_mma * num_k_mmas), k_mma_iter, k_mma, laneid);

            // load matrix B
            int b_reg_stage = 0;
            _load_b_segment(bv[b_reg_stage], obs_sh, 0, n_mma, k_mma_iter, k_mma, tid, laneid);

            #pragma unroll
            for (int n_mma_iter = 0; n_mma_iter < N_BLOCK; n_mma_iter++)
            { 
                // load matrix B
                if (n_mma_iter < N_BLOCK - 1)
                {
                    _load_b_segment(bv[b_reg_stage ^ 1], obs_sh, n_mma_iter + 1, n_mma, k_mma_iter, k_mma, tid, laneid);
                }

                // matrix-matrix product
                _tc_matmul(dv[n_mma_iter], av, bv[b_reg_stage], dv[n_mma_iter]);

                b_reg_stage ^= 1;

            } //n_mma_iter
        } //k_mma_iter

        store_results_to_smem(result_sh, dv, warpid, laneid);

        __syncthreads();

        store_results_to_gmem(co, result_sh, nt, tid);
    }
}

void call_compdef_gputc_ (
    const int *nt, const int *nm, const int *nc, const float *obs, const float *tpl, float *co)
{
    int device_id;
    cudaGetDevice(&device_id);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    constexpr size_t dyn_sh_size = sh_size_in_bytes;
    cudaFuncSetAttribute(compdef_gputc_, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_sh_size);

    int num_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, compdef_gputc_, block_size, sh_size_in_bytes);
    int grid_size = num_blocks_per_sm * prop.multiProcessorCount;

    compdef_gputc_<<<grid_size, block_size, sh_size_in_bytes>>>
    (nt, nm, nc, obs, tpl, co);
}

}
