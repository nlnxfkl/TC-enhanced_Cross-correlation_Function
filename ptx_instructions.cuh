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

#pragma once

#include <cstdint>

__device__ __forceinline__ void _tc_matmul(
    float* dv, uint32_t const* av, uint32_t const* bv, float const* cv)
{
    asm volatile("{\n\t" 
        "mma.sync.aligned.row.col.m16n8k8.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n\t"
        "}"
        : "=f"(dv[0]), "=f"(dv[1]), "=f"(dv[2]), "=f"(dv[3])
        : "r"(av[0]), "r"(av[1]), "r"(av[2]), "r"(av[3]),
          "r"(bv[0]), "r"(bv[1]),
          "f"(cv[0]), "f"(cv[1]), "f"(cv[2]), "f"(cv[3]));
}

__device__ __forceinline__ void _global_load(void* D, const void *ptr)
{
    uint4 *data = reinterpret_cast<uint4 *>(D);
    asm volatile("{\n\t"
        "ld.global.L2::128B.v4.u32 {%0, %1, %2, %3}, [%4];\n\t"
        "}"
        : "=r"(data->x), "=r"(data->y), "=r"(data->z), "=r"(data->w)
        : "l"(ptr));
}

__device__ __forceinline__ uint32_t _get_smem_pointer(const void* ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void _shared_load(void *dst, void* ptr)
{
    uint32_t smem_ptr = _get_smem_pointer(ptr);
    uint4 *dst_u128 = reinterpret_cast<uint4 *>(dst);
    asm volatile("{\n\t"
        "ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n\t"
        "}"
        : "=r"(dst_u128->x), "=r"(dst_u128->y), "=r"(dst_u128->z), "=r"(dst_u128->w)
        : "r"(smem_ptr));
}

__device__ __forceinline__ void _shared_store(void* ptr, void const *src)
{
    uint32_t smem_ptr = _get_smem_pointer(ptr);
    uint4 const *dst_u128 = reinterpret_cast<uint4 const *>(src);
    asm volatile("{\n\t"
        "st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n\t"
        "}"
        : : "r"(smem_ptr),
        "r"(dst_u128->x), "r"(dst_u128->y), "r"(dst_u128->z), "r"(dst_u128->w));
}

__device__ __forceinline__ void _load_a_matrix(void *dst, const void* ptr)
{
    uint32_t smem_ptr = _get_smem_pointer(ptr);
    uint4 *dst_u128 = reinterpret_cast<uint4 *>(dst);
    asm volatile("{\n\t"
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n\t"
        "}"
        : "=r"(dst_u128->x), "=r"(dst_u128->y), "=r"(dst_u128->z), "=r"(dst_u128->w)
        : "r"(smem_ptr));
}

__device__ __forceinline__ void _async_copy(void *dst, const void* global_ptr)
{
    constexpr int size_in_bytes = 16;
    uint32_t smem_ptr = _get_smem_pointer(dst);
    asm volatile("{\n\t"
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n\t"
        "}"
        : : "r"(smem_ptr), "l"(global_ptr), "n"(size_in_bytes));
}

__device__ __forceinline__ void _async_commit()
{
    asm volatile("{\n\t"
        "cp.async.commit_group ;\n\t"
        "}"
        : : );
}

__device__ __forceinline__ void _async_wait()
{
    asm volatile("{\n\t"
        "cp.async.wait_group 1;\n\t"
        "}"
        : : );
}
