#ifndef CUDA_BF16_HPP
#define CUDA_BF16_HPP

#include "cuda_bf16.h"

namespace cuda {

    class BFloat16 {
    public:
        nv_bfloat16 value;

        __host__ __device__ BFloat16() : value(0) {}
        __host__ __device__ BFloat16(float f) { value = __float2half(f); }
        __host__ __device__ operator float() const { return __half2float(value); }
    };

} // namespace cuda

#endif // CUDA_BF16_HPP
