# Patch b5050 23106f9  - 2025-04-05

Clone, checkout, change 6 files and create 2 new ones.

## 11 Steps

### 1. Clone repository

``` sh
git clone https://github.com/ggml-org/llama.cpp llama5050.cpp
cd llama5050.cpp
git checkout 23106f9
git checkout -b llamaCUDA
```

### 2. Add 3 lines below line 14

`nano CMakeLists.txt` insert 3 lines after line 14:

``` c
if(NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES 53)
endif()
```

### 3. Add 2 lines below line 274

`nano ggml/CMakeLists.txt`  insert 2 lines after line 274:

``` c
target_link_libraries(ggml PRIVATE stdc++fs)
add_link_options(-Wl,--copy-dt-needed-entries)
```

### 4. to 7. Remove *constexpr* and comment lines in 3 files

- `nano ggml/src/ggml-cuda/common.cuh` in line 455: remove **constexpr** after *static*
- `nano ggml/src/ggml-cuda/fattn-common.cuh` in line 623: add "//" in the front
- `nano ggml/src/ggml-cuda/fattn-vec-f32.cuh` in line 71: add "//" in the front
- `nano ggml/src/ggml-cuda/fattn-vec-f16.cuh` in line 73: add "//" in the front

### 8. Create `cuda_bf16.h`

`sudo nano /usr/local/cuda/include/cuda_bf16.h` and give it the content:

``` h
#ifndef CUDA_BF16_H
#define CUDA_BF16_H
#include <cuda_fp16.h>
// Define nv_bfloat16 as half
typedef half nv_bfloat16;
#endif // CUDA_BF16_H
```

### 9. Create `cuda_bf16.hpp`

`sudo nano /usr/local/cuda/include/cuda_bf16.hpp` and give it the content:

``` hpp
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
```

### 10. Prepare compilation

``` sh
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=ON -DCMAKE_CUDA_STANDARD=14 -DCMAKE_CUDA_STANDARD_REQUIRED=true -DGGML_CPU_ARM_ARCH=armv8-a -DGGML_NATIVE=off
```

### 11. Compile - 84 minutes

``` sh
cmake --build build --config Release
```

## Explanation for the steps

In addition to **b4400** we need 2 new files to define **bfloat16** in llama.cpp. We won't be using it anyways, since our old Maxwell 5.3 architecture does not support it. Since it's a 16bit format, we replace it with *half*. Creating two new files `cuda_bf16.h` and `cuda_bf16.hpp` is the easier and faster option.

- /usr/local/cuda/include/cuda_bf16.h
- /usr/local/cuda/include/cuda_bf16.hpp

The locations for edits in these 6 files have slightly changed:

- `nano CMakeLists.txt` 14
- `nano ggml/CMakeLists.txt` 274
- `nano ggml/src/ggml-cuda/common.cuh` 455
- `nano ggml/src/ggml-cuda/fattn-common.cuh` 623
- `nano ggml/src/ggml-cuda/fattn-vec-f32.cuh` 71
- `nano ggml/src/ggml-cuda/fattn-vec-f16.cuh` 73

``` sh
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=ON -DCMAKE_CUDA_STANDARD=14 -DCMAKE_CUDA_STANDARD_REQUIRED=true -DGGML_CPU_ARM_ARCH=armv8-a -DGGML_NATIVE=off
```
