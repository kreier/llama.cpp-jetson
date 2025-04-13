# Patch b5050 23106f9 2025-04-05

In addition to **b4400** we need 2 new files to define **bfloat16** in llama.cpp. We won't be using it anyways, since our old Maxwell 5.3 architecture does not support it. Since it's a 16bit format, we replace it with *half*. Creating two new files `cuda_bf16.h` and `cuda_bf16.hpp` is the easier and faster option.

- /usr/local/cuda/include/cuda_bf16.h
- /usr/local/cuda/include/cuda_bf16.hpp

The locations for edits in these 6 files have slightly changed:

- `nano CMakeLists.txt` 14
- `nano ggml/CMakeLists.txt 274
- `nano ggml/src/ggml-cuda/common.cuh 455
- `nano ggml/src/ggml-cuda/fattn-common.cuh 623
- `nano ggml/src/ggml-cuda/fattn-vec-f32.cuh 71
- `nano ggml/src/ggml-cuda/fattn-vec-f16.cuh 73

``` sh
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=ON -DCMAKE_CUDA_STANDARD=14 -DCMAKE_CUDA_STANDARD_REQUIRED=true -DGGML_CPU_ARM_ARCH=armv8-a -DGGML_NATIVE=off
```
