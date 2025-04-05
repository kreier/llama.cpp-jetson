# Patch files to llama.cpp on Jetson Nano 2019

Instead of manually changing files for different builds this folder contains the edited files for each build.

## b5050 2025-04-05

In addition to b4400 we need 2 new files to define **bfloat16** in llama.cpp. We won't be using it anyways, since our old Maxwell 5.3 architecture does not support it. Since it's a 16bit format, we replace it with *half*. Creating two new files `cuda_bf16.h` and `cuda_bf16.hpp` is the easier and faster option.

- /usr/local/cuda/include/cuda_bf16.h
- /usr/local/cuda/include/cuda_bf16.hpp

The locations for edits in these 6 files have slightly changed:

- CMakeLists.txt 14
- ggml/CMakeLists.txt 274
- ggml/src/ggml-cuda/common.cuh 455
- ggml/src/ggml-cuda/fattn-common.cuh 523
- ggml/src/ggml-cuda/fattn-vec-f32.cuh 70
- ggml/src/ggml-cuda/template-instances/../fattn-vec-f16.cuh 73

## b4400 2024-12-31

Just 6 files have to be edited, and the first call of `cmake` has to be adjusted:

``` sh
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=ON -DCMAKE_CUDA_STANDARD=14 -DCMAKE_CUDA_STANDARD_REQUIRED=true -DGGML_CPU_ARM_ARCH=armv8-a -DGGML_NATIVE=off
```

- CMakeLists.txt 14
- ggml/CMakeLists.txt 274
- ggml/src/ggml-cuda/common.cuh 455
- ggml/src/ggml-cuda/fattn-common.cuh 632
- ggml/src/ggml-cuda/fattn-vec-f32.cuh 71
- ggml/src/ggml-cuda/template-instances/../fattn-vec-f16.cuh 73
