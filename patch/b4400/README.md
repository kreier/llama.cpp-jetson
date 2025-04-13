# Patch b4400 6e1531a 2024-12-31

Just 6 files have to be edited, and the first call of `cmake` has to be adjusted:

- `nano CMakeLists.txt` 14
- `nano ggml/CMakeLists.txt` 249
- `nano ggml/src/ggml-cuda/common.cuh` 348
- `nano ggml/src/ggml-cuda/fattn-common.cuh` 532
- `nano ggml/src/ggml-cuda/fattn-vec-f32.cuh` 70
- `nano ggml/src/ggml-cuda/fattn-vec-f16.cuh` 73

``` sh
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=ON -DCMAKE_CUDA_STANDARD=14 -DCMAKE_CUDA_STANDARD_REQUIRED=true -DGGML_CPU_ARM_ARCH=armv8-a -DGGML_NATIVE=off
```
