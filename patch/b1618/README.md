# Patch b1618 81bc921 2023-12-07

Add 5 lines into one file:

- ggml-cuda.cu

```
mkdir build && cd build
cmake .. -DLLAMA_CUBLAS=ON
make -j 2
```
