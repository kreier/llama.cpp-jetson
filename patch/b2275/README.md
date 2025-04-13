# Patch b2275 a33e6a0 2024-02-26

Only one file is changed, it's three lines in the `Makefile`. And then a call of `make` instead of `cmake`. It only takes **7 minutes** to compile.

- Makefile

```
make LLAMA_CUBLAS=1 CUDA_DOCKER_ARCH=sm_62 -j 6
```
