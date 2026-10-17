CUDA_SRCS = src/backends/cuda/float_kernels.cu src/backends/cuda/activation_kernels.cu src/backends/cuda/matmul_kernels.cu src/backends/cuda/concat_kernels.cu src/backends/cuda/where_kernels.cu src/backends/cuda/broadcast_ops.cu src/backends/cuda/scalar_ops.cu src/backends/cuda/unary_ops.cu src/backends/cuda/reduction_ops.cu src/backends/cuda/indexed_ops.cu src/backends/cuda/factory_kernels.cu src/backends/cuda/nn_backend.cu
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)

NVCC_FLAGS = -arch=$(CUDA_ARCH_FLAG) -O3 --use_fast_math -Xcompiler -fPIC $(CUDA_CUBLAS_FLAG) $(CUDA_BACKEND_FLAGS)

-include $(CUDA_OBJS:.o=.d)

.PHONY: cuda-flags-check
cuda-flags-check:
	@printf '%s\n' '$(NVCC) $(NVCC_FLAGS)' > .cuda-flags.tmp; \
	if ! cmp -s .cuda-flags.tmp .cuda-flags; then mv .cuda-flags.tmp .cuda-flags; else rm .cuda-flags.tmp; fi

.cuda-flags: cuda-flags-check

$(CUDA_OBJS): .cuda-flags

./phptensors.la: libcudakernels.a

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -MMD -MP -MF $(@:.o=.d) -c -o $@ $<

libcudakernels.a: $(CUDA_OBJS)
	rm -f $@
	ar rcs $@ $^
