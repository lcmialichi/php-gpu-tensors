CUDA_SRCS = src/cuda/float_kernels.cu src/cuda/activation_kernels.cu src/cuda/matmul_kernels.cu src/cuda/concat_kernels.cu src/cuda/where_kernels.cu src/cuda/broadcast_ops.cu src/cuda/scalar_ops.cu src/cuda/unary_ops.cu src/cuda/reduction_ops.cu src/cuda/indexed_ops.cu src/cuda/factory_kernels.cu src/cuda/nn_backend.cu
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)

NVCC_FLAGS = -arch=$(CUDA_ARCH_FLAG) -O3 --use_fast_math -Xcompiler -fPIC $(CUDA_CUBLAS_FLAG) $(CUDA_BACKEND_FLAGS)

-include $(CUDA_OBJS:.o=.d)

.PHONY: cuda-flags-check
cuda-flags-check:
	@printf '%s\n' '$(NVCC) $(NVCC_FLAGS)' > .cuda-flags.tmp; \
	if ! cmp -s .cuda-flags.tmp .cuda-flags; then mv .cuda-flags.tmp .cuda-flags; else rm .cuda-flags.tmp; fi

.cuda-flags: cuda-flags-check

$(CUDA_OBJS): .cuda-flags

./cuda.la: libcudakernels.a

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -MMD -MP -MF $(@:.o=.d) -c -o $@ $<

libcudakernels.a: $(CUDA_OBJS)
	rm -f $@
	ar rcs $@ $^
