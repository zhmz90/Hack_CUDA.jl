JULIA_DIR=/home/guo/julia
FLAGS= -I $(JULIA_DIR)/src -I $(JULIA_DIR)/src/support -I $(JULIA_DIR)/usr/include -L $(JULIA_DIR)/usr/lib -ljulia
atomic: atomic_intrinsics.cu
	nvcc atomic_intrinsics.cu -o atomic $(FLAGS)

test: array_julia.c
	nvcc array_julia.c $(FLAGS) -o test
