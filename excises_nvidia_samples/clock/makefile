
JULIA_DIR=/home/guo/julia
simple_clock: simple_clock.cu
	nvcc  simple_clock.cu -o  simple_clock -gencode arch=compute_52,code=compute_52 -I $(JULIA_DIR)/src -I $(JULIA_DIR)/src/support -m64  \
		-I $(JULIA_DIR)/usr/include -L $(JULIA_DIR)/usr/lib -ljulia

