
JULIA_DIR=/home/guo/julia
simple_assert: simple_assert.cu
	nvcc  simple_assert.cu -o  simple_assert -I $(JULIA_DIR)/src -I $(JULIA_DIR)/src/support \
		-I $(JULIA_DIR)/usr/include -L $(JULIA_DIR)/usr/lib -ljulia

