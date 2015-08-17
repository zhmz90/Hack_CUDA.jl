
JULIA_DIR=/home/guo/julia
print_info: simple_printf.cu
	nvcc  simple_printf.cu -o  simple_printf -I $(JULIA_DIR)/src -I $(JULIA_DIR)/src/support \
		-I $(JULIA_DIR)/usr/include -L $(JULIA_DIR)/usr/lib -ljulia

