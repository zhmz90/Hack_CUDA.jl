
JULIA_DIR=/home/guo/julia
simple_stream: simple_stream.cu
	nvcc  simple_stream.cu -o  simple_stream -I $(JULIA_DIR)/src -I $(JULIA_DIR)/src/support \
		-I $(JULIA_DIR)/usr/include -L $(JULIA_DIR)/usr/lib -ljulia

