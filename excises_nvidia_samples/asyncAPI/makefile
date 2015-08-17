
JULIA_DIR="/home/guo/julia"
asyncAPI: asyncAPI.cu
	nvcc asyncAPI.cu -o asyncAPI -I $(JULIA_DIR)/src -I $(JULIA_DIR)/src/support \
		-I $(JULIA_DIR)/usr/include -L $(JULIA_DIR)/usr/lib -ljulia

