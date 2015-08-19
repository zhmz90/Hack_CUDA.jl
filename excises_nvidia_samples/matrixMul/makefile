
JULIA_DIR=/home/guo/julia
inlinePTX : inlinePTX.cu
	nvcc  inlinePTX.cu -o inlinePTX -gencode arch=compute_52,code=sm_52 -I $(JULIA_DIR)/src \
		-I $(JULIA_DIR)/src/support  -I $(JULIA_DIR)/usr/include -L $(JULIA_DIR)/usr/lib -ljulia
