JULIA_DIR=/home/guo/julia
JULIA_INC= -I $(JULIA_DIR)/src -I $(JULIA_DIR)/src/support -I $(JULIA_DIR)/usr/include -L $(JULIA_DIR)/usr/lib -ljulia

NV_INC= -I /home/guo/haplox/GPU/NVIDIA_CUDA-7.0_Samples/common/inc
IPC : simple_ipc.cu
	nvcc simple_ipc.cu -o IPC  $(JULIA_INC) $(NV_INC)


