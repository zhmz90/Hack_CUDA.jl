INC=/home/guo/haplox/GPU/NVIDIA_CUDA-7.0_Samples/common/inc
JULIA_DIR=/home/guo/julia

call_back: simple_call_back.cu multithreading.o simple_call_back.o
	nvcc multithreading.o simple_call_back.o -o simple_call_back -gencode arch=compute_52,code=sm_52 -I $(JULIA_DIR)/src \
-I $(JULIA_DIR)/src/support  -I $(JULIA_DIR)/usr/include -L $(JULIA_DIR)/usr/lib -ljulia

multithreading.o : multithreading.cpp multithreading.h
	g++ -I $(INC) -m64 -c multithreading.cpp -o multithreading.o

simple_call_back.o : simple_call_back.cu  multithreading.o
	nvcc -ccbin g++ -I $(INC) -c simple_call_back.cu -m64 -o simple_call_back.o -gencode arch=compute_52,code=sm_52 -I $(JULIA_DIR)/src \
		-I $(JULIA_DIR)/src/support  -I $(JULIA_DIR)/usr/include -L $(JULIA_DIR)/usr/lib -ljulia


