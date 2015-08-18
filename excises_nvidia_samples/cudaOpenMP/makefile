
JULIA_DIR=/home/guo/julia
cudaOpenMP: cudaOpenMP.cu
	nvcc -ccbin g++  -m64 cudaOpenMP.cu -Xcompiler -fopenmp -o  cudaOpenMP -gencode arch=compute_52,code=compute_52 -I $(JULIA_DIR)/src -I $(JULIA_DIR)/src/support   \
		-I $(JULIA_DIR)/usr/include -L $(JULIA_DIR)/usr/lib -ljulia

test_omp: test_omp.c
	gcc test_omp.c -o test_omp -fopenmp

clean:
	rm -f test_omp
	rm -f cudaOpenMP
