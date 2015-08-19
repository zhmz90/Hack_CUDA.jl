
JULIA_DIR=/home/guo/julia
quick_sort.o: quick_sort.cu
	nvcc  -c quick_sort.cu -o  quick_sort.o  -dc -gencode arch=compute_52,code=sm_52 -lcudadevrt -I $(JULIA_DIR)/src -I $(JULIA_DIR)/src/support \
		-I $(JULIA_DIR)/usr/include -L $(JULIA_DIR)/usr/lib -ljulia

quick_sort: quick_sort.o
	nvcc  quick_sort.o -o  quick_sort  -gencode arch=compute_52,code=sm_52 -lcudadevrt -I $(JULIA_DIR)/src -I $(JULIA_DIR)/src/support \
		-I $(JULIA_DIR)/usr/include -L $(JULIA_DIR)/usr/lib -ljulia

clean:
	rm -f quick_sort

