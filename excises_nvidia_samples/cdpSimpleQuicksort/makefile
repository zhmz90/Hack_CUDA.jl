
JULIA_DIR=/home/guo/julia
quick_sort: quick_sort.cu
	nvcc  quick_sort.cu -o  quick_sort -I $(JULIA_DIR)/src -I $(JULIA_DIR)/src/support \
		-I $(JULIA_DIR)/usr/include -L $(JULIA_DIR)/usr/lib -ljulia

