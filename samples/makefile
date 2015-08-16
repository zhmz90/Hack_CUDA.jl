JULIA_DIR="/home/guo/julia"

test: *.cu *.h
	nvcc samples_hacking.cu -o test
embedding_julia: *.c
	clang -o test_julia embedding_julia.c  -I $(JULIA_DIR)/src -I $(JULIA_DIR)/src/support \
	-I $(JULIA_DIR)/usr/include  -L $(JULIA_DIR)/usr/lib  -ljulia
