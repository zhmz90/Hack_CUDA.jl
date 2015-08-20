sepCom: device_library.o device_library.a separate_comp.o
	nvcc -m64 -gencode arch=compute_52,code=sm_52 -o sepCom device_library.a separate_comp.o
device_library.o: device_library.cu
	nvcc -m64 -dc -gencode arch=compute_52,code=sm_52 -o device_library.o -c device_library.cu

device_library.a: device_library.o
	nvcc -m64 -lib -o device_library.a device_library.o 

separate_comp.o: separate_compilation.cu
	nvcc -m64 -dc -gencode arch=compute_52,code=sm_52 -o separate_comp.o -c separate_compilation.cu

clean:
	rm -f *.o *.a sepCom
