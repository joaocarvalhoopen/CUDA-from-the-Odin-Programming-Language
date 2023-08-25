# Variables

# Targets
all: my_cuda_odin_program my_cuda_c_program 

my_cuda_odin_program: libmyvectorAddlibrary.a main_cuda_odin.odin
	odin build . -out:my_cuda_odin_program

my_cuda_c_program: libmyvectorAddlibrary.a main.c
	clang -O3 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart main.c libmyvectorAddlibrary.a -o my_cuda_c_program

# With out external library only with different compilation units.
#	clang -O3 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart main.c vectorAdd.o -o my_cuda_c_program

libmyvectorAddlibrary.a: vectorAdd.cu
	nvcc -arch=sm_60 -c vectorAdd.cu -o vectorAdd.o
	ar crv libmyvectorAddlibrary.a vectorAdd.o

clean:
	rm vectorAdd.o libmyvectorAddlibrary.a my_cuda_c_program my_cuda_odin_program
