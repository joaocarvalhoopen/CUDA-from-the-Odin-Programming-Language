# CUDA from the Odin Programming Language
This is a example of how to call CUDA Kernels from the Odin programming language.

## Description
Simply put all the kernels and CUDA functions that you use inside the file .cu or several files, then export every function as a external C function, compile in NVCC for a static lib, and in the Odin program, simply import the external library that you made import the CUDA runtime library libcudart.so so that the linker can use it. Then inside, when passing parameters to the function do just some casting glue that you can even wrap around a function that you can inline in Odin. And that's it!!! <br>
The project has the static library that contains the kernels and the exported calls, then it has the Odin host program, and a c version of the program. This is for Linux.

## How to compile it

``` bash
$ make
```

## How to run it

``` bash
$ ./my_cuda_odin_program
$ ./my_cuda_c_program
```

## License
MIT Open Source License

## Have fun
Best regards, <br>
Joao Carvalho
