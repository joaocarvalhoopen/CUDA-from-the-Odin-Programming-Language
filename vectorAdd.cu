#include <cuda_runtime.h>
#include <stdio.h>


// External kernel declaration
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements);

/*
typedef enum {
    HOST_TO_DEVICE = cudaMemcpyHostToDevice,
    DEVICE_TO_HOST = cudaMemcpyDeviceToHost,
    // ... add other cudaMemcpyKind types if needed ...
} MyMemcpyKind;
*/

extern "C" void my_cudaMalloc( void** devicePtr, size_t size ) {
    cudaMalloc( (void**)devicePtr, size );
}

/*
extern "C" void my_cudaMemcpy( void* dst, const void* src, size_t count, cudaMemcpyKind kind ) {
    cudaMemcpy( dst, src, count, kind );
}
*/

extern "C" void my_cudaMemcpy_host_to_device( void* dst, const void* src, size_t count ) {
    cudaMemcpy( dst, src, count, cudaMemcpyHostToDevice );
}

extern "C" void my_cudaMemcpy_device_to_host( void* dst, const void* src, size_t count ) {
    cudaMemcpy( dst, src, count, cudaMemcpyDeviceToHost );
}

extern "C" void my_cudaFree(void* devicePtr) {
    cudaFree( devicePtr );
}



extern "C" void runVectorAdd( const float *A, const float *B, float *C, int N ) {
    int threadsPerBlock = 256;  // or another suitable value
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    
    cudaDeviceSynchronize();  // wait for the kernel to finish
}



// -----------------------------------
// Kernel's

// const int N = 512;  // size of vectors

__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= 0 && i <= 5) {
        printf("Hello from inside the kernrl blockDim: %d [%d], block: [%d], thread: [%d] A[i]:%f  B[i]:%f \n",
            i,
            blockDim.x, blockIdx.x, threadIdx.x,
            A[i], B[i] );
    }

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }

    if (i >= 0 && i <= 5) {
        printf("A[i]: %f + B[i]: %f = C[i]: %f \n",
                A[i], B[i], C[i] );
    }
}
