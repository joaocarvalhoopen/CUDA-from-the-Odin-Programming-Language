// #include <iostream>
// #include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>


/*
typedef enum {
    HOST_TO_DEVICE = cudaMemcpyHostToDevice,
    DEVICE_TO_HOST = cudaMemcpyDeviceToHost,
    // ... add other cudaMemcpyKind types if needed ...
} MyMemcpyKind;
*/

extern void my_cudaMalloc( void** devicePtr, size_t size );

// extern void my_cudaMemcpy( void* dst, const void* src, size_t count, cudaMemcpyKind kind );

extern void my_cudaMemcpy_host_to_device( void* dst, const void* src, size_t count );

extern void my_cudaMemcpy_device_to_host( void* dst, const void* src, size_t count );

extern void my_cudaFree(void* devicePtr);

extern void runVectorAdd( const float *A, const float *B, float *C, int N );


// External kernel declaration
// extern /* __global__ */ void vectorAdd(const float *A, const float *B, float *C, int numElements);

const int N = 512;

int main()
{
    printf("Start of the CUDA_C program...\n");

    // Allocate host vectors
    // float *h_A = float[N];
    // float *h_B = float[N];
    // float *h_C = float[N];

    float *h_A = malloc( N * sizeof( float ) );
    float *h_B = malloc( N * sizeof( float ) );
    float *h_C = malloc( N * sizeof( float ) );

    // Initialize host vectors
    for (int i = 0; i < N; ++i)
    {
        // h_A[i] = rand() % 100;
        // h_B[i] = rand() % 100;

        h_A[i] = 1;
        h_B[i] = 2;
    }

    // Allocate device vectors
    float *d_A, *d_B, *d_C;
    my_cudaMalloc((void **)&d_A, N * sizeof(float));
    my_cudaMalloc((void **)&d_B, N * sizeof(float));
    my_cudaMalloc((void **)&d_C, N * sizeof(float));

    // Copy host vectors to device
    // cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Copy host vectors to device
    my_cudaMemcpy_host_to_device(d_A, h_A, N * sizeof(float) /* , MyMemcpyKind.HOST_TO_DEVICE */ /* cudaMemcpyHostToDevice */);
    my_cudaMemcpy_host_to_device(d_B, h_B, N * sizeof(float) /* , MyMemcpyKind.HOST_TO_DEVICE */ /* cudaMemcpyHostToDevice */);


    // Launch kernel
//    int threadsPerBlock = 256;
//    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
//    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Launch kernel
    runVectorAdd(d_A, d_B, d_C, N);

    // Copy result vector from device to host
    // cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Copy result vector from device to host
    my_cudaMemcpy_device_to_host( h_C, d_C, N * sizeof(float) /* , MyMemcpyKind.DEVICE_TO_HOST */ /*  cudaMemcpyDeviceToHost */ );


    // Print result
    for (int i = 0; i < N; ++i)
    {
        // std::cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << std::endl;
        printf( "%f + %f = %f\n", h_A[i], h_B[i], h_C[i] );
    }

    // Free device memory
    my_cudaFree(d_A);
    my_cudaFree(d_B);
    my_cudaFree(d_C);

    // Free host memory
    // delete[] h_A;
    // delete[] h_B;
    // delete[] h_C;

    // Free host memory
    free( h_A );
    free( h_B );
    free( h_C );

    printf("...end of the CUDA_C program.\n");

    return 0;
}
