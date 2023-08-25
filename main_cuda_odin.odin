package cuda_odin

import "core:fmt"
import "core:strings"

import "core:c"

// External C library import.
// when ODIN_OS == .Linux   do foreign import foo {"libmyvectorAddlibrary.a", "system:/usr/local/cuda/lib64/libcudart.so" }

when ODIN_OS == .Linux   do foreign import foo {"libmyvectorAddlibrary.a", "system:cudart" }

foreign foo {
    my_cudaMalloc :: proc "c" ( devicePtr : ^rawptr, size : int ) ---

    my_cudaMemcpy_host_to_device :: proc "c" ( dst : rawptr, src : rawptr, count : int ) ---

    my_cudaMemcpy_device_to_host :: proc "c" ( dst : rawptr, src : rawptr, count : int ) ---

    my_cudaFree :: proc "c" ( devicePtr : rawptr ) ---

    runVectorAdd :: proc "c" ( A : [^]f32, B : [^]f32, C : [^]f32, N : int ) ---
}

main :: proc () {
    fmt.printf("Starting CUDA from Odin ... :-) ) ...\n")

    N :: 512

    h_A : ^[N]f32 = new( [N]f32 )
    h_B : ^[N]f32 = new( [N]f32 )
    h_C : ^[N]f32 = new( [N]f32 )

    // Initialize host vectors
    for _, i in h_A {
        // h_A[i] = rand() % 100;
        // h_B[i] = rand() % 100;

        h_A[i] = 1;
        h_B[i] = 2;
    }

    // Allocate device vectors
    d_A, d_B, d_C : [^]f32
    my_cudaMalloc( cast( ^rawptr ) & d_A, N * size_of( f32 ) )
    my_cudaMalloc( cast( ^rawptr ) & d_B, N * size_of( f32 ) )
    my_cudaMalloc( cast( ^rawptr ) & d_C, N * size_of( f32 ) )

    // my_cudaMalloc((void **)&d_C, N * size_of(float));


    // Copy host vectors to device
    // cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Copy host vectors to device
    my_cudaMemcpy_host_to_device( rawptr( d_A ), rawptr( h_A ), N * size_of( f32 ) )
    my_cudaMemcpy_host_to_device( rawptr( d_B ), rawptr( h_B ), N * size_of( f32 ) )


    // Launch kernel
//    int threadsPerBlock = 256;
//    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
//    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Launch kernel
    runVectorAdd( d_A, d_B, d_C, N )

    // Copy result vector from device to host
    // cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Copy result vector from device to host
    my_cudaMemcpy_device_to_host( rawptr( h_C ), rawptr( d_C ), N * size_of( f32 ) )


    // Print result
    for i in 0..<(N / 50) {
        fmt.printf( "%v + %v = %v\n", h_A[i], h_B[i], h_C[i] )
    }

    // Free device memory
    my_cudaFree( rawptr( d_A ) )
    my_cudaFree( rawptr( d_B ) )
    my_cudaFree( rawptr( d_C ) )

    // Free host memory
    // delete[] h_A;
    // delete[] h_B;
    // delete[] h_C;

    // Free host memory
    free( rawptr( h_A ) )
    free( rawptr( h_B ) )
    free( rawptr( h_C ) )

    fmt.printf("...end of the CUDA_ODIN program.\n")
}