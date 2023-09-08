#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <iomanip>
//CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
using namespace std;

//calculate offset from row,col and ld in row-major matrix,ld is the width of the matrix
__device__ __forceinline__ int OFFSET(int row,int col,int ld)
{
    return row*ld+col;
}

__device__ __forceinline__ float* ADDR(float* BaseAddress,size_t pitch,int Row,int Column)
{
    return ((float*)((char*)BaseAddress + Row * pitch) + Column);
}

//transfer float4
//transfer float4
__device__ __forceinline__  float4& FETCH_FLOAT4(float &pointer)
{
    return reinterpret_cast<float4*>(&pointer)[0];
}

//该函数不能显示调用处的行数
// __host__ __device__ __forceinline__ void checkCudaErrors(cudaError_t e)
// {
//     if(e!=cudaSuccess)
//         printf("%s %d CUDA: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));
// }

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}

// K: ldA
// N: ldB
template <
    const int BLOCK_SIZE_M,  // height of block of C that each thread block calculate
    const int BLOCK_SIZE_K,  // width of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,  // width of block of C that each thread block calculate
    const int THREAD_SIZE_Y, // height of block of C that each thread calculate
    const int THREAD_SIZE_X,  // width of block of C that each thread calculate
    const bool ENABLE_DOUBLE_BUFFER // whether enable double buffering or not
    > 
__global__ void gemm( 
    float * __restrict__ A,
    float * __restrict__ B,
    float * __restrict__ C, 
    const int M,
    const int N,
    const int K,
    size_t pitch_A,
    size_t pitch_B,
    size_t pitch_C) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // the threads number in Block of X,Y
    const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;

    // thread id in cur Block
    const int tid = ty * THREAD_X_PER_BLOCK + tx;

    // shared memory
    __shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];
    // registers for C
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};
    // registers for A and B
    float frag_a[2][THREAD_SIZE_Y];
    float frag_b[2][THREAD_SIZE_X];
    // registers load global memory
    const int ldg_num_a = BLOCK_SIZE_M * BLOCK_SIZE_K / (THREAD_NUM_PER_BLOCK * 4);
    const int ldg_num_b = BLOCK_SIZE_K * BLOCK_SIZE_N / (THREAD_NUM_PER_BLOCK * 4);
    float ldg_a_reg[4*ldg_num_a];
    float ldg_b_reg[4*ldg_num_b];

    // threads number in one row
    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

    // row number and col number that needs to be loaded by this thread
    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

    const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4; 
    const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;

    // row stride that thread uses to load multiple rows of a tile
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    const int BLOCK_SIZE_M_limit=M%BLOCK_SIZE_M;
    const int BLOCK_SIZE_N_limit=N%BLOCK_SIZE_N;

    int THREAD_SIZE_Y_limit=THREAD_SIZE_Y;
    int THREAD_SIZE_X_limit=THREAD_SIZE_X;
    if(by==M/BLOCK_SIZE_M)
    {
        if(ty*THREAD_SIZE_X>=BLOCK_SIZE_M_limit)
        {
            THREAD_SIZE_Y_limit=0;
        }
        else if((ty+1)*THREAD_SIZE_Y>BLOCK_SIZE_M_limit)
        {
            THREAD_SIZE_Y_limit=BLOCK_SIZE_M_limit-ty*THREAD_SIZE_Y;
        }
    }
    if(bx== N/BLOCK_SIZE_N)
    {       
        if(tx*THREAD_SIZE_X>=BLOCK_SIZE_N_limit)
        {
            THREAD_SIZE_X_limit=0;
        }
        else if((tx+1)*THREAD_SIZE_X>BLOCK_SIZE_N_limit)
        {
            THREAD_SIZE_X_limit=BLOCK_SIZE_N_limit-tx*THREAD_SIZE_X;
        }
    }

    A=ADDR(A,pitch_A,by*BLOCK_SIZE_M,0);
    B=ADDR(B,pitch_B,0,bx*BLOCK_SIZE_N);  

    //transfer first tile from global mem to shared mem
    // load A from global memory to shared memory
    if(by*BLOCK_SIZE_M+A_TILE_ROW_START<M)
    {
        FETCH_FLOAT4(ldg_a_reg[0]) = FETCH_FLOAT4(*ADDR(
        A,
        pitch_A,
        A_TILE_ROW_START,
        A_TILE_COL));
        As[0][A_TILE_COL][A_TILE_ROW_START]=ldg_a_reg[0];
        As[0][A_TILE_COL+1][A_TILE_ROW_START]=ldg_a_reg[1];
        As[0][A_TILE_COL+2][A_TILE_ROW_START]=ldg_a_reg[2];
        As[0][A_TILE_COL+3][A_TILE_ROW_START]=ldg_a_reg[3];
    }
    
    
    // load B from global memory to shared memory  
    if(B_TILE_ROW_START<K)
    FETCH_FLOAT4(Bs[0][B_TILE_ROW_START][B_TILE_COL]) = FETCH_FLOAT4(*ADDR(
        B,
        pitch_B,
        B_TILE_ROW_START,
        B_TILE_COL));
        
    __syncthreads();
    // load A from shared memory to register
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
        FETCH_FLOAT4(frag_a[0][thread_y]) = FETCH_FLOAT4(As[0][0][THREAD_SIZE_Y * ty + thread_y]);
    }
    // load B from shared memory to register
    #pragma unroll
    for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
        FETCH_FLOAT4(frag_b[0][thread_x]) = FETCH_FLOAT4(Bs[0][0][THREAD_SIZE_X * tx + thread_x]);
    }

    int write_stage_idx = 1;
    int load_stage_idx;

    for(int tile_idx=BLOCK_SIZE_K;tile_idx<K;tile_idx += BLOCK_SIZE_K){
        // load next tile from global mem
        if(by*BLOCK_SIZE_M+A_TILE_ROW_START<M)
            FETCH_FLOAT4(ldg_a_reg[0]) = FETCH_FLOAT4(*ADDR(
                A,
                pitch_A,
                A_TILE_ROW_START,
                A_TILE_COL + tile_idx));
        if(tile_idx + B_TILE_ROW_START<K)        
            FETCH_FLOAT4(ldg_b_reg[0]) = FETCH_FLOAT4(*ADDR(
                B,
                pitch_B,
                tile_idx + B_TILE_ROW_START,
                B_TILE_COL));
                          

        load_stage_idx = write_stage_idx ^ 1;

        
        #pragma unroll
        for(int j=0; j<BLOCK_SIZE_K-1; ++j){
            // load next tile from shared mem to register 
            // load A from shared memory to register
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
                FETCH_FLOAT4(frag_a[(j+1)%2][thread_y]) = FETCH_FLOAT4(As[load_stage_idx][j+1][THREAD_SIZE_Y * ty + thread_y]);
            }
            // load B from shared memory to register
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
                FETCH_FLOAT4(frag_b[(j+1)%2][thread_x]) = FETCH_FLOAT4(Bs[load_stage_idx][j+1][THREAD_SIZE_X * tx + thread_x]);
            }
            // compute C THREAD_SIZE_X x THREAD_SIZE_Y
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                #pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                    accum[thread_y][thread_x] += frag_a[j%2][thread_y] * frag_b[j%2][thread_x];
                }
            }
        }

        // load A from global memory to shared memory
        As[write_stage_idx][A_TILE_COL][A_TILE_ROW_START]=ldg_a_reg[0];
        As[write_stage_idx][A_TILE_COL+1][A_TILE_ROW_START]=ldg_a_reg[1];
        As[write_stage_idx][A_TILE_COL+2][A_TILE_ROW_START]=ldg_a_reg[2];
        As[write_stage_idx][A_TILE_COL+3][A_TILE_ROW_START]=ldg_a_reg[3];

        // load B from global memory to shared memory
        FETCH_FLOAT4(Bs[write_stage_idx][B_TILE_ROW_START][B_TILE_COL]) = FETCH_FLOAT4(ldg_b_reg[0]);

        // use double buffer, only need one sync
        __syncthreads();
        // switch
        write_stage_idx ^= 1;

        // load first tile from shared mem to register of next iter
        // load A from shared memory to register
        #pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
            FETCH_FLOAT4(frag_a[BLOCK_SIZE_K%2][thread_y]) = FETCH_FLOAT4(As[load_stage_idx^1][0][THREAD_SIZE_Y * ty + thread_y]);
        }
        // load B from shared memory to register
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
            FETCH_FLOAT4(frag_b[BLOCK_SIZE_K%2][thread_x]) = FETCH_FLOAT4(Bs[load_stage_idx^1][0][THREAD_SIZE_X * tx + thread_x]);
        }
        //compute last tile mma THREAD_SIZE_X x THREAD_SIZE_Y
        #pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                accum[thread_y][thread_x] += frag_a[(BLOCK_SIZE_K+1)%2][thread_y] * frag_b[(BLOCK_SIZE_K+1)%2][thread_x];
            }
        }    
    }


    load_stage_idx = write_stage_idx ^ 1;

    #pragma unroll
    for(int j=0; j<7; ++j){
    // load next tile from shared mem to register 
    // load A from shared memory to register
        #pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
            FETCH_FLOAT4(frag_a[(j+1)%2][thread_y]) = FETCH_FLOAT4(As[load_stage_idx][j+1][THREAD_SIZE_Y * ty + thread_y]);
        }
        // load B from shared memory to register
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
            FETCH_FLOAT4(frag_b[(j+1)%2][thread_x]) = FETCH_FLOAT4(Bs[load_stage_idx][j+1][THREAD_SIZE_X * tx + thread_x]);
        }
        // compute C THREAD_SIZE_X x THREAD_SIZE_Y
        #pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                if(thread_y<THREAD_SIZE_Y_limit&&thread_x<THREAD_SIZE_X_limit)
                    accum[thread_y][thread_x] += frag_a[j%2][thread_y] * frag_b[j%2][thread_x];
            }
        }
    }
    // switch
    write_stage_idx ^= 1;

    //compute last tile mma THREAD_SIZE_X x THREAD_SIZE_Y
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
            if(thread_y<THREAD_SIZE_Y_limit&&thread_x<THREAD_SIZE_X_limit)
                accum[thread_y][thread_x] += frag_a[1][thread_y] * frag_b[1][thread_x];
        }
    }

    // store back to C
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
            if(thread_y<THREAD_SIZE_Y_limit&&thread_x<THREAD_SIZE_X_limit)
                C[OFFSET(
                    BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y,
                    BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x,
                    N)] =accum[thread_y][thread_x];
            }
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("usage: ./main [M] [K] [N]\n");
        exit(0);
    }
    size_t M = atoi(argv[1]);
    size_t K = atoi(argv[2]);
    size_t N = atoi(argv[3]);

    size_t bytes_A = sizeof(float) * M * K;
    size_t bytes_B = sizeof(float) * K * N;
    size_t bytes_C = sizeof(float) * M * N;
    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);
    float* h_C = (float*)malloc(bytes_C);
    float* h_C1 = (float*)malloc(bytes_C);

    float* d_A;
    float* d_B;
    float* d_C;

    checkCudaErrors(cudaMalloc(&d_A, bytes_A));
    checkCudaErrors(cudaMalloc(&d_B, bytes_B));
    checkCudaErrors(cudaMalloc(&d_C, bytes_C));
    double msecPerMatrixMul[2] = {0, 0};
    double gigaFlops[2] = {0, 0};
    double flopsPerMatrixMul = 2.0 * M * N * K;

    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_K = 8;
    const int BLOCK_SIZE_N = 128;
    const int THREAD_SIZE_X = 8;
    const int THREAD_SIZE_Y = 8;
    const bool ENABLE_DOUBLE_BUFFER = false;

    // generate A
    for( int i = 0; i < M * K; i++ ){
        h_A[i] = i / 13;
    }

    // generate B
    for( int i = 0; i < K * N; i++ ) {
        h_B[i] = i % 13;
    }

    

    checkCudaErrors(cudaMemcpy( d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    float* d_A_p;
    float* d_B_p;
    float* d_C_p;

    size_t pitch_A;
    size_t pitch_B;
    size_t pitch_C;

    checkCudaErrors(cudaMallocPitch((void**)&d_A_p,&pitch_A,K*sizeof(float),M));
    checkCudaErrors(cudaMallocPitch((void**)&d_B_p,&pitch_B,N*sizeof(float),K));
    checkCudaErrors(cudaMallocPitch((void**)&d_C_p,&pitch_C,N*sizeof(float),M));

    checkCudaErrors(cudaMemcpy2D(d_A_p,pitch_A,h_A,K*sizeof(float),K*sizeof(float),M,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy2D(d_B_p,pitch_B,h_B,N*sizeof(float),N*sizeof(float),K,cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msecTotal = 0;
    int nIter = 1000;


    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0 ; run < nIter; run ++ ) {
        dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
        dim3 dimGrid((N%BLOCK_SIZE_N)?N / BLOCK_SIZE_N+1:N/BLOCK_SIZE_N, 
        (M%BLOCK_SIZE_M)?M / BLOCK_SIZE_M+1:M/BLOCK_SIZE_M);
        gemm<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X, ENABLE_DOUBLE_BUFFER> 
        <<< dimGrid, dimBlock >>>(d_A_p, d_B_p, d_C, M, N, K,pitch_A,pitch_B,pitch_C);
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));


    checkCudaErrors(cudaMemcpy( h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    // checkCudaErrors(cudaMemcpy2D(h_C,N*sizeof(float),d_C_p,pitch_C,pitch_C,M,cudaMemcpyDeviceToHost));

    msecPerMatrixMul[0] = msecTotal / nIter;
    gigaFlops[0] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0] / 1000.0f);
    printf( "My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        gigaFlops[0],
        msecPerMatrixMul[0],
        flopsPerMatrixMul);

    // cublas
    cublasHandle_t blas_handle;  
    cublasCreate(&blas_handle);
    float alpha = 1.0;
    float beta = 0;
    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0 ; run < nIter; run ++ ) {
        cublasSgemm (blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 
            N,M, K, &alpha, 
            d_B, N,d_A, K,  &beta, d_C, N
        );
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    checkCudaErrors(cudaMemcpy( h_C1, d_C, bytes_C, cudaMemcpyDeviceToHost));

    msecPerMatrixMul[1] = msecTotal / nIter;
    gigaFlops[1] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[1] / 1000.0f);
    printf( "CuBlas Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        gigaFlops[1],
        msecPerMatrixMul[1],
        flopsPerMatrixMul);

    cublasDestroy(blas_handle); 
    
    double eps = 1.e-6;  // machine zero
    bool correct = true;

    for (int i = 0; i < M * N; i++) {
        int row = i / N;
        int col = i % N;
        double abs_err = fabs(h_C[i] - h_C1[i]);
        double dot_length = M;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps) {
            printf("Error! row=%d,col=%d,Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    row,col,i, h_C[i], h_C1[i], eps);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
    printf("ratio= %f\n", gigaFlops[0] / gigaFlops[1]);

    // Free Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_A_p);
    cudaFree(d_B_p);
    cudaFree(d_C_p);
    
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C1);
}

