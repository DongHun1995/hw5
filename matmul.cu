#include <cstdio>
#include "matmul.h"
#include "util.h"

#include <cuda_runtime.h>
#include <mpi.h>

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

static __global__ void matmul_kernel(float *A, float *B, float *C, int M, int N, int K)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  if (i >= M || j >= N) return;
  float sum = 0.0;
  for (int k = 0; k < K; k++)
  {
    sum += A[i * K + k] * B[k * N + j];
  }
  C[i * N + j] = sum;
}

#define BLOCKS 4

static size_t Mbegin[BLOCKS], Mend[BLOCKS];
static cudaStream_t data_stream, calc_stream;
static cudaEvent_t events[BLOCKS];
static float *A_gpu, *B_gpu, *C_gpu;

void matmul_initialize(int M, int N, int K) 
{
  for (size_t i = 0; i < BLOCKS; i++)
  {
    Mbegin[i] = M / BLOCKS * i;
    Mend[i] = M / BLOCKS * (i + 1);
    if (i == BLOCKS - 1) Mend[i] = M;
  }

  CHECK_CUDA(cudaStreamCreate(&data_stream));
  CHECK_CUDA(cudaStreamCreate(&calc_stream));
  for (int i = 0; i < BLOCKS; i++)
  {
    CHECK_CUDA(cudaEventCreate(&events[i]));
  }
  CHECK_CUDA(cudaMalloc(&A_gpu, M * K * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&B_gpu, K * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&C_gpu, M * N * sizeof(float)));
}


void matmul(const float *A, const float *B, float *C, int M, int N, int K) 
{
  CHECK_CUDA(cudaMemcpyAsync(B_gpu, B, K * N * sizeof(float), cudaMemcpyHostToDevice, data_stream));
  for(int i = 0; i < BLOCKS; i++)
  {
    CHECK_CUDA(cudaMemcpyAsync(&A_gpu[Mbegin[i] * K], &A[Mbegin[i] * K], (Mend[i] - Mbegin[i]) * K * sizeof(float), cudaMemcpyHostToDevice, data_stream));
    CHECK_CUDA(cudaEventRecord(events[i], data_stream));
  }

  for (int i =0; i < BLOCKS; i++)
  {
    dim3 blockDim(32, 32);
    dim3 gridDim((Mend[i] - Mbegin[i] + 32 - 1) / 32, (N + 32 - 1) / 32);
    CHECK_CUDA(cudaStreamWaitEvent(calc_stream, events[i]));
    matmul_kernel<<<gridDim, blockDim, 0, calc_stream>>>(&A_gpu[Mbegin[i] * K], B_gpu, &C_gpu[Mbegin[i] * N], (Mend[i] - Mbegin[i]), N, K);
  }

  CHECK_CUDA(cudaStreamSynchronize(calc_stream));
  CHECK_CUDA(cudaMemcpyAsync(C, C_gpu, M * N * sizeof(float), cudaMemcpyDeviceToHost, data_stream));
  CHECK_CUDA(cudaStreamSynchronize(data_stream));

}


void matmul_finalize() 
{
  CHECK_CUDA(cudaFree(A_gpu));
  CHECK_CUDA(cudaFree(B_gpu));
  CHECK_CUDA(cudaFree(C_gpu));
  CHECK_CUDA(cudaStreamDestroy(data_stream));
  CHECK_CUDA(cudaStreamDestroy(calc_stream));
  for (int i =0; i < BLOCKS; i++)
  {
    CHECK_CUDA(cudaEventDestroy(events[i]));
  }
}
