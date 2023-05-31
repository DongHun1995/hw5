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

#define TS 64
#define WPT 4
#define RTS TS / WPT


static __global__ void matmul_kernel(float *A, float *B, float *C, int M, int N, int K)
{  
  int globalRow = blockDim.y * blockIdx.y + threadIdx.y;
  int globalCol = WPT * blockDim.x * blockIdx.x + threadIdx.x;
  int row = threadIdx.y;
  int col = threadIdx.x;

  __shared__ float Asub[TS][TS];
  __shared__ float Bsub[TS][TS];

  float acc[WPT];
  for (int i =0; i < WPT; i++)
  {
    acc[i] = 0.0;
  }

  for (int offset =0; offset < K; offset += TS)
  {
    int tiledRow = offset + row;
    int tiledCol = offset + col;

    for (int i=0; i < WPT; i++)
    {
      Asub[row][col + i * RTS] = A[globalRow * K + (tiledCol + i * RTS)];
      Bsub[row][col + i * RTS] = B[tiledRow * N + (globalCol + i * RTS)];
    }

    __syncthreads();

    for (int k=0; k < TS; ++k)
    {
      for (int i =0; i < WPT; i++)
      {
        acc[i] += Asub[row][k] * Bsub[k][col + i * RTS];
      }
    }
    __syncthreads();
  }
  for (int i =0; i < WPT; i++)
  {
    C[globalRow * N + (globalCol + i * RTS)] = acc[i];
  }
  
}

#define NGPU 4
#define EVENTS_PER_GPU 1 //INCREASE as needed

static size_t Mbegin[NGPU], Mend[NGPU];
static size_t ngpu;
static cudaStream_t streams[NGPU];
static cudaEvent_t events[NGPU][EVENTS_PER_GPU];
static float *A_gpu[NGPU], *B_gpu[NGPU], *C_gpu[NGPU];

static int mpi_rank, mpi_world_size;
static int Node_M;
static float *Node_A, *Node_B, *Node_C;

void matmul_initialize(int M, int N, int K) 
{
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

  Node_M = M / mpi_world_size;
  alloc_mat(&Node_A, Node_M, K);
  alloc_mat(&Node_C, Node_M, N);

  ngpu = 4;

  for (size_t i = 0; i < ngpu; i++)
  {
    Mbegin[i] = Node_M / ngpu * i;
    Mend[i] = Node_M / ngpu * (i + 1);
    if (i == ngpu - 1) Mend[i] = Node_M;
  }

  for (size_t i = 0; i < ngpu; i++)
  {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaStreamCreate(&streams[i]));
    for (int j=0; j < EVENTS_PER_GPU; j++)
    {
      CHECK_CUDA(cudaEventCreate(&events[i][j]));
    }
  }

  for (size_t i =0; i < ngpu; i++)
  {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMalloc(&A_gpu[i], (Mend[i] - Mbegin[i]) * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&B_gpu[i], K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&C_gpu[i], (Mend[i] - Mbegin[i]) * N * sizeof(float)));
  }

}


void matmul(const float *A, const float *B, float *C, int M, int N, int K) 
{
  Node_B = (float *)B;
  MPI_Request req1, req2;

  MPI_Iscatter(A, Node_M * K, MPI_FLOAT, Node_A, Node_M * K, MPI_FLOAT, 0, MPI_COMM_WORLD, &req1);

  MPI_Ibcast(Node_B, K * N, MPI_FLOAT, 0, MPI_COMM_WORLD, &req2);

  MPI_Wait(&req1, MPI_STATUS_IGNORE);
  for (size_t i =0; i < ngpu; i++)
  {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMemcpyAsync(A_gpu[i], &Node_A[Mbegin[i] * K], (Mend[i] - Mbegin[i]) * K * sizeof(float), cudaMemcpyHostToDevice, streams[i]));
  }

  MPI_Wait(&req2, MPI_STATUS_IGNORE);
    for (size_t i =0; i < ngpu; i++)
  {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMemcpyAsync(B_gpu[i], Node_B, K * N * sizeof(float), cudaMemcpyHostToDevice, streams[i]));
  }

  for (size_t i =0; i < ngpu; i++)
  {
    CHECK_CUDA(cudaSetDevice(i));
    dim3 blockDim(TS / WPT, TS);
    dim3 gridDim((N + TS - 1) / TS, (Mend[i] - Mbegin[i] + TS - 1) / TS);
    matmul_kernel<<<gridDim, blockDim>>>(A_gpu[i], B_gpu[i], C_gpu[i], Mend[i] - Mbegin[i], N, K);
    CHECK_CUDA(cudaGetLastError());
  }

  for(size_t i =0; i < ngpu; i++)
  {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMemcpyAsync(&Node_C[Mbegin[i] * N], C_gpu[i], (Mend[i] - Mbegin[i]) * N * sizeof(float), cudaMemcpyDeviceToHost, streams[i]));
  }

  for (size_t i =0; i < ngpu; i++)
  {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaStreamSynchronize(streams[i]));
  }

  MPI_Gather(Node_C, Node_M * N, MPI_FLOAT, C, Node_M * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
}


void matmul_finalize() 
{
  for(size_t i = 0; i < ngpu; i++)
  {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaFree(A_gpu[i]));
    CHECK_CUDA(cudaFree(B_gpu[i]));
    CHECK_CUDA(cudaFree(C_gpu[i]));
    CHECK_CUDA(cudaStreamDestroy(streams[i]));
    for (int j = 0; j < EVENTS_PER_GPU; j++)
    {
      CHECK_CUDA(cudaEventDestroy(events[i][j]));
    }
  }
  cudaFreeHost(Node_A);
  cudaFreeHost(Node_C);
}