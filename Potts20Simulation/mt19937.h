// (c) Copyright 2013 Lev Barash, Landau Institute for Theoretical Physics, Russian Academy of Sciences
// This is supplement to the paper:
// L.Yu. Barash, L.N. Shchur, "PRAND: GPU accelerated parallel random number generation library: Using most reliable algorithms and applying parallelism of modern GPUs and CPUs".
// e-mail: barash @ itp.ac.ru (remove space)

#define mt19937_N 624

typedef struct{
  unsigned mt[mt19937_N];
  int mti;
} mt19937_state;

typedef struct{
  unsigned mt_aligned[3*mt19937_N+5] __attribute__ ((aligned(16)));
  unsigned out[3*mt19937_N+5] __attribute__ ((aligned(16)));
  unsigned *mt;
  int mti;
} mt19937_sse_state;

__host__ void mt19937_init_device_consts_();

__host__ void mt19937_free_device_consts_();

__device__ void mt19937_dev_skipahead_(mt19937_state* state, unsigned long long offset0,unsigned offset_log);

__host__ void mt19937_skipahead_(mt19937_state* state, unsigned long long offset0,unsigned offset_log);

__device__ __host__ void mt19937_init_(mt19937_state* state);

__device__ void mt19937_dev_init_sequence_(mt19937_state* state, unsigned long long SequenceNumber);

__host__ void mt19937_init_sequence_(mt19937_state* state, unsigned long long SequenceNumber);

__device__ __host__ unsigned mt19937_generate_(mt19937_state* state);

__device__ __host__ float mt19937_generate_uniform_float_(mt19937_state* state);

__host__ unsigned int mt19937_sse_generate_(mt19937_sse_state* state);

__device__ __host__ void mt19937_get_sse_state_(mt19937_state* state, mt19937_sse_state* sse_state);

__host__ void mt19937_print_state_(mt19937_state* state);

__host__ void mt19937_generate_array_(mt19937_state* state, unsigned int* out, long length);

__host__ void mt19937_generate_gpu_array_(mt19937_state* state, unsigned int* out, long length);

__host__ void mt19937_generate_gpu_array_float_(mt19937_state* state, float* out, long length);

__host__ void mt19937_generate_gpu_array_double_(mt19937_state* state, double* out, long length);
