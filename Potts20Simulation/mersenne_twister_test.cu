#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "mt19937.h"

#define NNEIBORS 4 // number of nearest neighbors, is 4 for 2d lattice
#define BLOCKS 64 //64
#define THREADS 256 // 64
#define NGENERATORS BLOCKS * THREADS
#define LatticeType float

// Computes energy delta from flipping spin at site i to value e with neighbors a,b,c,d 
template <class T>
__global__	int del(T i, T a, T b, T c, T d, T e) {
	return (i == a) + (i == b) + (i == c) + (i == d) - (e == a) - (e == b) - (e == c) - (e == d);
}

__global__ void initLattice(curandStateMtgp32* state, LatticeType* devR, int N)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
//	float r;
	if (tid < NGENERATORS)
	{
		for (int i = tid; i < N; i += NGENERATORS)
		{
			devR[i] = curand(&state[blockIdx.x]);
		}
	}
}


int reset_timers(clock_t* previous, clock_t* current) // not pure code, but unimportante
{
	*previous = *current;
	*current = clock();
	return *current - *previous;
}

int main(int argc, char* argv[]) {
	// Parameters:
	clock_t begin = clock(), previous = begin, current = begin;
	// random number generation
	curandStateMtgp32* devMTGPStates;
	mtgp32_kernel_params* devKernelParams;
	LatticeType* R, * devR;
	int N = 10, seed = 0;

	// Allocate space for lattice on host 
	R = (LatticeType*)calloc(N, sizeof(LatticeType));

	// Allocate space for results on device 
	cudaMalloc((void**)&devR, N * sizeof(LatticeType));

	// Init MTRG
	cudaMalloc((void**)&devMTGPStates, BLOCKS * sizeof(curandStateMtgp32));
	cudaMalloc((void**)&devKernelParams, sizeof(mtgp32_kernel_params));
	curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, devKernelParams);
	curandMakeMTGP32KernelState(devMTGPStates, mtgp32dc_params_fast_11213, devKernelParams, BLOCKS, seed);

	std::cout << "Initialization took: " << reset_timers(&previous, &current) << " clocks;\n";
	//actually working part


	initLattice <<< BLOCKS, THREADS >>> (devMTGPStates, devR, N);
	cudaMemcpy(R, devR, N * sizeof(LatticeType), cudaMemcpyDeviceToHost);

	//end of acctually working part

	for (int i = 0; i < N; i++)
		std::cout << R[i] << " ";
	std::cout << "\n" << "Whole programm took: " << reset_timers(&previous, &begin) << " clocks;\n";

	free(R);
	cudaFree(devMTGPStates);
	cudaFree(devKernelParams);
	cudaFree(devR);
	return 0;
}
