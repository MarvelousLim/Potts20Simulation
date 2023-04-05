#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <curand_kernel.h>

#include <time.h>

#include "minepar.h"

// check errors
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

/*-----------------------------------------------------------------------------------------------------------
		Name agreement:

		s					array of all spins in spin-replica order
		SLF					spin lookup function for alone replica
		E					array of energies of replicas
		R					fixed population size (number of replicas)
		r					current replica number
		L					linear size of lattice
		N					fixed number of spins (N=L^2)
		j					current index inside alone replica
		q					Potts model parameter
		e					new spin value (char)
		n_i					neibors indexes inside alone replica
		n					neibors spin values
		replicaFamily		family (index of source replica after number of resamples); used to measure rho t
		rho_t				wtf value for checking equilibrium quality; A suitable condition is that rho_t << R
		energyOrder			ordering array, used during resampling step of algorithm
		MaxHistNumber		Maximum number of replicas to crease histogram from
		update              index of new replica to replace with

		To avoid confusion, lets describe the placement of the spin-replica array.
		This three-dimensional structure (L_x * L_y * R) lies like one-dimensional array,
		first goes, one by one, strings of first replica, then second etc. Here we calculate
		everything inside one replica, adding r factor later

		Also, when its about generation random numbers, we use R threads, one for each replica

-------------------------------------------------------------------------------------------------------------*/

curandStatePhilox4_32_10_t* devStates;
int BLOCKS;
int THREADS;

void BLTH(int BL, int TH){
	BLOCKS=BL;THREADS=TH;
}
void cudaMPImalloc(void** ptr, size_t size){
	gpuErrchk( cudaMalloc(ptr, size) );
	
}
void cudaMPIfree(void* ptr){
	gpuErrchk( cudaFree(ptr) );
}
void cudaMPImallocdevstate(int size){
	gpuErrchk( cudaMalloc((void**)&devStates, (size_t)size*sizeof(curandStatePhilox4_32_10_t)) );
}

void cudaPeekAtLastErrorMPI(){
	gpuErrchk(cudaPeekAtLastError());
}
void cudaDeviceSynchronizeMPI(){
	gpuErrchk(cudaDeviceSynchronize());
}
void cudaMPIend(){
	cudaMPIfree(devStates);
}

__host__ __device__ struct neibors_indexes SLF(int j, int L, int N) {
	struct neibors_indexes result;
	result.up = (j - L + N) % N; // N member is for positivity
	result.right = (j + 1) % L + L * (j / L);
	result.down = (j + L) % N;
	result.left = (j - 1 + L) % L + L * (j / L); // L member is for positivity
	return result;
}

__device__ struct neibors get_neibors_values(char* s, struct neibors_indexes n_i, int replica_shift) {
	return {
		s[n_i.up + replica_shift],
		s[n_i.right + replica_shift],
		s[n_i.down + replica_shift],
		s[n_i.left + replica_shift]
	};
}

__host__ __device__ struct energy_parts localEnergyParts(char currentSpin, struct neibors n) {
	// Computes energy of spin i with neighbors a, b, c, d 
	// it summirezes each join 2 times
	return {
		- (currentSpin * n.up)
		- (currentSpin * n.right)
		- (currentSpin * n.down)
		- (currentSpin * n.left)
		, (currentSpin * currentSpin)
	};
}

__device__ struct energy_parts addEnergyParts(struct energy_parts A, struct energy_parts B) {
	return { A.Ising + B.Ising, A.Blume + B.Blume };
}

__device__ struct energy_parts subEnergyParts(struct energy_parts A, struct energy_parts B) {
	return { A.Ising - B.Ising, A.Blume - B.Blume };
}

__device__ struct energy_parts calcEnergyParts(char* s, float* E, int L, int N, float D, int r) {
	struct energy_parts sum = { 0, 0 };
	for (int j = 0; j < N; j++) {
		// do not forget double joint summarization!
		int replica_shift = r * N;
		char i = s[j + replica_shift]; // current spin value
		struct neibors_indexes n_i = SLF(j, L, N);
		struct neibors n = get_neibors_values(s, n_i, replica_shift); // we look into r replica and j spin
		struct energy_parts tmp = localEnergyParts(i, n);
		sum = addEnergyParts(sum, tmp);
	}
	return sum;
}

__device__ float calcEnergyFromParts(struct energy_parts energyParts, float D) {
	return (energyParts.Ising / 2) + (D * energyParts.Blume); // div 2 because of double joint summarization
}

__global__ void deviceEnergy(char* s, float* E, int L, int N, float D) {
	int r = threadIdx.x + blockIdx.x * blockDim.x;
	struct energy_parts sum = calcEnergyParts(s, E, L, N, D, r);
	E[r] = calcEnergyFromParts(sum, D); 
}

// hardcoded spin suggestion for init
__device__ char suggestSpin(curandStatePhilox4_32_10_t* state, int r) {
	return curand(&state[r]) % 3 - 1;
}

// hardcoded spin suggestion for equilibration
__device__ char suggestSpinSwap(curandStatePhilox4_32_10_t* state, int r, char currentSpin) {
	return (currentSpin + 2 + (curand(&state[r]) % 2)) % 3 - 1; // little trick
}

__device__ float warpReduceSum(float val)
{
	for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
		val += __shfl_down_sync(FULL_MASK, val, offset);
	return val;
}


__global__ void equilibrate(curandStatePhilox4_32_10_t* state, char* s, float* E, int L, int N, int R, int q, int nSteps, float U, float D, bool heat){//, int* acceptance_number) {
	/*---------------------------------------------------------------------------------------------
		Main Microcanonical Monte Carlo loop.  Performs update sweeps on each replica in the
		population;
		There, one could change calcEnergyParts for system of carrying arrays of energy parts,
		but:
			1. This is not the bottleneck (which is for loop over N * nSteps
	---------------------------------------------------------------------------------------------*/

	int r = threadIdx.x + blockIdx.x * blockDim.x;
	int replica_shift = r * N;

	struct energy_parts baseEnergyParts = calcEnergyParts(s, E, L, N, D, r);

	for (int k = 0; k < N * nSteps; k++)
	{
		int j = curand(&state[r]) % N;
		char currentSpin = s[j + replica_shift];
		char suggestedSpin = suggestSpinSwap(state, r, currentSpin);
		//char suggestedSpin = curand(&state[r]) % 3 - 1;
		struct neibors_indexes n_i = SLF(j, L, N);
		struct neibors n = get_neibors_values(s, n_i, replica_shift);
		struct energy_parts suggestedLocalEnergyParts = localEnergyParts(suggestedSpin, n);

		struct energy_parts currentLocalEnergyParts = localEnergyParts(currentSpin, n);
		struct energy_parts deltaLocalEnergyParts = subEnergyParts(suggestedLocalEnergyParts, currentLocalEnergyParts);
		//local energy delta calculated for single spin; but should be for whole lattice
		//thus we need to count Ising part twice - for the neibors change in energy as well
		//but not Blume part!
		deltaLocalEnergyParts.Ising *= 2;
		struct energy_parts suggestedEnergyParts = addEnergyParts(baseEnergyParts, deltaLocalEnergyParts);
		float suggestedEnergy = calcEnergyFromParts(suggestedEnergyParts, D);

		/*
		if (r == 0) {
			printf("thread: %i reports:\n", r);
			printf("\tj: %i \n", j);
			printf("\tcurrentSpin: %i \n", currentSpin);
			printf("\tsuggestedSpin: %i \n", suggestedSpin);
			printf("\tn_i.up: %i \n", n_i.up);
			printf("\tn_i.right: %i \n", n_i.right);
			printf("\tn_i.down: %i \n", n_i.down);
			printf("\tn_i.left: %i \n", n_i.left);
			printf("\tn.up: %i \n", n.up);
			printf("\tn.right: %i \n", n.right);
			printf("\tn.down: %i \n", n.down);
			printf("\tn.left: %i \n", n.left);
			printf("\tbaseEnergyParts: %i %i\n", baseEnergyParts.Ising, baseEnergyParts.Blume);
			printf("\tsuggestedLocalEnergyParts: %i %i\n", suggestedLocalEnergyParts.Ising, suggestedLocalEnergyParts.Blume);
			printf("\tcurrentLocalEnergyParts: %i %i\n", currentLocalEnergyParts.Ising, currentLocalEnergyParts.Blume);
			printf("\tdeltaLocalEnergyParts: %i %i\n", deltaLocalEnergyParts.Ising, deltaLocalEnergyParts.Blume);
			printf("\tsuggestedEnergyParts: %i %i\n", suggestedEnergyParts.Ising, suggestedEnergyParts.Blume);
			printf("\tsuggestedEnergy: %f \n", suggestedEnergy);
			printf("\tcondition result: %i \n",
				((!heat && (suggestedEnergy + EPSILON < U)) || (heat && (suggestedEnergy - EPSILON > U))));
			printf("thread: %i report end;\n", r);
		}
		*/
		
		if (( !heat && (suggestedEnergy + EPSILON < U) ) || (heat && (suggestedEnergy - EPSILON > U) )) {
			baseEnergyParts = suggestedEnergyParts;
			E[r] = suggestedEnergy;
			s[j + replica_shift] = suggestedSpin;
			/*
			float reductionRes = warpReduceSum(1);
			if ((threadIdx.x & (warpSize - 1)) == 0)
				atomicAdd(acceptance_number, reductionRes);
			*/

		}
	}
}
void equilibrateMPI(char* deviceSpin, float* deviceE, int L, int N, int R, int q, int nSteps, float U, float D, bool heat) {
	equilibrate <<< BLOCKS, THREADS >>> (devStates, deviceSpin, deviceE, L, N, R, q, nSteps, U, D, heat);
}



__global__ void initializePopulation(curandStatePhilox4_32_10_t* state, char* s, int N, int q) {
	/*---------------------------------------------------------------------------------------------
		Initializes population on gpu(!) by randomly assigning each spin a value from suggestSpin function
	----------------------------------------------------------------------------------------------*/
	int r = threadIdx.x + blockIdx.x * blockDim.x;
	for (int k = 0; k < N; k++) {
		int arrayIndex = r * N + k;
		char spin = suggestSpin(state, r);
		s[arrayIndex] = spin;
	}
}
void initializePopulationMPI(char* s, int N, int q) {
	initializePopulation <<< BLOCKS, THREADS >>> (devStates, s, N, q);
}
void cudaMPImemset(void* ptr, int val, size_t size){
	gpuErrchk( cudaMemset(ptr, val, size) );
}

void cudaMPImemcpyD2H(void* dst, const void* src, size_t count)
{
	gpuErrchk( cudaMemcpy(dst,src,count,cudaMemcpyDeviceToHost) );
}
void cudaMPImemcpyH2D(void* dst, const void* src, size_t count)
{
	gpuErrchk( cudaMemcpy(dst,src,count,cudaMemcpyHostToDevice) );
}


__global__ void updateReplicas(char* s, float* E, int* update, int N) {
	/*---------------------------------------------------------------------------------------------
		Updates the population after the resampling step (done on cpu) by replacing indicated
		replicas by the proper other replica
	-----------------------------------------------------------------------------------------------*/
	int r = threadIdx.x + blockIdx.x * blockDim.x;
	int replica_shift = r * N;
	int source_r = update[r];
	int source_replica_shift = source_r * N;
	if (source_r != r) {
		for (int j = 0; j < N; j++) {
			s[j + replica_shift] = s[j + source_replica_shift];
		}
		E[r] = E[update[r]];
	}
}
void updateReplicasMPI(char* deviceSpin, float* deviceE, int* deviceUpdate, int N) {
	updateReplicas <<< BLOCKS, THREADS >>> (deviceSpin, deviceE, deviceUpdate, N);
}

__global__ void setup_kernel(curandStatePhilox4_32_10_t* state, int seed)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	/* Each thread gets same seed, a different sequence
	   number, no offset */
	curand_init(seed, id, 0, state + id);
}

void setup_kernelMPI(int seed){
	setup_kernel <<< BLOCKS, THREADS >>> (devStates, seed);
}

