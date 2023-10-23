#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <curand_kernel.h>

#include <time.h>

#include "minepar.h"

typedef curandStatePhilox4_32_10_t RNGState;

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

RNGState* devStates;
int BLOCKS;
int THREADS;
dim3 DimGridRes;

void cudaMPIset(int device)
{
	gpuErrchk(cudaSetDevice(device));
}
void BLTH(int BL, int TH){
	BLOCKS=BL;THREADS=TH;
}
void defND(int Ri, int TH){
	BLOCKS = (int)ceil(Ri/(double)TH);
	THREADS = TH;
	DimGridRes = dim3(Ri,N/EQthreads,1);
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

__host__ __device__ struct neibors_indexes SLF(int j) {
	struct neibors_indexes result;
	result.up = (j - L + N) % N; // N member is for positivity
	result.right = (j + 1) % L + L * (j / L);
	result.down = (j + L) % N;
	result.left = (j - 1 + L) % L + L * (j / L); // L member is for positivity
	return result;
}

__device__ struct neibors get_neibors_values(Replica* Rep, struct neibors_indexes n_i, int r) {
	return {
		//s[n_i.up + replica_shift],
		Rep[r].sp[n_i.up],
		//s[n_i.right + replica_shift],
		Rep[r].sp[n_i.right],
		//s[n_i.down + replica_shift],
		Rep[r].sp[n_i.down],
		//s[n_i.left + replica_shift]
		Rep[r].sp[n_i.left]
	};
}

__host__ __device__ struct energy_parts localEnergyParts(signed char currentSpin, struct neibors n) {
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

__device__ struct energy_parts calcEnergyParts(Replica* Rep, float D, int r) {
	struct energy_parts sum = { 0, 0 };
	for (int j = 0; j < N; j++) {
		// do not forget double joint summarization!
		//int replica_shift = r * N;
		//signed char i = s[j + replica_shift]; // current spin value
		signed char i = Rep[r].sp[j]; // current spin value
		struct neibors_indexes n_i = SLF(j);
		struct neibors n = get_neibors_values(Rep, n_i, r); // we look into r replica and j spin
		struct energy_parts tmp = localEnergyParts(i, n);
		sum = addEnergyParts(sum, tmp);
	}
	return sum;
}

__device__ float calcEnergyFromParts(struct energy_parts energyParts, float D) {
	return (energyParts.Ising / 2) + (D * energyParts.Blume); // div 2 because of double joint summarization
}

__global__ void deviceEnergy(Replica* Rep, EnOr* E, float D) {
	int r = threadIdx.x + blockIdx.x * blockDim.x;
	struct energy_parts sum = calcEnergyParts(Rep, D, r);
	E[r].Energy = calcEnergyFromParts(sum, D); 
}

// hardcoded spin suggestion for init
__device__ signed char suggestSpin(RNGState* state) {
	return curand(state) % 3 - 1;
}

// hardcoded spin suggestion for equilibration
__device__ signed char suggestSpinSwap(RNGState* state, char currentSpin) {
	return (currentSpin + 2 + (curand(state) % 2)) % 3 - 1; // little trick
}

template <class sometype> __inline__ __device__ sometype warpReduceSum(sometype val)
//__device__ float warpReduceSum(float val)
{
	for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
		val += __shfl_down_sync(FULL_MASK, val, offset);
	return val;
}

template <class sometype> __inline__ __device__ sometype blockReduceSum(sometype val)	 // use when blockDim.x is divisible by 32
{
	static __shared__ sometype shared[32];			// one needs to additionally synchronize threads after execution
	int lane = threadIdx.x % warpSize;			// in the case of multiple use of blockReduceSum in a single kernel
	int wid = threadIdx.x / warpSize;
	val = warpReduceSum(val);
	if (lane==0) shared[wid]=val;
	__syncthreads();
	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
	if (wid==0) val = warpReduceSum(val);
	return val;
}

__global__ void equilibrate(unsigned long long seed, unsigned long long initial_sequence, Replica* Rep, EnOr* E, int R, int q, int nSteps, float U, float D, int heat){//, int* acceptance_number) {
	/*---------------------------------------------------------------------------------------------
		Main Microcanonical Monte Carlo loop.  Performs update sweeps on each replica in the
		population;
		There, one could change calcEnergyParts for system of carrying arrays of energy parts,
		but:
			1. This is not the bottleneck (which is for loop over N * nSteps
	---------------------------------------------------------------------------------------------*/

	int r = threadIdx.x + blockIdx.x * blockDim.x;
	//int replica_shift = r * N;
	RNGState localrng; curand_init(seed,initial_sequence+r,0,&localrng);
	if(r<R){
		energy_parts baseEnergyParts = calcEnergyParts(Rep, D, r);

		for (int k = 0; k < N * nSteps; k++)
		{
			int j = curand(&localrng) % N;
			signed char currentSpin = Rep[r].sp[j];
			signed char suggestedSpin = suggestSpinSwap(&localrng, currentSpin);
			//char suggestedSpin = curand(&state[r]) % 3 - 1;
			neibors_indexes n_i = SLF(j);
			neibors n = get_neibors_values(Rep, n_i, r);
			energy_parts suggestedLocalEnergyParts = localEnergyParts(suggestedSpin, n);

			energy_parts currentLocalEnergyParts = localEnergyParts(currentSpin, n);
			energy_parts deltaLocalEnergyParts = subEnergyParts(suggestedLocalEnergyParts, currentLocalEnergyParts);
			//local energy delta calculated for single spin; but should be for whole lattice
			//thus we need to count Ising part twice - for the neibors change in energy as well
			//but not Blume part!
			deltaLocalEnergyParts.Ising *= 2;
			energy_parts suggestedEnergyParts = addEnergyParts(baseEnergyParts, deltaLocalEnergyParts);
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
				E[r].Energy = suggestedEnergy;
				//s[j + replica_shift] = suggestedSpin;
				Rep[r].sp[j] = suggestedSpin;
			/*
			float reductionRes = warpReduceSum(1);
			if ((threadIdx.x & (warpSize - 1)) == 0)
				atomicAdd(acceptance_number, reductionRes);
			*/

			}
		}
	}
}
void equilibrateMPI(unsigned long long seed, unsigned long long initial_sequence, Replica* Rep, EnOr* deviceE, int R, int q, int nSteps, float U, float D, int heat) {
	equilibrate <<< BLOCKS, THREADS >>> (seed, initial_sequence, Rep, deviceE, R, q, nSteps, U, D, heat);
}



__global__ void initializePopulation(unsigned long long seed, unsigned long long initial_sequence, Replica* Rep, int q, int R) {
	/*---------------------------------------------------------------------------------------------
		Initializes population on gpu(!) by randomly assigning each spin a value from suggestSpin function
	----------------------------------------------------------------------------------------------*/
	int r = threadIdx.x + blockIdx.x * blockDim.x;
	RNGState localrng; curand_init(seed,initial_sequence+r,0,&localrng);
	if(r<R){
		for (int k = 0; k < N; k++) {
			//int arrayIndex = r * N + k;
			signed char spin = suggestSpin(&localrng);
			//s[arrayIndex] = spin;
			Rep[r].sp[k] = spin;
		}
	}
}
void initializePopulationMPI(unsigned long long seed, unsigned long long initial_sequence, Replica* Rep, int q, int R) {
	initializePopulation <<< BLOCKS, THREADS >>> (seed, initial_sequence, Rep, q, R);
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


__global__ void updateReplicas(Replica* Rep, float* E, int* update) {
	/*---------------------------------------------------------------------------------------------
		Updates the population after the resampling step (done on cpu) by replacing indicated
		replicas by the proper other replica
	-----------------------------------------------------------------------------------------------*/
	int r = threadIdx.x + blockIdx.x * blockDim.x;
	//int replica_shift = r * N;
	int source_r = update[r];
	//int source_replica_shift = source_r * N;
	if (source_r != r) {
		for (int j = 0; j < N; j++) {
			//s[j + replica_shift] = s[j + source_replica_shift];
			Rep[r].sp[j] = Rep[source_r].sp[j];
		}
		E[r] = E[source_r];
	}
}
void updateReplicasMPI(Replica* Rep, float* deviceE, int* deviceUpdate) {
	updateReplicas <<< BLOCKS, THREADS >>> (Rep, deviceE, deviceUpdate);
}

__global__ void setup_kernel(curandStatePhilox4_32_10_t* state, unsigned long long seed)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	/* Each thread gets same seed, a different sequence
	   number, no offset */
	curand_init(seed, id, 0, state + id);
}

void setup_kernelMPI(unsigned long long seed){
	setup_kernel <<< BLOCKS, THREADS >>> (devStates, seed);
}

__global__ void copyreploff(Replica* Rep, EnOr* reploff, int R) {
	int r = threadIdx.x + blockIdx.x * blockDim.x;
	if (r<R){	
		Rep[r].Roff=reploff[r].Roff;
	}
}
void copyreploffMPI(Replica* Rep, EnOr* reploff, int R) {
	copyreploff <<< BLOCKS, THREADS >>> (Rep, reploff, R);
}

__global__ void CalcParSum(Replica* Repd, int R) // calculation of {sum_{j=0}^i Roff} for each replica
{
	unsigned int parS; __shared__ unsigned int val;
	int j, t = threadIdx.x, b = blockIdx.x;
	int idx = t + blockDim.x * b; unsigned int MyParSum = 0;
	for (j = 0; j<b; j+=blockDim.x){
		parS = (t+j<b) ? Repd[(t+j)*blockDim.x].ValInt[1] : 0;
		parS = blockReduceSum<unsigned int>(parS); 
		if(t==0) val = parS; __syncthreads(); MyParSum += val; // we sum Roff for all blocks from 0 to (b-1).
	}
	if(idx<R){
		for(j=blockDim.x*b;j<idx;j++) MyParSum+=Repd[j].Roff;	// we add Roff for current block threads from 0 to (t-1)
		Repd[idx].ValInt[0] = MyParSum;
	}
}
void cudaCalcParSum(Replica* Repptr, int Rc)
{
	//CalcParSum<<< NblocksR, Nthreads >>> (Repptr, Rc);
	CalcParSum<<< BLOCKS, THREADS >>> (Repptr, Rc);
}
__global__ void resampleKer(Replica* Repd, Replica* Repdnew) // copying replicas (the main part of the resampling process)
{
	int j, jnext, B = blockIdx.x, idx = threadIdx.x + EQthreads * blockIdx.y;
	j = Repd[B].ValInt[0]; jnext = j + Repd[B].Roff;
	for(; j < jnext; j++){ 
		Repdnew[j].sp[idx]=Repd[B].sp[idx];
	}
}
void cudaresampleKer(Replica* Repptr, Replica* Repptrdest)
{
	resampleKer<<< DimGridRes, EQthreads >>> (Repptr,Repptrdest);
}
__global__ void blocksumKer(Replica* Rd, int Rlocal, unsigned int* Rnew) // calculation of sums of offsprings for blocks 
{
	unsigned int parS;
	int t = threadIdx.x; int b = blockIdx.x;	
	int idx = t + blockIdx.x * b;
	if (idx < Rlocal){		
		parS = Rd[idx].Roff;
	} else parS = 0;
	parS = blockReduceSum<unsigned int>(parS);
	if(t==0){ 
		Rd[idx].ValInt[1] = parS;  // sum of Roff for all threads in current block
		atomicAdd(Rnew,parS); // we save new population size
	}
}
void cudablocksumKer(Replica* Repptr, int Rlocalc, unsigned int* Rnewc)
{
	blocksumKer <<< BLOCKS, THREADS >>> (Repptr, Rlocalc, Rnewc);
}
