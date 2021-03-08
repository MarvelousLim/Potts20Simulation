#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <curand_kernel.h>

#define NNEIBORS 2 // number of nearest neighbors, is 4 for 2d lattice

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

struct neibors_indexes {
	int right;
	int left;
};

__device__ struct neibors_indexes SLF(int j, int L, int N) {
	struct neibors_indexes result;
	result.right = (j + 1) % N;
	result.left = (j - 1 + N) % N; // L member is for positivity
	return result;
}

struct neibors {
	char left;
	char right;
};

__device__ struct neibors get_neibors_values(char* s, struct neibors_indexes n_i, int replica_shift) {
	struct neibors result = { s[n_i.left + replica_shift], s[n_i.right + replica_shift] };
	return result;
}

__device__ int LocalE(char currentSpin, struct neibors n) { 	// Computes energy of spin i with neighbors a, b, c, d 
	return -(currentSpin == n.left) - (currentSpin == n.right);
}

__device__ int DeltaE(char currentSpin, char suggestedSpin, struct neibors n) { // Delta of local energy while i -> e switch
	return LocalE(suggestedSpin, n) - LocalE(currentSpin, n);
}

__global__ void deviceEnergy(char* s, int* E, int L, int N) {
	int r = threadIdx.x + blockIdx.x * blockDim.x;
	int sum = 0;
	for (int j = 0; j < N; j++) {
		// 0.5 by doubling the summarize
		int replica_shift = r * N;
		char i = s[j + replica_shift]; // current spin value
		struct neibors_indexes n_i = SLF(j, L, N);
		struct neibors n = get_neibors_values(s, n_i, replica_shift); // we look into r replica and j spin
		sum += LocalE(i, n);
	}
	E[r] = sum / 2;
}

void CalcPrintAvgE(FILE* efile, int* E, int R, int U) {
	double avg = 0.0;
	for (int i = 0; i < R; i++) {
		avg += E[i];
	}
	avg /= R;
	fprintf(efile, "%d %f\n", U, avg);
	printf("E: %f\n", avg);
}

__device__ int getBFSSize(char* s, int start, int replica_shift, int N, int L, bool* deviceVisited, int* deviceStack, bool colorBlindMode) {
	// colorBlindMode disables check of spin value
	int currentClusterSize = 0;
	int stack_index = 0;

	char spinValue = s[replica_shift + start];
	deviceStack[replica_shift + stack_index++] = start;
	deviceVisited[replica_shift + start] = 1;

	while (stack_index > 0) {
		int currentIndex = deviceStack[replica_shift + --stack_index];
		currentClusterSize++;
		//printf("stack is %d, currentIndex %d, currentClusterSize %d\n", stack_index, currentIndex, currentClusterSize);
		struct neibors_indexes n = SLF(currentIndex, L, N);
		int possibleIndexes[NNEIBORS] = { n.left, n.right };
		for (int indexIndex = 0; indexIndex < NNEIBORS; indexIndex++) {
			int suggestedIndex = possibleIndexes[indexIndex];
			if (!deviceVisited[replica_shift + suggestedIndex] && (colorBlindMode || s[replica_shift + suggestedIndex] == spinValue)) {
				deviceStack[replica_shift + stack_index++] = suggestedIndex;
				deviceVisited[replica_shift + suggestedIndex] = 1;
			}
		}
	}
	return currentClusterSize;
}

__global__ void cudaReplicaBFS(char* s, int* E, int N, int L, int U, bool* deviceVisited, int* deviceClusterSizeArray, int* deviceStack) {
	int r = threadIdx.x + blockIdx.x * blockDim.x;
	int replica_shift = r * N;
	if (E[r] == U + 1) {
		int argmax = 0;
		int max = 0;
		// Go sequentially through lattice and assign spins to clusters for first time
		// to find main cluster
		for (int i = 0; i < N; i++) {
			if (!deviceVisited[replica_shift + i]) {
				int currentClusterSize = getBFSSize(s, i, replica_shift, N, L, deviceVisited, deviceStack, false);
				if (currentClusterSize > max) {
					max = currentClusterSize;
					argmax = i;
				}
			}
		}
		// now we know the main cluster; we want to return Visited to zero and then go again
		for (int i = 0; i < N; i++)
			deviceVisited[replica_shift + i] = 0;
		// but without main cluster and spin-value check
		int currentClusterSize = getBFSSize(s, argmax, replica_shift, N, L, deviceVisited, deviceStack, false);
		for (int i = 0; i < N; i++) {
			if (!deviceVisited[replica_shift + i]) {
				// watch for colorBlindMode here
				int currentClusterSize = getBFSSize(s, i, replica_shift, N, L, deviceVisited, deviceStack, true);
				atomicAdd(deviceClusterSizeArray + currentClusterSize - 1, 1);
			}
		}

	}
}

void makeClusterHistogram(char* s, int* E, int N, int L, int BLOCKS, int THREADS, int U, FILE* chfile, bool* deviceVisited, int* deviceClusterSizeArray, int* deviceStack, int* hostClusterSizeArray) {
	/*------------------------------------------------------------------------------------------------
		Disordered Cluster Histogram Algorithm
		Steps to procedure:
			- Identify replicas with energy one less than the energy ceiling
			x Determine the Majority (ordered) cluster
			x Identify clusters of spins not in the majority phase. For low
				enough energies, these should be isolated clusters.
			+ I decided to calculate all cluster, because why not. Can always kill em on postcalc
			- compile histogram of cluster sizes
	-------------------------------------------------------------------------------------------------*/
	cudaMemset(deviceVisited, 0, N * BLOCKS * THREADS * sizeof(bool));
	cudaMemset(deviceClusterSizeArray, 0, N * sizeof(unsigned int));

	cudaReplicaBFS << <BLOCKS, THREADS >> > (s, E, N, L, U, deviceVisited, deviceClusterSizeArray, deviceStack);
	cudaMemcpy(hostClusterSizeArray, deviceClusterSizeArray, N * sizeof(int), cudaMemcpyDeviceToHost);

	fprintf(chfile, "%d ", U);
	// write results to output files
	for (int i = 0; i < N; i++) {
		int size = i + 1;
		int freq = hostClusterSizeArray[i];
		if (freq > 0)
			fprintf(chfile, "%d, %d; ", size, freq);
	}
	fprintf(chfile, "\n");
}

void CalculateRhoT(const int* replicaFamily, FILE* ptfile, int R, int U) {
	// histogram of family sizes
	int* famHist = (int*)calloc(R, sizeof(int));
	for (int i = 0; i < R; i++) {
		famHist[replicaFamily[i]]++;
	}
	double sum = 0;
	for (int i = 0; i < R; i++) {
		sum += famHist[i] * famHist[i];
	}
	sum /= R;
	fprintf(ptfile, "%d %f\n", U, sum);
	sum /= R;
	printf("RhoT:\t%f\n", sum);
	free(famHist);
}

__global__ void initializePopulation(curandStatePhilox4_32_10_t* state, char* s, int N, int q) {
	/*---------------------------------------------------------------------------------------------
		Initializes population on gpu(!) by randomly assigning each spin a value from 0 to q-1
	----------------------------------------------------------------------------------------------*/
	int r = threadIdx.x + blockIdx.x * blockDim.x;
	for (int k = 0; k < N; k++) {
		int arrayIndex = r * N + k;
		char spin = curand(&state[r]) % q;
		s[arrayIndex] = spin;
	}
}

__global__ void equilibrate(curandStatePhilox4_32_10_t* state, char* s, int* E, int L, int N, int R, int q, int nSteps, int U) {
	/*---------------------------------------------------------------------------------------------
		Main Microcanonical Monte Carlo loop.  Performs update sweeps on each replica in the
		population;
	---------------------------------------------------------------------------------------------*/

	int r = threadIdx.x + blockIdx.x * blockDim.x;
	int replica_shift = r * N;
	for (int k = 0; k < N * nSteps; k++) {
		int j = curand(&state[blockIdx.x]) % N;
		char currentSpin = s[j + replica_shift];
		char suggestedSpin = curand(&state[r]) % q;
		struct neibors_indexes n_i = SLF(j, L, N);
		struct neibors n = get_neibors_values(s, n_i, replica_shift);
		int dE = DeltaE(currentSpin, suggestedSpin, n);
		if (E[r] + dE < U) {
			E[r] = E[r] + dE;
			s[j + replica_shift] = suggestedSpin;
		}
	}
}

void Swap(int* A, int i, int j) {
	int temp = A[i];
	A[i] = A[j];
	A[j] = temp;
}

void quicksort(int* E, int* O, int left, int right, int cool) {
	int Min = (left + right) / 2;
	int i = left;
	int j = right;
	double pivot = cool * E[O[Min]];

	while (left < j || i < right)
	{
		while (cool * E[O[i]] > pivot)
			i++;
		while (cool * E[O[j]] < pivot)
			j--;

		if (i <= j) {
			Swap(O, i, j);
			i++;
			j--;
		}
		else {
			if (left < j)
				quicksort(E, O, left, j, cool);
			if (i < right)
				quicksort(E, O, i, right, cool);
			return;
		}
	}
}

void resample(int* E, int* O, int* update, int* replicaFamily, int R, int U, FILE* e2file, FILE* Xfile) {
	//std::sort(O, O + R, [&E](int a, int b) {return E[a] > E[b]; }); // greater sign for descending order
	quicksort(E, O, 0, R - 1, 1); //Sorts O by energy

	int nCull = 0;
	fprintf(e2file, "%d %d\n", U, E[O[0]]);
	while (E[O[nCull]] == U - 1) {
		nCull++;
		if (nCull == R) {
			break;
		}
	}
	// culling fraction
	double X = nCull;
	X /= R;
	fprintf(Xfile, "%d %f\n", U - 1, X);
	printf("Culling fraction:\t%f\n", X);
	for (int i = 0; i < R; i++)
		update[i] = i;
	if (nCull < R) {
		for (int i = 0; i < nCull; i++) {
			// random selection of unculled replica
			int r = (rand() % (R - nCull)) + nCull; // different random number generator for
			update[O[i]] = O[r];
			replicaFamily[O[i]] = replicaFamily[O[r]];
		}
	}
}

__global__ void updateReplicas(char* s, int* E, int* update, int N) {
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

__global__ void setup_kernel(curandStatePhilox4_32_10_t* state, int seed)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	/* Each thread gets same seed, a different sequence
	   number, no offset */
	curand_init(seed, id, 0, state + id);
}

int main(int argc, char* argv[]) {
	// Parameters:
	int q = 2;	// q parameter for potts model, each spin variable can take on values 0 - q-1

	int run_number = atoi(argv[1]);	// A number to label this run of the algorithm, used for data keeping purposes, also, a seed
	int seed = run_number;
	//int grid_width = atoi(argv[2]);	// should not be more than 256 due to MTGP32 limits
	int L = atoi(argv[2]);	// Lattice size
	int N = L * L;
	//int R = grid_width * BLOCKS;	// Population size
	int BLOCKS = atoi(argv[3]);
	int THREADS = atoi(argv[4]);
	int nSteps = atoi(argv[5]);
	int R = BLOCKS * THREADS;

	// initializing files to write in
	char s[100];
	sprintf(s, "datasets//1DIsing_N%d_R%d_nSteps%d_run%de.txt", N, R, nSteps, run_number);
	FILE* efile = fopen(s, "w");	// average energy
	sprintf(s, "datasets//1DIsing_N%d_R%d_nSteps%d_run%de2.txt", N, R, nSteps, run_number);
	FILE* e2file = fopen(s, "w");	// surface (culled) energy
	sprintf(s, "datasets//1DIsing_N%d_R%d_nSteps%d_run%dX.txt", N, R, nSteps, run_number);
	FILE* Xfile = fopen(s, "w");	// culling fraction
	sprintf(s, "datasets//1DIsing_N%d_R%d_nSteps%d_run%dpt.txt", N, R, nSteps, run_number);
	FILE* ptfile = fopen(s, "w");	// rho t
	sprintf(s, "datasets//1DIsing_N%d_R%d_nSteps%d_run%dn.txt", N, R, nSteps, run_number);
	FILE* nfile = fopen(s, "w");	// number of sweeps
	sprintf(s, "datasets//1DIsing_N%d_R%d_nSteps%d_run%dch.txt", N, R, nSteps, run_number);
	FILE* chfile = fopen(s, "w");	// cluster size histogram


	size_t fullLatticeByteSize = R * N * sizeof(char);

	// Allocate space on host 
	int* hostE = (int*)malloc(R * sizeof(int));
	int* hostUpdate = (int*)malloc(R * sizeof(int));
	int* replicaFamily = (int*)malloc(R * sizeof(int));
	int* energyOrder = (int*)malloc(R * sizeof(int));
	for (int i = 0; i < R; i++) {
		energyOrder[i] = i;
		replicaFamily[i] = i;
	}

	// Allocate memory on device
	char* deviceSpin; // s, d_s
	int* deviceE;
	int* deviceUpdate;
	cudaMalloc((void**)&deviceSpin, fullLatticeByteSize);
	cudaMalloc((void**)&deviceE, R * sizeof(int));
	cudaMalloc((void**)&deviceUpdate, R * sizeof(int));

	// Allocate memory for histogram calculation
	int* hostClusterSizeArray = (int*)malloc(N * sizeof(int));
	bool* deviceVisited;
	int* deviceClusterSizeArray;
	int* deviceStack;
	cudaMalloc((void**)&deviceVisited, N * R * sizeof(bool));
	cudaMalloc((void**)&deviceClusterSizeArray, N * sizeof(int));
	cudaMalloc((void**)&deviceStack, N * R * sizeof(int));

	// Init Philox
	curandStatePhilox4_32_10_t* devStates;
	cudaMalloc((void**)&devStates, R * sizeof(curandState));
	setup_kernel <<< BLOCKS, THREADS >>> (devStates, seed);

	// Init std random generator for little host part
	srand(seed);

	// Actually working part
	initializePopulation <<<BLOCKS, THREADS >>> (devStates, deviceSpin, N, q);
	//cudaMemset(deviceE, 0, R * sizeof(int));
	deviceEnergy <<<BLOCKS, THREADS >>> (deviceSpin, deviceE, L, N);

	int U = 0;	// U is energy ceiling

	while (U >= -N) {
		fprintf(nfile, "%d %d\n", U, nSteps);
		printf("U:\t%d out of %d; nSteps: %d;\n", U, -N, nSteps);
		// Perform monte carlo sweeps on gpu
		equilibrate <<< BLOCKS, THREADS >>> (devStates, deviceSpin, deviceE, L, N, R, q, nSteps, U);

		// Create disordered cluster size histogram in particular energy range
		//if (U <= -1.5 * N)
		//	makeClusterHistogram(deviceSpin, deviceE, N, L, BLOCKS, THREADS, U, chfile, deviceVisited, deviceClusterSizeArray, deviceStack, hostClusterSizeArray);

		cudaMemcpy(hostE, deviceE, R * sizeof(int), cudaMemcpyDeviceToHost);
		// record average energy and rho t
		CalcPrintAvgE(efile, hostE, R, U);
		CalculateRhoT(replicaFamily, ptfile, R, U);
		// perform resampling step on cpu
		resample(hostE, energyOrder, hostUpdate, replicaFamily, R, U, e2file, Xfile);
		U--;
		// copy list of replicas to update back to gpu
		cudaMemcpy(deviceUpdate, hostUpdate, R * sizeof(int), cudaMemcpyHostToDevice);
		updateReplicas <<< BLOCKS, THREADS >>> (deviceSpin, deviceE, deviceUpdate, N);
	}

	// Free memory and close files
	cudaFree(devStates);
	cudaFree(deviceSpin);
	cudaFree(deviceE);
	cudaFree(deviceUpdate);
	cudaFree(deviceClusterSizeArray);
	cudaFree(deviceStack);
	cudaFree(deviceVisited);

	free(hostE);
	free(hostUpdate);
	free(replicaFamily);
	free(energyOrder);
	free(hostClusterSizeArray);

	fclose(efile);
	fclose(e2file);
	fclose(Xfile);
	fclose(ptfile);
	fclose(nfile);
	fclose(chfile);

	// End
	return 0;
}
