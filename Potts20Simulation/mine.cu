#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <curand_kernel.h>

#include <time.h>

#define NNEIBORS 4 // number of nearest neighbors, is 4 for 2d lattice
#define STATISTICS_NUMBER 7
#define EPSILON 0

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

struct neibors_indexes {
	int up;
	int right;
	int down;
	int left;
};

__host__ __device__ struct neibors_indexes SLF(int j, int L, int N) {
	struct neibors_indexes result;
	result.up = (j - L + N) % N; // N member is for positivity
	result.right = (j + 1) % L + L * (j / L);
	result.down = (j + L) % N;
	result.left = (j - 1 + L) % L + L * (j / L); // L member is for positivity
	return result;
}

struct neibors {
	char up;
	char right;
	char down;
	char left;
};

struct energy_parts {
	int Ising;
	int Blume;
};

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
		-(currentSpin * n.up)
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

__device__ struct energy_parts calcEnergyParts(char* s, int L, int N, int r) {
	struct energy_parts sum = { 0, 0 };
	int replica_shift = r * N;

	for (int j = 0; j < N; j++) {
		// do not forget double joint summarization!
		char i = s[j + replica_shift]; // current spin value
		struct neibors_indexes n_i = SLF(j, L, N);
		struct neibors n = get_neibors_values(s, n_i, replica_shift); // we look into r replica and j spin
		struct energy_parts tmp = localEnergyParts(i, n);
		sum = addEnergyParts(sum, tmp);
	}
	return sum;
}

__device__ int calcEnergyFromParts(struct energy_parts energyParts, int D_div, int D_base) { // D = D_div / D_base
	return (D_base * energyParts.Ising / 2) + (D_div * energyParts.Blume); // div 2 because of double joint summarization
}

__global__ void deviceEnergy(char* s, int* E, int L, int N, int D_div, int D_base) {
	int r = threadIdx.x + blockIdx.x * blockDim.x;
	struct energy_parts sum = calcEnergyParts(s, L, N, r);
	E[r] = calcEnergyFromParts(sum, D_div, D_base);
}

// hardcoded spin suggestion for init
__device__ char suggestSpin(curandStatePhilox4_32_10_t* state, int r) {
	return curand(&state[r]) % 3 - 1;
}

// hardcoded spin suggestion for equilibration
__device__ char suggestSpinSwap(curandStatePhilox4_32_10_t* state, int r, char currentSpin) {
	return (currentSpin + 2 + (curand(&state[r]) % 2)) % 3 - 1; // little trick
}

#define FULL_MASK 0xffffffff

__device__ float warpReduceSum(float val)
{
	for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
		val += __shfl_down_sync(FULL_MASK, val, offset);
	return val;
}

__global__ void equilibrate(curandStatePhilox4_32_10_t* state, char* s, int* E, int L, int N, int R, int q, int nSteps, int U, int D_div, int D_base, bool heat) {//, int* acceptance_number) {
	/*---------------------------------------------------------------------------------------------
		Main Microcanonical Monte Carlo loop.  Performs update sweeps on each replica in the
		population;
		There, one could change calcEnergyParts for system of carrying arrays of energy parts,
		but:
			1. This is not the bottleneck (which is for loop over N * nSteps
	---------------------------------------------------------------------------------------------*/

	int r = threadIdx.x + blockIdx.x * blockDim.x;
	int replica_shift = r * N;

	struct energy_parts baseEnergyParts = calcEnergyParts(s, L, N, r);

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
		int suggestedEnergy = calcEnergyFromParts(suggestedEnergyParts, D_div, D_base);

		if ((!heat && (suggestedEnergy + EPSILON < U)) || (heat && (suggestedEnergy - EPSILON > U))) {
			baseEnergyParts = suggestedEnergyParts;
			E[r] = suggestedEnergy;
			s[j + replica_shift] = suggestedSpin;
		}
	}
}

__global__ void calcMagnetization(char* s, int* E, int N, int R, int U, int* deviceMagnetization) {//, int* acceptance_number) {

	int r = threadIdx.x + blockIdx.x * blockDim.x;
	//printf("kernel %d reporting: E = %i, U = %i\n", r, E[r], U);
	int replica_shift = r * N;
	if (E[r] == U) {
		int m = 0;
		int m_sq = 0;


		for (int j = 0; j < N; j++) {
			char i = s[j + replica_shift]; // current spin value
			m += i;
			m_sq += i * i;
		}
		//printf("kernel %d reporting success: E = %i, U = %i; m=%i, m_sq=%i\n", r, E[r], U, m, m_sq);

		deviceMagnetization[r * STATISTICS_NUMBER + 0] = 1; // number of replicas with E = U
		deviceMagnetization[r * STATISTICS_NUMBER + 1] = abs(m); // | magnetization |
		deviceMagnetization[r * STATISTICS_NUMBER + 2] = m * m; // | magnetization^2 |
		deviceMagnetization[r * STATISTICS_NUMBER + 3] = m * m * m * m; // | magnetization^2 |
		deviceMagnetization[r * STATISTICS_NUMBER + 4] = m_sq; // | concentration |
		deviceMagnetization[r * STATISTICS_NUMBER + 5] = m_sq * m_sq; // | concentration^2 |
		deviceMagnetization[r * STATISTICS_NUMBER + 6] = m_sq * m_sq * m_sq * m_sq; // | concentration^2 |
	}
}


void CalcPrintAvgE(FILE* efile, int* E, int R, int U, int D_base) {
	float avg = 0.0;
	for (int i = 0; i < R; i++) {
		avg += E[i];
	}
	avg /= R;
	fprintf(efile, "%f %f\n", 1.0 * U / D_base, avg);
	printf("E: %f\n", avg);
}

void CalculateRhoT(const int* replicaFamily, FILE* ptfile, int R, int U, int D_base) {
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
	fprintf(ptfile, "%f %f\n", 1.0 * U / D_base, sum);
	sum /= R;
	printf("RhoT:\t%f\n", sum);
	free(famHist);
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

void Swap(int* A, int i, int j) {
	int temp = A[i];
	A[i] = A[j];
	A[j] = temp;
}

void quicksort(int* E, int* O, int left, int right, int direction) {
	int Min = (left + right) / 2;
	int i = left;
	int j = right;
	double pivot = direction * E[O[Min]];

	while (left < j || i < right)
	{
		while (direction * E[O[i]] > pivot)
			i++;
		while (direction * E[O[j]] < pivot)
			j--;

		if (i <= j) {
			Swap(O, i, j);
			i++;
			j--;
		}
		else {
			if (left < j)
				quicksort(E, O, left, j, direction);
			if (i < right)
				quicksort(E, O, i, right, direction);
			return;
		}
	}
}

int resample(int* E, int* O, int* update, int* replicaFamily, int R, int* U, int D_base, FILE* e2file, FILE* Xfile, bool heat) {
	//std::sort(O, O + R, [&E](int a, int b) {return E[a] > E[b]; }); // greater sign for descending order
	quicksort(E, O, 0, R - 1, 1 - 2 * heat); //Sorts O by energy

	int nCull = 0;
	fprintf(e2file, "%f %i\n", 1.0 * (*U) / D_base, E[O[0]]);

	//update energy seiling to the highest available energy
	int U_old = *U;
	int U_new;

	for (int i = 0; i < R; i++) {
		U_new = E[O[i]];
		if ((!heat && U_new < U_old - EPSILON) || (heat && U_new > U_old + EPSILON)) {
			*U = U_new;
			break;
		}
	}

	if (fabs(*U - U_old) <= EPSILON) {
		return 1; // out of replicas
	}

	while ((!heat && E[O[nCull]] >= *U - EPSILON) || (heat && E[O[nCull]] <= *U + EPSILON)) {
		nCull++;
		if (nCull == R) {
			break;
		}
	}
	// culling fraction
	double X = nCull;
	X /= R;
	//printf("U: %d %d %d\n", U_new, U_old, *U);
	fprintf(Xfile, "%f %f\n", 1.0 * (*U) / D_base, X);
	fflush(Xfile);
	printf("Culling fraction:\t%f\n", X);
	fflush(stdout);
	for (int i = 0; i < R; i++)
		update[i] = i;
	if (nCull < R) {
		for (int i = 0; i < nCull; i++) {
			// random selection of unculled replica
			int r = (rand() % (R - nCull)) + nCull; // different random number generator for resampling
			update[O[i]] = O[r];
			replicaFamily[O[i]] = replicaFamily[O[r]];
		}
	}

	return 0;
}

void PrintMagnetization(int* U, int D_base, FILE* mfile, int R, int* hostMagnetization) {
	long long int n[STATISTICS_NUMBER];
	for (int j = 0; j < STATISTICS_NUMBER; j++) {
		n[j] = 0;
	}

	for (int i = 0; i < R; i++) {
		for (int j = 0; j < STATISTICS_NUMBER; j++) {
			n[j] += hostMagnetization[i * STATISTICS_NUMBER + j];
		}
	}
	fprintf(mfile, "%f", 1.0 * (*U) / D_base);
	for (int j = 0; j < STATISTICS_NUMBER; j++) {
		fprintf(mfile, "%lld ", n[j]);
	}
	fprintf(mfile, "\n ");
	fflush(mfile);
	fflush(mfile);
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

	clock_t start, end;
	double cpu_time_used;

	start = clock();

	// Parameters:

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

	// q parameter for potts model, each spin variable can take on values 0 - q-1
	// strictly hardcoded
	int q = 3;

	//Blume-Capel model parameter
	int D_div = atoi(argv[6]);
	int D_base = atoi(argv[7]);
	float D = (float)D_div / D_base;
	bool heat = atoi(argv[8]); // 0 if cooling (default) and 1 if heating


	// initializing files to write in
	const char* heating = heat ? "Heating" : "";

	printf("running 2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%de.txt\n", heating, q, D, N, R, nSteps, run_number);

	char s[100];
	const char* prefix = "2DBlume";
	sprintf(s, "datasets//%s//2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%de.txt", prefix, heating, q, D, N, R, nSteps, run_number);
	FILE* efile = fopen(s, "w");	// average energy
	sprintf(s, "datasets//%s//2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%de2.txt", prefix, heating, q, D, N, R, nSteps, run_number);
	FILE* e2file = fopen(s, "w");	// surface (culled) energy
	sprintf(s, "datasets//%s//2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%dX.txt", prefix, heating, q, D, N, R, nSteps, run_number);
	FILE* Xfile = fopen(s, "w");	// culling fraction
	sprintf(s, "datasets//%s//2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%dpt.txt", prefix, heating, q, D, N, R, nSteps, run_number);
	FILE* ptfile = fopen(s, "w");	// rho t
	sprintf(s, "datasets//%s//2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%dn.txt", prefix, heating, q, D, N, R, nSteps, run_number);
	FILE* nfile = fopen(s, "w");	// number of sweeps
	sprintf(s, "datasets//%s//2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%dch.txt", prefix, heating, q, D, N, R, nSteps, run_number);
	FILE* chfile = fopen(s, "w");	// cluster size histogram
	sprintf(s, "datasets//%s//2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%dm.txt", prefix, heating, q, D, N, R, nSteps, run_number);
	FILE* mfile = fopen(s, "w");	// magnetization in format m = sum(sigma), m^2=sum(sigma^2), N
	/*
	sprintf(s, "datasets//2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%de3.txt", heating, q, D, N, R, nSteps, run_number);
	FILE* e3file = fopen(s, "w");
	*/
	/*
	sprintf(s, "datasets//hysteresis//2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%ds.txt", heating, q, D, N, R, nSteps, run_number);
	FILE* sfile = fopen(s, "w");	// spin system sample
	*/

	size_t fullLatticeByteSize = R * N * sizeof(char);

	// Allocate space on host
	//char* hostSpin = (char*)malloc(fullLatticeByteSize);
	int* hostE = (int*)malloc(R * sizeof(int));
	int* hostUpdate = (int*)malloc(R * sizeof(int));
	int* replicaFamily = (int*)malloc(R * sizeof(int));
	int* energyOrder = (int*)malloc(R * sizeof(int));
	for (int i = 0; i < R; i++) {
		energyOrder[i] = i;
		replicaFamily[i] = i;
	}
	int MagnetizationArraySize = R * STATISTICS_NUMBER;
	int* hostMagnetization = (int*)malloc(MagnetizationArraySize * sizeof(int));

	// Allocate memory on device
	char* deviceSpin; // s, d_s
	int* deviceE;
	int* deviceUpdate;
	int* deviceMagnetization;
	gpuErrchk(cudaMalloc((void**)&deviceSpin, fullLatticeByteSize));
	gpuErrchk(cudaMalloc((void**)&deviceE, R * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&deviceUpdate, R * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&deviceMagnetization, MagnetizationArraySize * sizeof(int)));

	// Allocate memory for histogram calculation
	/*
	int* hostClusterSizeArray = (int*)malloc(N * sizeof(int));
	bool* deviceVisited;
	int* deviceClusterSizeArray;
	int* deviceStack;
	gpuErrchk( cudaMalloc((void**)&deviceVisited, N * R * sizeof(bool)) );
	gpuErrchk( cudaMalloc((void**)&deviceClusterSizeArray, N * sizeof(int)) );
	gpuErrchk( cudaMalloc((void**)&deviceStack, N * R * sizeof(int)) );
	*/

	// Init Philox
	curandStatePhilox4_32_10_t* devStates;
	gpuErrchk(cudaMalloc((void**)&devStates, R * sizeof(curandStatePhilox4_32_10_t)));
	setup_kernel << < BLOCKS, THREADS >> > (devStates, seed);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	// Init std random generator for little host part
	srand(seed);

	// Actually working part
	initializePopulation << < BLOCKS, THREADS >> > (devStates, deviceSpin, N, q);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	cudaMemset(deviceE, 0, R * sizeof(int));

	//init testing values
	/*
	deviceEnergy <<< BLOCKS, THREADS >>> (deviceSpin, deviceE, L, N, D);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk( cudaMemcpy(hostE, deviceE, R * sizeof(int), cudaMemcpyDeviceToHost) );

	char* hostSpin = (char*)malloc(N * sizeof(char)); // test shit
	gpuErrchk(cudaMemcpy(hostSpin, deviceSpin, N * sizeof(char), cudaMemcpyDeviceToHost)); // take one replica (first)
	for (int i = 0; i < L; i++) {
		for (int j = 0; j < L; j++) {
			printf("%i ", hostSpin[i * L + j]);
		}
		printf("\n");
	}


	int host_acceptance_number = 0;
	int* device_acceptance_number;
	gpuErrchk(cudaMalloc((void**)&device_acceptance_number, sizeof(int)));
	*/


	int upper_energy = N * abs(D_div) + 2 * N * D_base;
	int lower_energy = -N * abs(D_div) - 2 * N * D_base;

	int U = (heat ? lower_energy : upper_energy);	// U is energy ceiling

	//CalcPrintAvgE(efile, hostE, R, U);

	while ((U >= lower_energy && !heat) || (U <= upper_energy && heat)) {

		fprintf(nfile, "%d %d\n", U, nSteps);
		printf("U:\t%f out of %d; nSteps: %d;\n", 1.0 * U / D_base, -2 * N, nSteps);

		equilibrate << < BLOCKS, THREADS >> > (devStates, deviceSpin, deviceE, L, N, R, q, nSteps, U, D_div, D_base, heat);// , device_acceptance_number);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk(cudaMemcpy(hostE, deviceE, R * sizeof(int), cudaMemcpyDeviceToHost));

		// record average energy and rho t
		CalcPrintAvgE(efile, hostE, R, U, D_base);
		CalculateRhoT(replicaFamily, ptfile, R, U, D_base);
		// perform resampling step on cpu
		// also lowers energy seiling U

		int error = resample(hostE, energyOrder, hostUpdate, replicaFamily, R, &U, D_base, e2file, Xfile, heat);


		cudaMemset(deviceMagnetization, 0, MagnetizationArraySize * sizeof(int));
		calcMagnetization << < BLOCKS, THREADS >> > (deviceSpin, deviceE, N, R, U, deviceMagnetization);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk(cudaMemcpy(hostMagnetization, deviceMagnetization, MagnetizationArraySize * sizeof(int), cudaMemcpyDeviceToHost));
		PrintMagnetization(&U, D_base, mfile, R, hostMagnetization);


		if (error)
		{
			printf("Process ended with zero replicas\n");
			break;
		}
		// copy list of replicas to update back to gpu
		gpuErrchk(cudaMemcpy(deviceUpdate, hostUpdate, R * sizeof(int), cudaMemcpyHostToDevice));
		updateReplicas <<< BLOCKS, THREADS >> > (deviceSpin, deviceE, deviceUpdate, N);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		printf("\n");


	}



	// Free memory and close files

	cudaFree(deviceSpin);
	cudaFree(deviceE);
	cudaFree(deviceUpdate);
	cudaFree(deviceMagnetization);
	//cudaFree(deviceClusterSizeArray);
	//cudaFree(deviceStack);
	//cudaFree(deviceVisited);
	//cudaFree(device_acceptance_number);

	free(hostMagnetization);
	//free(hostSpin);
	free(hostE);
	free(hostUpdate);
	free(replicaFamily);
	free(energyOrder);
	//free(hostClusterSizeArray);

	fclose(efile);
	fclose(e2file);
	fclose(Xfile);
	fclose(ptfile);
	fclose(nfile);
	fclose(chfile);
	fclose(mfile);

	//fclose(e3file);

	end = clock();
	cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

	sprintf(s, "datasets//%s//2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%dtime.txt", prefix, heating, q, D, N, R, nSteps, run_number);
	FILE* timefile = fopen(s, "w");

	fprintf(timefile, "%f seconds", cpu_time_used);

	fclose(timefile);

	// End
	return 0;
}
