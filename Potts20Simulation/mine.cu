#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <queue>
#include <vector>
#include <map>

#include "omp.h"

#include <cuda.h>
#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#define NNEIBORS 4 // number of nearest neighbors, is 4 for 2d lattice
#define BLOCKS 200
//#define MAX_THREADS 256 // curand property
//#define MAX_NGENERATORS BLOCKS * THREADS

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

// more or less host function

struct neibors_indexes {
	int up;
	int down;
	int left;
	int right;
};

__host__ __device__ struct neibors_indexes SLF(int j, int L, int N) {
	struct neibors_indexes result;
	result.right = (j + 1) % L + L * (j / L);
	result.left = (j - 1 + L) % L + L * (j / L); // L member is for positivity
	result.down = (j + L) % N;
	result.up = (j - L + N) % N; // N member is for positivity
	return result;
}

struct neibors {
	char up;
	char down;
	char left;
	char right;
};

__host__ __device__ struct neibors get_neibors_values(char* s, struct neibors_indexes n_i, int replica_shift) {
	return { s[n_i.up + replica_shift], s[n_i.down + replica_shift], s[n_i.left + replica_shift], s[n_i.right + replica_shift] };
}


__host__ __device__ int LocalE(char currentSpin, struct neibors n) { 	// Computes energy of spin i with neighbors a, b, c, d 
	return -(currentSpin == n.up) - (currentSpin == n.down) - (currentSpin == n.left) - (currentSpin == n.right);
}
__host__ __device__ int DeltaE(char currentSpin, char suggestedSpin, struct neibors n) { // Delta of local energy while i -> e switch
	return LocalE(suggestedSpin, n) - LocalE(currentSpin, n);
}


void hostEnergy(char* s, thrust::host_vector<int>& E, int R, int L, int N) {
	for (int r = 0; r < R; r++) {
		int sum = 0;
		for (int j = 0; j < N; j++) {
			// 0.5 by doubling the summarize (stupid)
			int replica_shift = r * N;
			char i = s[j + replica_shift]; // current spin value
			struct neibors_indexes n_i = SLF(j, L, N);
			struct neibors n = get_neibors_values(s, n_i, replica_shift); // we look into r replica and j spin
			sum += LocalE(i, n);
		}
		E[r] = sum / 2;
	}
}

void CalcPrintAvgE(std::ofstream& efile, const thrust::host_vector<int>& E, int U, int R) {
	float avg = 0.0;
	for (int i = 0; i < R; i++)
		avg += E[i];
	avg /= R;
	efile << U << " " << avg << std::endl;
}

// Part of disordered cluster histogram algorithm
int BFS(char* s, int start, std::vector<bool>& visited, int L, int N, int replica_shift) {

	std::queue<int> BFS_queue;
	BFS_queue.push(start);
	visited[start] = 1;
	int clusterSize = 0;
	char spinValue = s[start + replica_shift];

	while (!BFS_queue.empty()) {
		int currentIndex = BFS_queue.front();
		BFS_queue.pop();
		clusterSize++;
		struct neibors_indexes n = SLF(currentIndex, L, N);
		std::vector<int> possibleIndexes = { n.up, n.down, n.left, n.right };
		for (int newIndex : possibleIndexes)
			if ((s[newIndex + replica_shift] == spinValue) && (!visited[newIndex])) {
				BFS_queue.push(newIndex);
				visited[newIndex] = 1;
			}
	}
	return clusterSize;
}

void makeClusterHistogram(char* s, thrust::host_vector<int>& E, int q, int N, int L, int R, int MaxHistNumber, int U, std::ofstream& chfile) {
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
	std::map<int, int> clusterSizeFreqMap; // should be safe to assume the default value as 0 (?)
	int HistNumber = 0;
	#pragma omp for
	for (int r = 0; r < R; r++) {
		std::vector<bool> visited(N); // default-false for every replica
		int replica_shift = r * N;
		if (E[r] == U - 1 && HistNumber < MaxHistNumber) {
			HistNumber++;
			// Go sequentially through lattice and assign spins to clusters
			for (int i = 0; i < N; i++) {
				if (!visited[i]) {
					int currentClusteSize = BFS(s, i, visited, L, N, replica_shift);
					#pragma omp atomic
					clusterSizeFreqMap[currentClusteSize]++;
				}
			}
		}
		else if (HistNumber >= MaxHistNumber)
			break;
	}
	// write results to output files
	if (HistNumber > 0) {
		chfile << HistNumber << " ";
		for (auto& item : clusterSizeFreqMap) {
			int size = item.first;
			int freq = item.second;
			chfile << size << " " << freq << " ";
		}
		chfile << std::endl;
	}
}

void CalculateRhoT(const thrust::host_vector<int>& replicaFamily, int R, std::ofstream& ptfile, int U) {
	// histogram of family sizes
	std::vector<int> famHist(R);
	for (int i = 0; i < R; i++) {
		famHist[replicaFamily[i]]++;
	}
	float sum = 0;
	for (int i = 0; i < R; i++) {
		sum += famHist[i] * famHist[i];
	}
	sum /= R;
	ptfile << U << " " << sum << std::endl;
}


// device functions

__global__ void initializePopulation(curandStateMtgp32* state, char* s, int N, int q, int R) {
	/*---------------------------------------------------------------------------------------------
		Initializes population on gpu(!) by randomly assigning each spin a value from 0 to q-1
	----------------------------------------------------------------------------------------------*/
	int r = threadIdx.x + blockIdx.x * blockDim.x;
	for (int k = 0; k < N; k++) {
		int arrayIndex = r * N + k;
		char spin = curand(&state[blockIdx.x]) % q;
		s[arrayIndex] = spin;
	}
}

__global__ void equilibrate(curandStateMtgp32* state, char* s, int* E, int L, int N, int R, int q, int nSteps, int U) {
	/*---------------------------------------------------------------------------------------------
		Main Microcanonical Monte Carlo loop.  Performs update sweeps on each replica in the
		population;
	---------------------------------------------------------------------------------------------*/

	int r = threadIdx.x + blockIdx.x * blockDim.x;
	int replica_shift = r * N;
	for (int k = 0; k < N * nSteps; k++) {
		int j = curand(&state[blockIdx.x]) % N;
		char currentSpin = s[j + replica_shift];
		char suggestedSpin = curand(&state[blockIdx.x]) % q;
		struct neibors_indexes n_i = SLF(j, L, N);
		struct neibors n = get_neibors_values(s, n_i, replica_shift);
		int dE = DeltaE(currentSpin, suggestedSpin, n);
		if (E[r] + dE < U) {
			E[r] = E[r] + dE;
			s[j + replica_shift] = suggestedSpin;
		}
	}
}

void resample(const thrust::host_vector<int>& E, thrust::host_vector<int>& O, thrust::host_vector<int>& update,
	thrust::host_vector<int>& replicaFamily, int R, int U, std::ofstream& e2file, std::ofstream& Xfile) {
	//quicksort(E, O, 0, R - 1); // Sorts O by energy
	thrust::sort(O.begin(), O.end(), [&E](int a, int b) {return E[a] > E[b]; }); // greater sign for descending order
	int nCull = 0;
	e2file << U << " " << E[O[0]] << std::endl;
	while (E[O[nCull]] == U - 1) {
		nCull++;
		if (nCull == R) {
			break;
		}
	}
	// culling fraction
	double X = (double)nCull / R;
	Xfile << U << " " << X << std::endl;
	std::cout << "Culling fraction:\t" << X << std::endl;
	for (int i = 0; i < R; i++)
		update[i] = i;
	if (nCull < R) {
		for (int i = 0; i < nCull; i++) {
			// random selection of unculled replica
			int r = (std::rand() % (R - nCull)) + nCull; // different random number generator for
			update[O[i]] = O[r];
			replicaFamily[O[i]] = replicaFamily[O[r]];
		}
	}
}

__global__ void updateReplicas(char* s, int* E, int* update, int N, int R) {
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

template<class T>
void PrintVector(const T& v, std::string prefix) {
	std::cout << prefix;
	for (auto& item : v)
		std::cout << item << " ";
	std::cout << std::endl;
}

void PrintArray(char* s, std::string prefix, int L, int R) {
	std::cout << prefix << std::endl;
	int N = L * L;
	for (int r = 0; r < R; r++) {
		std::cout << "replica " << r << std::endl;
		for (int i = 0; i < L; i++) {
			for (int j = 0; j < L; j++) {
				std::cout << (int)s[j + i * L + r * N] << "\t";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

int main(int argc, char* argv[]) {
	// Parameters:
	int nSteps = 1;
	int q = 20;	// q parameter for potts model, each spin variable can take on values 0 - q-1
	int U = 1;	// U is energy ceiling

	// random number generation
	curandStateMtgp32* devMTGPStates;
	mtgp32_kernel_params* devKernelParams;

	int run_number = atoi(argv[1]);	// A number to label this run of the algorithm, used for data keeping purposes, also, a seed
	int seed = run_number;
	int grid_width = atoi(argv[2]);	// should not be more than 256 due to MTGP32 limits
	int L = atoi(argv[3]);	//Lattice size.  Total number of spins is N=L^2
	int N = L * L;
	int R = grid_width * BLOCKS;	// Population size

	// initializing files to write in
	std::stringstream s;
	s << "datasets//L" << L << "_R" << R << "_run" << run_number;
	std::string S = s.str();
	std::ofstream efile(S + "e.txt", std::ofstream::trunc);	// average energy
	std::ofstream e2file(S + "e2.txt", std::ofstream::trunc);	// surface (culled) energy
	std::ofstream Xfile(S + "X.txt", std::ofstream::trunc);	// culling fraction
	std::ofstream ptfile(S + "pt.txt", std::ofstream::trunc);	// rho t
	std::ofstream nfile(S + "n.txt", std::ofstream::trunc);	// number of sweeps
	std::ofstream chfile(S + "ch.txt", std::ofstream::trunc);	// cluster size histogram


	size_t fullLatticeByteSize = R * N * sizeof(char);

	// Allocate space on host 
	char* hostSpin = (char*)malloc(fullLatticeByteSize); // s, h_s
	thrust::host_vector<int> hostE(R);
	thrust::host_vector<int> replicaFamily(R);
	thrust::host_vector<int> energyOrder(R);
	thrust::host_vector<int> hostUpdate(R);
	for (int i = 0; i < R; i++) {
		energyOrder[i] = i;
		replicaFamily[i] = i;
	}
	// Allocate memory on device
	char* deviceSpin; // s, d_s
	thrust::device_vector<int> deviceE(R);
	thrust::device_vector<int> deviceUpdate(R);
	int* deviceEPointer = thrust::raw_pointer_cast(deviceE.data()); // assume, that pointers does not changes since there are no reallocations
	int* deviceUpdatePointer = thrust::raw_pointer_cast(deviceUpdate.data());
	cudaMalloc((void**)&deviceSpin, fullLatticeByteSize);

	// Init MTGP32
	cudaMalloc((void**)&devMTGPStates, BLOCKS * sizeof(curandStateMtgp32));
	cudaMalloc((void**)&devKernelParams, sizeof(mtgp32_kernel_params));
	curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, devKernelParams);
	curandMakeMTGP32KernelState(devMTGPStates, mtgp32dc_params_fast_11213, devKernelParams, BLOCKS, seed);

	// Init std random generator for little host part
	std::srand(seed);

	// Actually working part
	initializePopulation << <BLOCKS, grid_width >> > (devMTGPStates, deviceSpin, N, q, R);
	cudaMemcpy(hostSpin, deviceSpin, fullLatticeByteSize, cudaMemcpyDeviceToHost);
	//	PrintArray(hostSpin, "s", L, R); // debug
	hostEnergy(hostSpin, hostE, R, L, N);
	//	PrintVector(hostE, "E "); // debug
	deviceE = hostE;

	int loop = 0;
	while (U > -2 * N) {
		loop++;
		// Adjust the sweep schedule
		// Most sweeps are performed in the region when simulation is most difficult
		if (U < -(3 * N / 2))
			nSteps = 10;
		else if (U < -N / 2)
			nSteps = 30;
		else // (U >= -N / 2)
			nSteps = 2;

		nfile << U << " " << nSteps << std::endl;
		std::cout << "U:\t" << U << " out of " << -2 * N << "; nSteps: " << nSteps << ";" << std::endl;
		// Perform monte carlo sweeps on gpu
		equilibrate<<<BLOCKS, grid_width>>>(devMTGPStates, deviceSpin, deviceEPointer, L, N, R, q, nSteps, U);
		//		PrintArray(hostSpin, "s", L, R); // debug
		hostE = deviceE;
		//		PrintVector(hostE, "E "); // debug

		int numHist = 100000;
		// Create disordered cluster size histogram in particular energy range
		if (U <= -1.5 * N) {
			// Copy energy and spin configuration back to host
			cudaMemcpy(hostSpin, deviceSpin, fullLatticeByteSize, cudaMemcpyDeviceToHost);
			makeClusterHistogram(hostSpin, hostE, q, N, L, R, numHist, U, chfile);
		}
		// record average energy and rho t
		CalcPrintAvgE(efile, hostE, U, R);
		CalculateRhoT(replicaFamily, R, ptfile, U);
		// perform resampling step on cpu
		resample(hostE, energyOrder, hostUpdate, replicaFamily, R, U, e2file, Xfile);
		U--;
//		PrintVector(energyOrder, "O "); // debug
//		PrintVector(hostUpdate, "update "); // debug
				// copy list of replicas to update back to gpu
		deviceUpdate = hostUpdate;
		updateReplicas<<<BLOCKS, grid_width>>>(deviceSpin, deviceEPointer, deviceUpdatePointer, N, R);
//		cudaMemcpy(hostSpin, deviceSpin, fullLatticeByteSize, cudaMemcpyDeviceToHost); // debug
//		PrintArray(hostSpin, "s_updated", L, R); // debug
//		PrintVector(replicaFamily, "replicaFamily "); // debug
	}
	CalcPrintAvgE(efile, hostE, U, R);

	// Free memory and close files
	cudaFree(devMTGPStates);
	cudaFree(devKernelParams);
	cudaFree(deviceSpin);
	free(hostSpin);
	efile.close();
	e2file.close();
	Xfile.close();
	ptfile.close();
	nfile.close();
	chfile.close();

	// End
	return 0;
}
