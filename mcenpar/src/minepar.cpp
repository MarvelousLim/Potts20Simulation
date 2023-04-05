#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#include <time.h>

#include "minepar.h"

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
bool cmpf(float x, float y) {
	return fabs(x - y) < EPSILON;
};
void BLTH(int BL, int TH);
void cudaMPImalloc(void** ptr, size_t size);
void cudaMPIfree(void* ptr);
void cudaMPImallocdevstate(int size);
void setup_kernelMPI(int seed);
void cudaPeekAtLastErrorMPI();
void cudaDeviceSynchronizeMPI();
void initializePopulationMPI(char* s, int N, int q);
void cudaMPImemset(void* ptr, int val, size_t size);
void cudaMPImemcpyD2H(void* dst, const void* src, size_t count);
void cudaMPImemcpyH2D(void* dst, const void* src, size_t count);
void equilibrateMPI(char* deviceSpin, float* deviceE, int L, int N, int R, int q, int nSteps, float U, float D, bool heat);
void updateReplicasMPI(char* s, float* E, int* update, int N);
void cudaMPIend();
void CalcPrintAvgE(FILE* efile, float* E, int R, float U) {
	float avg = 0.0;
	for (int i = 0; i < R; i++) {
		avg += E[i];
	}
	avg /= R;
	fprintf(efile, "%f %f\n", U, avg);
	printf("E: %f\n", avg);
}

void printAllE(FILE* e3file, float* E, int R, float U) {
	fprintf(e3file, "%f ", U);
	for (int i = 0; i < R; i++)
		fprintf(e3file, "%f ", E[i]);
	fprintf(e3file, "\n");
}

void CalculateRhoT(const int* replicaFamily, FILE* ptfile, int R, float U) {
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
	fprintf(ptfile, "%f %f\n", U, sum);
	sum /= R;
	printf("RhoT:\t%f\n", sum);
	free(famHist);
}

void Swap(int* A, int i, int j) {
	int temp = A[i];
	A[i] = A[j];
	A[j] = temp;
}

void quicksort(float* E, int* O, int left, int right, int direction) {
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

int resample(float* E, int* O, int* update, int* replicaFamily, int R, float* U, FILE* e2file, FILE* Xfile, bool heat) {
	//std::sort(O, O + R, [&E](int a, int b) {return E[a] > E[b]; }); // greater sign for descending order
	quicksort(E, O, 0, R - 1, 1 - 2 * heat); //Sorts O by energy

	int nCull = 0;
	fprintf(e2file, "%f %f\n", U, E[O[0]]);

	//update energy seiling to the highest available energy
	float U_old = *U;
	float U_new;

	for (int i = 0; i < R; i++) {
		U_new = E[O[i]];
		if ((!heat && U_new < U_old - EPSILON) || (heat && U_new > U_old + EPSILON)) {
			*U = U_new;
			break;
		}
	}

	if (fabs(*U - U_old) < EPSILON) {
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
	fprintf(Xfile, "%f %f\n", *U, X);
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

int main(int argc, char* argv[]) {

	// Parameters:

	int run_number = atoi(argv[1]);	// A number to label this run of the algorithm, used for data keeping purposes, also, a seed
	int seed = run_number;
	//int grid_width = atoi(argv[2]);	// should not be more than 256 due to MTGP32 limits
	int L = atoi(argv[2]);	// Lattice size
	int N = L * L;
	//int R = grid_width * BLOCKS;	// Population size
	int BLOCKS_C = atoi(argv[3]);
	int THREADS_C = atoi(argv[4]);
	int nSteps = atoi(argv[5]);

	int R = BLOCKS_C * THREADS_C;

	// q parameter for potts model, each spin variable can take on values 0 - q-1
	// strictly hardcoded
	int q = 3;

	//Blume-Capel model parameter
	float D = atof(argv[6]);
	bool heat = atoi(argv[7]); // 0 if cooling (default) and 1 if heating


	// initializing files to write in
	const char* heating = heat ? "Heating" : "";

	printf("running 2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%de.txt\n", heating, q, D, N, R, nSteps, run_number);

	char s[100];
	sprintf(s, "datasets//2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%de.txt", heating, q, D, N, R, nSteps, run_number);
	FILE* efile = fopen(s, "w");	// average energy
	sprintf(s, "datasets//2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%de2.txt", heating, q, D, N, R, nSteps, run_number);
	FILE* e2file = fopen(s, "w");	// surface (culled) energy
	sprintf(s, "datasets//2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%dX.txt", heating, q, D, N, R, nSteps, run_number);
	FILE* Xfile = fopen(s, "w");	// culling fraction
	sprintf(s, "datasets//2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%dpt.txt", heating, q, D, N, R, nSteps, run_number);
	FILE* ptfile = fopen(s, "w");	// rho t
	sprintf(s, "datasets//2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%dn.txt", heating, q, D, N, R, nSteps, run_number);
	FILE* nfile = fopen(s, "w");	// number of sweeps
	sprintf(s, "datasets//2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%dch.txt", heating, q, D, N, R, nSteps, run_number);
	FILE* chfile = fopen(s, "w");	// cluster size histogram
	sprintf(s, "datasets//2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%de3.txt", heating, q, D, N, R, nSteps, run_number);
	FILE* e3file = fopen(s, "w");	// cluster size histogram
	sprintf(s, "datasets//2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%ds.txt", heating, q, D, N, R, nSteps, run_number);
	FILE* sfile = fopen(s, "w");	// spin system sample

	size_t fullLatticeByteSize = R * N * sizeof(char);

	// Allocate space on host
	char* hostSpin = (char*)malloc(fullLatticeByteSize);
	float* hostE = (float*)malloc(R * sizeof(float));
	int* hostUpdate = (int*)malloc(R * sizeof(int));
	int* replicaFamily = (int*)malloc(R * sizeof(int));
	int* energyOrder = (int*)malloc(R * sizeof(int));
	for (int i = 0; i < R; i++) {
		energyOrder[i] = i;
		replicaFamily[i] = i;
	}

	// Allocate memory on device
	char* deviceSpin; // s, d_s
	float* deviceE;
	int* deviceUpdate;
	cudaMPImalloc((void**)&deviceSpin, fullLatticeByteSize);
	cudaMPImalloc((void**)&deviceE, R * sizeof(float));
	cudaMPImalloc((void**)&deviceUpdate, R * sizeof(int));
	BLTH(BLOCKS_C,THREADS_C);

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
	cudaMPImallocdevstate(R);
	setup_kernelMPI(seed);
	
	cudaPeekAtLastErrorMPI();
	cudaDeviceSynchronizeMPI();
	
	// Init std random generator for little host part
	srand(seed);

	// Actually working part
	initializePopulationMPI(deviceSpin, N, q);
	cudaPeekAtLastErrorMPI();
	cudaDeviceSynchronizeMPI();	
	cudaMPImemset(deviceE, 0, R * sizeof(float));


	
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

	float upper_energy = N * D + 2 * N;
	float lower_energy = - N * D - 2 * N;
	float U = (heat ? lower_energy : upper_energy);	// U is energy ceiling

	//CalcPrintAvgE(efile, hostE, R, U);


	while ((U >= lower_energy && !heat) || (U <= upper_energy && heat)) {
		fprintf(nfile, "%f %d\n", U, nSteps);
		printf("U:\t%f out of %d; nSteps: %d;\n", U, -2 * N, nSteps);

		// Perform monte carlo sweeps on gpu
		//clock_t begin = clock();

		//cudaMemset(device_acceptance_number, 0, sizeof(int));

		equilibrateMPI(deviceSpin, deviceE, L, N, R, q, nSteps, U, D, heat);// , device_acceptance_number);
		cudaPeekAtLastErrorMPI();
		cudaDeviceSynchronizeMPI();
		cudaMPImemcpyD2H(hostE, deviceE, R * sizeof(int));	
		
		/*
		gpuErrchk(cudaMemcpy(&host_acceptance_number, device_acceptance_number, sizeof(int), cudaMemcpyDeviceToHost));
		printf("acceptance_number: %i\nacceptance_ratio: %02f \n", host_acceptance_number, 100.0 * host_acceptance_number / (N * R * nSteps) );

		clock_t end = clock();
		double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
		printf("Time: %f seconds\n", time_spent);


		gpuErrchk(cudaMemcpy(hostSpin, deviceSpin, N * sizeof(char), cudaMemcpyDeviceToHost)); // take one replica (first)
		for (int i = 0; i < L; i++) {
			for (int j = 0; j < L; j++) {
				printf("%i ", hostSpin[i * L + j]);
			}
			printf("\n");
		}

		//cudaDeviceSynchronize();

		// Create disordered cluster size histogram in particular energy range
		//if (U <= -1.5 * N)
		//	makeClusterHistogram(deviceSpin, deviceE, N, L, BLOCKS, THREADS, U, chfile, deviceVisited, deviceClusterSizeArray, deviceStack, hostClusterSizeArray);

		//deviceEnergy <<< BLOCKS, THREADS >>> (deviceSpin, deviceE, L, N, D);
		//gpuErrchk(cudaPeekAtLastError());
		//gpuErrchk(cudaDeviceSynchronize());
		
		gpuErrchk( cudaMemcpy(hostE, deviceE, R * sizeof(int), cudaMemcpyDeviceToHost) );
		//printAllE(e3file, hostE, R, U);

		
		printf("E: ");
		for (int i = 0; i < 10; i++) {
			printf("%f ", hostE[i]);
		}
		printf("\n");

		printf("O: ");
		for (int i = 0; i < 10; i++) {
			printf("%i ", energyOrder[i]);
		}
		printf("\n");

		printf("E[O]: ");
		for (int i = 0; i < 10; i++) {
			printf("%f ", hostE[energyOrder[i]]);
		}
		printf("\n");
		*/

		// record average energy and rho t
		CalcPrintAvgE(efile, hostE, R, U);
		CalculateRhoT(replicaFamily, ptfile, R, U);
		// perform resampling step on cpu
		// also lowers energy seiling U
		
		int error = resample(hostE, energyOrder, hostUpdate, replicaFamily, R, &U, e2file, Xfile, heat);
		if (error)
		{
			printf("Process ended with zero replicas\n");
			break;
		}
		cudaMPImemcpyD2H(hostSpin, deviceSpin, fullLatticeByteSize);
		fprintf(sfile, "%f\n", U);
		for (int i = 0; i < L; i++) {
			for (int j = 0; j < L; j++) {
				fprintf(sfile, "%i ", hostSpin[i * L + j]);
			}
			fprintf(sfile, "\n");
		}
		// copy list of replicas to update back to gpu
		cudaMPImemcpyH2D(deviceUpdate, hostUpdate, R * sizeof(int));
		updateReplicasMPI(deviceSpin, deviceE, deviceUpdate, N);
		cudaPeekAtLastErrorMPI();
		cudaDeviceSynchronizeMPI();
		printf("\n");

		//gpuErrchk(cudaMemcpy(hostE, deviceE, R * sizeof(int), cudaMemcpyDeviceToHost));
	}
	

	// Free memory and close files
	cudaMPIfree(deviceSpin);
	cudaMPIfree(deviceE);
	cudaMPIfree(deviceUpdate);
	//cudaFree(deviceClusterSizeArray);
	//cudaFree(deviceStack);
	//cudaFree(deviceVisited);
	//cudaFree(device_acceptance_number);

	free(hostSpin);
	free(hostE);
	free(hostUpdate);
	free(replicaFamily);
	free(energyOrder);
	//free(hostClusterSizeArray);
	//free(hostSpin);

	fclose(efile);
	fclose(e2file);
	fclose(Xfile);
	fclose(ptfile);
	fclose(nfile);
	fclose(chfile);
	fclose(e3file);

	// End
	return 0;
}


