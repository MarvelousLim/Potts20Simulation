#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
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

int direction;
bool cmpf(float x, float y) {
	return fabs(x - y) < EPSILON;
};
int cstring_cmp(const void *a, const void *b) 
{ 
/*    const char **ia = (const char **)a;
    const char **ib = (const char **)b;*/
    return strcmp((char *)a, (char *)b);
	/* strcmp functions works exactly as expected from
 * 	comparison function */ 
}
int EnOr_cmpenergy(const void * a, const void * b)
{
    return (direction*((*(struct EnOr*)b).Energy - (*(struct EnOr*)a).Energy ));
}
int EnOr_cmprank(const void * a, const void * b)
{
    return (((*(struct EnOr*)b).Rank - (*(struct EnOr*)a).Rank ));
}
int EnOr_cmpnumber(const void * a, const void * b)
{
    return (((*(struct EnOr*)b).Number - (*(struct EnOr*)a).Number ));
}
void cudaMPIset(int device);
void BLTH(int BL, int TH);
void defND(int Ri, int TH);
void cudaMPImalloc(void** ptr, size_t size);
void cudaMPIfree(void* ptr);
void cudaMPImallocdevstate(int size);
void setup_kernelMPI(unsigned long long seed);
void cudaPeekAtLastErrorMPI();
void cudaDeviceSynchronizeMPI();
void initializePopulationMPI(unsigned long long seed, unsigned long long initial_sequence, Replica* Rep, int q, int R);
void cudaMPImemset(void* ptr, int val, size_t size);
void cudaMPImemcpyD2H(void* dst, const void* src, size_t count);
void cudaMPImemcpyH2D(void* dst, const void* src, size_t count);
void equilibrateMPI(unsigned long long seed, unsigned long long initial_sequence, Replica* Rep, EnOr* deviceEnOr, int R, int q, int nSteps, float U, float D, int heat);
void updateReplicasMPI(Replica* Rep, float* E, int* update);
void copyreploffMPI(Replica* Rep, EnOr* reploff, int R);
void cudaCalcParSum(Replica* Repptr, int Rc);
void cudaresampleKer(Replica* Repptr, Replica* Repptrdest);
void cudablocksumKer(Replica* Repptr, int Rlocalc, unsigned int* Rnewc);
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

int resample(EnOr* E, int R, float* U, FILE* e2file, FILE* Xfile, int heat) {
	//std::sort(O, O + R, [&E](int a, int b) {return E[a] > E[b]; }); // greater sign for descending order
	//quicksort(E, O, 0, R - 1, 1 - 2 * heat); //Sorts O by energy
	qsort(E,R,sizeof(EnOr),EnOr_cmpenergy);

	int nCull = 0;
	//fprintf(e2file, "%f %f\n", U, E[O[0]]);
	fprintf(e2file, "%f %f\n", U, E[0]);

	//update energy seiling to the highest available energy
	float U_old = *U;
	float U_new;

	for (int i = 0; i < R; i++) {
		//U_new = E[O[i]];
		U_new = E[i].Energy;
		if ((!heat && U_new < U_old - EPSILON) || (heat && U_new > U_old + EPSILON)) {
			*U = U_new;
			break;
		}
	}

	if (fabs(*U - U_old) < EPSILON) {
		return 1; // out of replicas
	}

	while ((!heat && E[nCull].Energy >= *U - EPSILON) || (heat && E[nCull].Energy <= *U + EPSILON)) {
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
	for (int i = 0; i < R; i++){
		E[i].Roff = 1;}
	if (nCull < R) {
		for (int i = 0; i < nCull; i++) {
			// random selection of unculled replica
			int r = (rand() % (R - nCull)) + nCull; // different random number generator for resampling
			E[i].Roff = 0; E[r].Roff++;
		}
	}

	return 0;
}

int main(int argc, char* argv[]) {

	int wank, nprocs, namelen, myrank, color, n;
	size_t bytes;
	int i, j;
	MPI_Comm nodeComm;
	MPI_Datatype ReplicaMPI, EnOrMPI;
	int EnOrCount=4;
	int array_of_blocklengthsEnOr[] = { 1,1,1,1 };
	MPI_Aint array_of_displacementsEnOr[] = { offsetof( EnOr, Energy ),
                                      offsetof( EnOr, Number ),
                                      offsetof( EnOr, Rank ),
                                      offsetof( EnOr, Roff)  };
    MPI_Datatype array_of_typesEnOr[] = { MPI_FLOAT, MPI_INT, MPI_INT, MPI_UNSIGNED };
	char host_name[MPI_MAX_PROCESSOR_NAME];
	char (*host_names)[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &wank);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	MPI_Get_processor_name(host_name,&namelen);

	bytes = nprocs * sizeof(char[MPI_MAX_PROCESSOR_NAME]);
	host_names = (char (*)[MPI_MAX_PROCESSOR_NAME]) malloc(bytes);
	strcpy(host_names[wank], host_name);
	for (n=0; n<nprocs; n++)
	{
		MPI_Bcast(&(host_names[n]),MPI_MAX_PROCESSOR_NAME, MPI_CHAR, n, MPI_COMM_WORLD);
	}
    printf("%d %d %s\n", wank, nprocs, host_name);
	qsort(host_names, nprocs,  sizeof(char[MPI_MAX_PROCESSOR_NAME]), cstring_cmp);
	color = 0;
	for (n=0; n<nprocs; n++)
	{ 
		if(n>0&&strcmp(host_names[n-1], host_names[n])) color++;
		if(strcmp(host_name, host_names[n]) == 0) break;
	}
	MPI_Comm_split(MPI_COMM_WORLD, color, 0, &nodeComm);
	MPI_Comm_rank(nodeComm, &myrank);

    /* Assign device to MPI process*/
	printf ("Assigning device %d  to process on node %s rank %d \n", myrank,  host_name, wank );
	cudaMPIset(myrank);
	
	exit(1);

	// Parameters:
	int run_number, BLOCKS_C, THREADS_C, nSteps;
	unsigned long long seed, initial_sequence = 0;
	unsigned int seedh;
	if(!wank){
		run_number = atoi(argv[1]);	// A number to label this run of the algorithm, used for data keeping purposes, also, a seed
		//int grid_width = atoi(argv[2]);	// should not be more than 256 due to MTGP32 limits
		//int L = atoi(argv[2]);	// Lattice size
		//int N = L * L;
		//int R = grid_width * BLOCKS;	// Population size
		BLOCKS_C = atoi(argv[2]);
		THREADS_C = atoi(argv[3]);
		nSteps = atoi(argv[4]);
	}
	MPI_Bcast(&run_number,1,MPI_INT,0,MPI_COMM_WORLD);
	seed = run_number+wank; seedh = run_number;
	MPI_Bcast(&BLOCKS_C,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&THREADS_C,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&nSteps,1,MPI_INT,0,MPI_COMM_WORLD);

	int R = BLOCKS_C * THREADS_C, Ract; // R is "typical" number of replicas per one node, Ract is actual number at the current step
	Ract = R;
	int Rlen=10000,Rlenadd=10000; // Rlen is length of array Rcur, after reallocation it is Rlen+Rlenadd
	int *Rcur,*Rcurnew; // Rcur is array with population size on each step, Rcurnew is the same array, when length has been exceeded 
	Rcur=(int*)malloc(Rlen*sizeof(int));
	int ir=0; // ir is counter in main cycle


	MPI_Type_create_struct( EnOrCount, array_of_blocklengthsEnOr, array_of_displacementsEnOr,
                        array_of_typesEnOr, &EnOrMPI );

    MPI_Type_commit( &EnOrMPI );

	
	Replica* hostSpin;
	Replica* deviceSpin;
	Replica* deviceSpinNew;
	FILE *efile,*e2file,*Xfile,*ptfile,*nfile,*chfile,*e3file,*sfile;

	// q parameter for potts model, each spin variable can take on values 0 - q-1
	// strictly hardcoded
	int q = 3;

	float D;
	int heat;
	if(!wank){
		//Blume-Capel model parameter
		D = atof(argv[5]);
		heat = atoi(argv[6]); // 0 if cooling (default) and 1 if heating

		// initializing files to write in
		const char* heating = heat ? "Heating" : "";
		printf("running 2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%de.txt\n", heating, q, D, N, R, nSteps, run_number);
		char s[100];
		sprintf(s, "datasets//2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%de.txt", heating, q, D, N, R, nSteps, run_number);
		efile = fopen(s, "w");	// average energy
		sprintf(s, "datasets//2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%de2.txt", heating, q, D, N, R, nSteps, run_number);
		e2file = fopen(s, "w");	// surface (culled) energy
		sprintf(s, "datasets//2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%dX.txt", heating, q, D, N, R, nSteps, run_number);
		Xfile = fopen(s, "w");	// culling fraction
		sprintf(s, "datasets//2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%dpt.txt", heating, q, D, N, R, nSteps, run_number);
		ptfile = fopen(s, "w");	// rho t
		sprintf(s, "datasets//2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%dn.txt", heating, q, D, N, R, nSteps, run_number);
		nfile = fopen(s, "w");	// number of sweeps
		sprintf(s, "datasets//2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%dch.txt", heating, q, D, N, R, nSteps, run_number);
		chfile = fopen(s, "w");	// cluster size histogram
		sprintf(s, "datasets//2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%de3.txt", heating, q, D, N, R, nSteps, run_number);
		e3file = fopen(s, "w");	// cluster size histogram
		sprintf(s, "datasets//2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%ds.txt", heating, q, D, N, R, nSteps, run_number);
		sfile = fopen(s, "w");	// spin system sample
	}
	MPI_Bcast(&D,1,MPI_FLOAT,0,MPI_COMM_WORLD);
	MPI_Bcast(&heat,1,MPI_INT,0,MPI_COMM_WORLD);
	direction=2*heat-1;
//### Move allocate space on host to main cycle body ### free devStates
	//size_t fullLatticeByteSize = R * N * sizeof(char);
	size_t fullLBS = R * sizeof(Replica);

	// Allocate space on host
	//hostSpin = (Replica*)malloc(fullLBS);
	//float* hostE;
	EnOr* hostEnOr;
	EnOr* hostrootEnOr;
	int* hostUpdate = (int*)malloc(R * nprocs * sizeof(int));
	int* replicaFamily = (int*)malloc(R * nprocs * sizeof(int));
	//unsigned int* reploffH = (unsigned int*)malloc(R * sizeof(unsigned int));
	//int* energyOrder = (int*)malloc(R * sizeof(int));
	for (int i = 0; i < R*nprocs; i++) {
		//energyOrder[i] = i;
		replicaFamily[i] = i;
	}
	
	int* Rcroot; // Rcroot is nprocs-size array just for current temperature step
	int *recvcounts, *displs; // arrays for MPI_Gatherv, MPI_Scatterv


	Rcroot=(int *)malloc(sizeof(int)*nprocs);
	
	// Allocate memory on device
	//char* deviceSpin; // s, d_s
	//float* deviceE;
	EnOr* deviceEnOr;
	int* deviceUpdate;
	//unsigned int* reploffD;
	unsigned int* Ridev;
	cudaMPImalloc((void**)&deviceSpin, fullLBS);
	//cudaMPImalloc((void**)&deviceE, R * sizeof(float));
	cudaMPImalloc((void**)&deviceUpdate, R * sizeof(int));
	//cudaMPImalloc((void**)&reploffD, R * sizeof(unsigned int));
	cudaMPImalloc((void**)&Ridev,sizeof(int));
	//BLTH(BLOCKS_C,THREADS_C);

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

	// Init Philox in new version is not needed - all random numbers are generated
	// directly from seed, initial_sequence
	// cudaMPImallocdevstate(R);
	// setup_kernelMPI(seed);
	
	cudaPeekAtLastErrorMPI();
	cudaDeviceSynchronizeMPI();
	
	// Init std random generator for little host part
	if(!wank){
		srand(seedh);
		hostrootEnOr = (EnOr*)malloc(R * nprocs * sizeof(EnOr));
		recvcounts = (int*)malloc(nprocs*sizeof(int));
		displs = (int*)malloc(nprocs*sizeof(int));
	}

	// Actually working part
	initializePopulationMPI(seed, initial_sequence, deviceSpin, q, R);initial_sequence+=R;
	cudaPeekAtLastErrorMPI();
	cudaDeviceSynchronizeMPI();	
	//cudaMPImemset(deviceE, 0, R * sizeof(float));


	
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

// replace condition in cycle by standard, when root process determines, whether to continue
	while ((U >= lower_energy && !heat) || (U <= upper_energy && heat)) {
		if (ir==Rlen){ // reallocating Rlen
			Rcurnew=(int*)malloc((Rlen+Rlenadd)*sizeof(int));
			memcpy(Rcurnew,Rcur,Rlen*sizeof(int));free(Rcur);Rcur=Rcurnew;Rlen+=Rlenadd;
		}
		fprintf(nfile, "%f %d\n", U, nSteps);
		printf("U:\t%f out of %d; nSteps: %d;\n", U, -2 * N, nSteps);
		Rcur[i]=Ract;

// defND here, define R at next step at the bottom of the cycle, add if(idx<R) in cuda functions
// defND after replica exchange, make R[] unsigned int instead of int
		defND(Ract,THREADS_C);
		fullLBS = Ract * sizeof(Replica);

		// Perform monte carlo sweeps on gpu
		//clock_t begin = clock();

		//cudaMemset(device_acceptance_number, 0, sizeof(int));
		cudaMPImalloc((void**)&deviceEnOr, Ract * sizeof(EnOr));
		equilibrateMPI(seed, initial_sequence, deviceSpin, deviceEnOr, Ract, q, nSteps, U, D, heat);initial_sequence+=Ract;// , device_acceptance_number);
		cudaPeekAtLastErrorMPI();
		cudaDeviceSynchronizeMPI();
		hostEnOr = (EnOr*)malloc(Ract * sizeof(EnOr));
		cudaMPImemcpyD2H(hostEnOr, deviceEnOr, Ract * sizeof(EnOr));
		for(i=0;i<Ract;i++){
			hostEnOr[i].Number=i;hostEnOr[i].Rank=wank;hostEnOr[i].Roff=0;
		}
		MPI_Gather(&Ract, 1, MPI_INT, Rcroot, 1, MPI_INT, 0, MPI_COMM_WORLD);
		if(!wank){
			displs[0]=0;
			for(i=0;i<nprocs-1;i++){
				recvcounts[i]=Rcroot[i];displs[i+1]=displs[i]+recvcounts[i];
			}
			recvcounts[nprocs-1]=Rcroot[nprocs-1];
		}
		MPI_Gatherv(hostEnOr,Ract,EnOrMPI,hostrootEnOr,recvcounts,displs,EnOrMPI,0,MPI_COMM_WORLD);
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
		// todo CalcPrintAvgE(efile, hostE, R, U);
		// CalculateRhoT(replicaFamily, ptfile, R, U);
		// perform resampling step on cpu
		// also lowers energy seiling U
		
		if(!wank){
			int error = resample(hostrootEnOr, R*nprocs, &U, e2file, Xfile, heat);
			if (error)
			{
				printf("Process ended with zero replicas\n");
				break;
			}
			qsort(hostrootEnOr,R*nprocs,sizeof(EnOr),EnOr_cmprank);
		}
		MPI_Scatterv(hostrootEnOr,recvcounts,displs,EnOrMPI,hostEnOr,Ract,EnOrMPI,0,MPI_COMM_WORLD);
		qsort(hostEnOr,Ract,sizeof(EnOr),EnOr_cmpnumber);
		// todo CalculateRhoT(replicaFamily, ptfile, R, U);
		
		hostSpin = (Replica*)malloc(fullLBS);
		cudaMPImemcpyD2H(hostSpin, deviceSpin, fullLBS);
		if(!wank){
			fprintf(sfile, "%f\n", U);
			for (i = 0; i < L; i++) {
				for (j = 0; j < L; j++) {
					fprintf(sfile, "%i ", hostSpin[0].sp[i * L + j]);
				}
				fprintf(sfile, "\n");
			}
		}
		free(hostSpin);
		// copy list of replicas to update back to gpu
		//cudaMPImemcpyH2D(deviceUpdate, hostUpdate, Ract * sizeof(int));
		cudaMPImemcpyH2D(deviceEnOr, hostEnOr, Ract * sizeof(EnOr));
		copyreploffMPI(deviceSpin, deviceEnOr, Ract);
		cudaMPIfree(deviceEnOr);free(hostEnOr);
		// replicas exchange here
		

		defND(Ract,THREADS_C);
		cudaCalcParSum(deviceSpin,Ract);

		cudaMPImemset(Ridev, 0, sizeof(unsigned int));
		cudablocksumKer(deviceSpin, Ract, Ridev);
		cudaMPImemcpyD2H(&Ract, Ridev, sizeof(unsigned int));		
		fullLBS = Ract * sizeof(Replica);
		cudaMPImalloc((void**)&deviceSpinNew, fullLBS);
		cudaresampleKer(deviceSpin,deviceSpinNew);
		
		//updateReplicasMPI(deviceSpin, deviceE, deviceUpdate);
		cudaMPIfree(deviceSpin);
		deviceSpin=deviceSpinNew;


		
		cudaPeekAtLastErrorMPI();
		cudaDeviceSynchronizeMPI();
		printf("\n");

		//gpuErrchk(cudaMemcpy(hostE, deviceE, R * sizeof(int), cudaMemcpyDeviceToHost));
		ir++;
	}
	

	// Free memory and close files
	cudaMPIfree(deviceSpin);
	//cudaMPIfree(deviceE);
	cudaMPIfree(deviceUpdate);
	//cudaFree(deviceClusterSizeArray);
	//cudaFree(deviceStack);
	//cudaFree(deviceVisited);
	//cudaFree(device_acceptance_number);

	//free(hostSpin);
	//free(hostE);
	free(hostUpdate);
	free(replicaFamily);
	//free(energyOrder);
	//free(hostClusterSizeArray);
	//free(hostSpin);

	fclose(efile);
	fclose(e2file);
	fclose(Xfile);
	fclose(ptfile);
	fclose(nfile);
	fclose(chfile);
	fclose(e3file);
	
	if(!wank){
		free(hostrootEnOr);free(recvcounts);free(displs);
	}

	MPI_Type_free( &EnOrMPI );

	MPI_Finalize();
	// End
	return 0;
}


