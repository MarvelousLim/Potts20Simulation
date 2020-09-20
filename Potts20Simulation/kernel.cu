#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>

#define NGENERATORS 64 * 256
#define BLOCK 64
#define THREADS 256

struct triangle {
	float CV;  // characteristic time for vert
	float CH;  //                     for hor
	float AV;  // actual time for vert
	float AH;  //             for hor
	float SV;  // state for vert
	float SH;  //       for hor
};

struct coord {
	int X;
	int Y;
};

#define FULL_MASK 0xffffffff

__device__ float warpReduceSum(float val)
{
	for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
		val += __shfl_down_sync(FULL_MASK, val, offset);
	return val;
}

__device__ int warpReduceMax(int val) {
	for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
		val = fmaxf(val, __shfl_down(val, offset));
	return val;
}

__global__ void initLattice(curandStateMtgp32* state, int N, struct triangle* devLattice, float t0, float u)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	float r;
	struct triangle* x;
	if (tid < NGENERATORS)
	{
		for (int i = tid; i < N; i += NGENERATORS)
		{
			x = devLattice + i;
			r = curand_uniform(&state[blockIdx.x]);
			x->CV = t0 * exp(u * r);
			r = curand_uniform(&state[blockIdx.x]);
			x->CH = t0 * exp(u * r);

			r = curand_uniform(&state[blockIdx.x]);
			x->AV = (-x->CV * log(r));
			r = curand_uniform(&state[blockIdx.x]);
			x->AH = (-x->CH * log(r));

			r = curand_uniform(&state[blockIdx.x]);
			x->SV = r;
			r = curand_uniform(&state[blockIdx.x]);
			x->SH = r;
		}
	}
}


__global__ void initParticles(int N, struct coord* devParticles, int center)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	struct coord* x;
	if (tid < NGENERATORS)
	{
		for (int i = tid; i < N; i += NGENERATORS)
		{
			x = devParticles + i;
			x->X = center;
			x->Y = center;
		}
	}
}

__global__ void calculateRSquared(float* devResults, int N, struct coord* devParticles, int center)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	struct coord* x;
	float r2;
	if (tid < NGENERATORS)
	{
		for (int i = tid; i < N; i += NGENERATORS)
		{
			x = devParticles + i;
			r2 = (x->X - center) * (x->X - center) + (x->Y - center) * (x->Y - center);
			float reductionRes = warpReduceSum(r2);
			if ((threadIdx.x & (warpSize - 1)) == 0)
				atomicAdd(devResults, reductionRes);
		}
	}
}

__global__ void step(curandStateMtgp32* state, struct coord* devParticles, struct triangle* devLattice, int N, int L, int center, int* devR)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < NGENERATORS)
	{
		for (int i = tid; i < N; i += NGENERATORS)
		{
			float r = curand_uniform(&state[blockIdx.x]);
			struct coord* x = devParticles + i;
			//jump
			int d1 = (devLattice + x->X + L * x->Y)->SH + (devLattice + (x->X - 1) + L * x->Y)->SH
				+ (devLattice + x->X + L * x->Y)->SV + (devLattice + x->X + L * (x->Y - 1))->SV;
			if (d1 != 0)
			{
				float  d2 = d1 * r;

				d2 -= (devLattice + x->X + L * x->Y)->SH;
				if (d2 <= 0)
					x->X++;
				else
				{
					d2 -= (devLattice + (x->X - 1) + L * x->Y)->SH;
					if (d2 <= 0)
						x->X--;
					else
					{
						d2 -= (devLattice + x->X + L * x->Y)->SV;
						if (d2 <= 0)
							x->Y++;
						else
							x->Y--;
					}
				}
			}

			int d = max(abs(x->X - center), abs(x->Y - center));

			int reductionRes = warpReduceMax(d);
			if ((threadIdx.x & (warpSize - 1)) == 0)
				atomicMax(devR, reductionRes);

		}
	}
}
__global__ void stepLattice(curandStateMtgp32* state, int N, struct triangle* devLattice)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	float r1, r2;
	struct triangle* x;
	if (tid < NGENERATORS)
	{
		for (int i = tid; i < N; i += NGENERATORS)
		{
			x = devLattice + i;
			r1 = curand_uniform(&state[blockIdx.x]);
			r2 = curand_uniform(&state[blockIdx.x]);

			x->AV--;
			if (x->AV < 0)
			{
				x->SV = 1 - x->SV;
				x->AV = (-(x->CV) * log(r1));
			}
			x->AH--;
			if (x->AH < 0)
			{
				x->SH = 1 - x->SH;
				x->AH = (-(x->CH) * log(r2));
			}
		}
	}
}

int main(int argc, char* argv[]) {

	int L = atoi(argv[1]);  //linear size of lattice
	int PN = atoi(argv[2]); //Particle number
	int linesNumber = atoi(argv[3]); //number of runs
	linesNumber = 1000;
	float u = atof(argv[4]);
	float t0 = atof(argv[5]);
	int seed = atoi(argv[6]); //seed

	int maxT = 10000000;
	int particle_dist_timing = atoi(argv[7]); // for u=60, t_0=1, seed=0

	int t, center = L / 2;

	/*
	FILE* D_output;
	char s[100];
	sprintf(s, "L%d_PN%d_u%g_t%g_seed%d.txt", L, PN, u, t0, seed, p);
	D_output = fopen(s, "w");
	printf("write into %s\n", s);
	
	FILE *D_particle_dist;
	char s_particle_dist[100];
	sprintf(s_particle_dist, "PD_L%d_PN%d_u%g_t%g_seed%d_time_%d.txt", L, PN, u, t0, seed, particle_dist_timing);
	D_particle_dist = fopen(s_particle_dist, "w");
	printf("write into %s\n", s_particle_dist);
	*/

	FILE *D_map;
	char s_map[100];
	sprintf(s_map, "MAP_L%d_PN%d_u%g_t%g_seed%dp.txt", L, PN, u, t0, seed);
	D_map = fopen(s_map, "w");
	printf("write into %s\n", s_map);

	curandStateMtgp32* devMTGPStates;
	mtgp32_kernel_params* devKernelParams;
	float* hostResults, * devResults;
	int* R, * devR;
	struct triangle* Lattice;
	struct triangle* devLattice;
	struct coord* Particles;
	struct coord* devParticles;

	// Allocate space for lattice on host 
	R = (int*)calloc(1, sizeof(int));
	Lattice = (struct triangle*)calloc(L * L, sizeof(struct triangle));
	Particles = (struct coord*)calloc(PN, sizeof(struct coord));
	hostResults = (float*)calloc(1, sizeof(float));

	// Allocate space for results on device 
	cudaMalloc((void**)&devR, 1 * sizeof(int));
	cudaMalloc((void**)&devLattice, L * L * sizeof(struct triangle));
	cudaMalloc((void**)&devParticles, PN * sizeof(struct coord));
	cudaMalloc((void**)&devResults, 1 * sizeof(float));

	// Init MTRG
	cudaMalloc((void**)&devMTGPStates, BLOCK * sizeof(curandStateMtgp32));
	cudaMalloc((void**)&devKernelParams, sizeof(mtgp32_kernel_params));
	curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, devKernelParams);
	curandMakeMTGP32KernelState(devMTGPStates, mtgp32dc_params_fast_11213,
		devKernelParams, BLOCK, seed);




	
	for (int b = 0; b < linesNumber; b++)
	{
		// Init Lattice and Particles
		initLattice << <BLOCK, THREADS >> > (devMTGPStates, L * L, devLattice, t0, u);
		cudaMemcpy(Lattice, devLattice, L * L * sizeof(struct triangle), cudaMemcpyDeviceToHost);
		for (int i = 0; i < L; i++)
			for (int j = 0; j < L; j++)
				fprintf(D_map, "%f %f ", Lattice[i + j * L].SV, Lattice[i + j * L].SH);

		fprintf(D_map, "\n");

		/*
		// Init Lattice and Particles
		initLattice << <BLOCK, THREADS >> > (devMTGPStates, L * L, devLattice, t0, u);
		initParticles << <BLOCK, THREADS >> > (PN, devParticles, center);

		//step
		t = 0;
		while (true)
		{
			t++;
			cudaMemset(devR, 0, 1 * sizeof(int));
			step << <BLOCK, THREADS >> > (devMTGPStates, devParticles, devLattice, PN, L, center, devR);
			cudaMemcpy(R, devR, 1 * sizeof(int), cudaMemcpyDeviceToHost);
			stepLattice << <BLOCK, THREADS >> > (devMTGPStates, L * L, devLattice);

			if ((t % 10000 == 0) || (*R >= L / 2))
			{
				cudaMemset(devResults, 0, 1 * sizeof(float));
				calculateRSquared << <BLOCK, THREADS >> > (devResults, PN, devParticles, center);
				cudaMemcpy(hostResults, devResults, 1 * sizeof(float), cudaMemcpyDeviceToHost);
				fprintf(D_output, "%g %d ", *hostResults / PN, t);

				//printf("circle %d out of %d, max radius is %d out of %d\n", b, linesNumber, *R, L / 2);
			}
			
			if (t == particle_dist_timing)
			{
				cudaMemcpy(Particles, devParticles, PN * sizeof(struct coord), cudaMemcpyDeviceToHost);
				for (int i = 0; i < PN; i++)
				{
					fprintf(D_particle_dist, "%d %d ", Particles[i].X, Particles[i].Y);
				}
				fprintf(D_particle_dist, "\n");

				printf("particle distribution has been written");

				cudaMemcpy(Lattice, devLattice, L * L * sizeof(struct triangle), cudaMemcpyDeviceToHost);
				for (int i = 0; i < L; i++)
					for (int j = 0; j < L; j++)
						fprintf(D_map, "%g %g %g %g %d %d ", Lattice[i + j * L].CV,
							Lattice[i + j * L].CH, Lattice[i + j * L].AV, Lattice[i + j * L].AH, Lattice[i + j * L].SV, Lattice[i + j * L].SH);

				fprintf(D_map, "\n");

				printf("map has been written");
			}
			

			if ((*R >= L / 2) || (t >= maxT))
				break;
		}
		fprintf(D_output, "\n");*/
	}
	
	/* Cleanup */
	free(Lattice);
	free(Particles);
	free(hostResults);
	free(R);
	cudaFree(devMTGPStates);
	cudaFree(devResults);
	cudaFree(devLattice);
	cudaFree(devParticles);
	cudaFree(devR);

	//fclose(D_output);
	//fclose(D_particle_dist);
	fclose(D_map);
	return 0;
}