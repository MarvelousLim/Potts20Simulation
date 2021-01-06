#include <stdio.h>
#include <random>
#include <iostream>

#define NNEIBORS 4 // number of nearest neighbors, is 4 for 2d lattice

using namespace std;

struct neibors_indexes {
	int up;
	int down;
	int left;
	int right;
};

struct neibors_indexes SLF(int j, int L, int N) {
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

struct neibors get_neibors_values(char* s, struct neibors_indexes n_i, int replica_shift) {
	struct neibors result = { s[n_i.up + replica_shift], s[n_i.down + replica_shift], s[n_i.left + replica_shift], s[n_i.right + replica_shift] };
	return result;
}

int LocalE(char currentSpin, struct neibors n) { 	// Computes energy of spin i with neighbors a, b, c, d 
	return -(currentSpin == n.up) - (currentSpin == n.down) - (currentSpin == n.left) - (currentSpin == n.right);
}

int DeltaE(char currentSpin, char suggestedSpin, struct neibors n) { // Delta of local energy while i -> e switch
	return LocalE(suggestedSpin, n) - LocalE(currentSpin, n);
}

int calcEnergy(char* s, int L, int N) {
	int sum = 0;
	for (int j = 0; j < N; j++) {
		// 0.5 by doubling the summarize
		int replica_shift = 0;
		char i = s[j + replica_shift]; // current spin value
		struct neibors_indexes n_i = SLF(j, L, N);
		struct neibors n = get_neibors_values(s, n_i, replica_shift); // we look into r replica and j spin
		sum += LocalE(i, n);
	}
	return sum / 2;
}

void initializePopulation(char* s, int N, int q, mt19937* gen) {
	uniform_int_distribution<> spin(0, q - 1);

	for (int k = 0; k < N; k++) {
		//char spin = curand(&state[r]) % q;
		s[k] = spin(*gen); // spin;
	}
}

void equilibrate(char* s, int* E, int L, int N, int q, int nSteps, int* H, double* lng, double f, mt19937* gen, FILE* efile) {
	uniform_int_distribution<> spin(0, q - 1);
	uniform_int_distribution<> site(0, N - 1);
	uniform_real_distribution<> real(0, 1);

	int replica_shift = 0;
	for (int _ = 0; _ < nSteps; _++) {
		for (int k = 0; k < N; k++) {
			int j = site(*gen);
			char currentSpin = s[j + replica_shift];
			char suggestedSpin = spin(*gen);
			struct neibors_indexes n_i = SLF(j, L, N);
			struct neibors n = get_neibors_values(s, n_i, replica_shift);
			int dE = DeltaE(currentSpin, suggestedSpin, n);
			double r = real(*gen);
			if (r < exp(lng[-*E] - lng[-*E - dE])) {
				*E = *E + dE;
				s[j + replica_shift] = suggestedSpin;
			}
			lng[-*E] += f;
			H[-*E]++;
		}
	}
}

double HStats(int* H, int N) {
	double preavg = 0, min = H[0];
	for (int i = 0; i <= 2 * N; i++) {
		preavg += 1.0 * H[i] / (2 * N + 1);
		if (H[i])
			min = (min < H[i] ? min : H[i]);
	}
	return 100.0 * min / preavg;
	//printf("avg H: %f; min H: %d; percentile: %f \n", avg, min, 100.0 * min / avg);
}

int main(int argc, char* argv[]) {
// 	int nSteps = 1;
	int q = 20;	// q parameter for potts model, each spin variable can take on values 0 - q-1

	int run_number = atoi(argv[1]);
	int seed = run_number;
	int L = atoi(argv[2]);
	int N = L * L;
	//double beta = log(1 + sqrt(q));

	char s[100];
	sprintf(s, "datasets//fugao_L%d_run%de.txt", L,  run_number);
	FILE* efile = fopen(s, "w");
	sprintf(s, "datasets//fugao_L%d_run%dlng.txt", L, run_number);
	FILE* lngfile = fopen(s, "w");

	size_t fullLatticeByteSize = 1 * N * sizeof(char);
	char* S = (char*)malloc(fullLatticeByteSize);
	int fullEnergyRange = 1 + 2 * N;
	int* H = (int*)malloc(fullEnergyRange * sizeof(int)); // energy histogram
	double* lng = (double*)malloc(fullEnergyRange * sizeof(double)); // log density of states
	for (int i = 0; i <= 2 * N; i++) {
		lng[i] = 1.0;
		H[i] = 0;
	}
	mt19937 gen(seed);
	int E;

	initializePopulation(S, N, q, &gen);
	E = calcEnergy(S, L, N);

	double f = 1.0;
	int counter = 0;
	while (f > 0.000000001) {
		// control
		int steps = 100000;
		// calc
		equilibrate(S, &E, L, N, q, steps, H, lng, f, &gen, efile);

		// results check
		counter += steps;

		// write results
		fprintf(efile, "%d ", counter);
		fprintf(lngfile, "%d ", counter);
		for (int i = 0; i <= 2 * N; i++) {
			fprintf(efile, "%d ", H[i]);
			fprintf(lngfile, "%lf ", lng[i]);
		}
		fprintf(efile, "\n");
		fprintf(lngfile, "\n");
		fflush(efile);
		fflush(lngfile);

		// reset
		double percentile = HStats(H, N);
		if (percentile > 80.0) { // reset
			f /= 2;
			for (int i = 0; i <= 2 * N; i++)
				H[i] = 0;
		}
		printf("steps %d; percentile %lf\n", counter, percentile);
		fflush(stdout);
	}

	free(S);
	fclose(efile);	
	return 0;
}
