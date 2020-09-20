#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <queue>
#include <algorithm>
#include <list>
#include <gsl/gsl_rng.h>
#include "cuda_runtime.h"
#include <curand.h>
/*-------------------------------------------------------------------------------------------------
    Microcanonical Population Annealing
    Written by Nathan Rose

    Code requires gpu library CUDA.  I have only tried version 8.0.61, so changes may be required
    to use newer versions.  CPU portions of the code also use scientific computing library GSL
    for random number generation (RNG).  The C++ standard RNG may be used instead without affecting
    performance.

    There are two portions of the code:
        - GPU portion includes Monte Carlo updates, the most computationally heavy part of the
         algorithm
        - CPU portion incudes resampling step and measuring and saving observables.
    
    Output of the program is saved in several .txt files, labelled with a letter indicating what
    it contains and a number that is given as an argument to the program.  For instance, culling
    fraction at each energy step is outuptted to p1X.txt

    Compilation is done through the Cuda compiler nvcc, for example:
        nvcc mcpa.cu -std=c++11 -lgsl -lgslcblas -lm 
    where extra flags are passed as compiler and linker arguments for the regular GCC compiler.

    The executable takes three arguments:
        - run_number:  A number used to identify a particular run of the algorithm
        - grid_width:  GPU parameter that determines how many threads to use.  The population size
                       is equal to grid_width times 32.
        - L:           Lattice side length, so the number of spins is L squared.
--------------------------------------------------------------------------------------------------*/

using namespace std;
double t1=time(0);
// number of nearest neighbors, is 4 for 2d lattice
#define nneighbs 4
// gpu block parameter, should be a small multiple of 32
#define block_width 32
// Parameters:
float nSteps=1;
// q parameter for potts model, each spin variable can take on values 0 - q-1
int q=20;
// U is energy ceiling
int U=1;
// random number generation
const gsl_rng_type * T;
gsl_rng * r;
// output files
// average energy
ofstream efile;
// surface (culled) energy
ofstream e2file;
// culling fraction
ofstream Xfile;
// rho t
ofstream ptfile;
// number of sweeps
ofstream nfile;
// cluster size histogram
ofstream chfile;
// Set up output files for data collection and rng
void setup(int run_number){
    stringstream s;
    s<<"p"<<run_number;
    string S=s.str();
    efile.open(S+"e.txt");
    e2file.open(S+"e2.txt");
    Xfile.open(S+"X.txt");
    ptfile.open(S+"pt.txt");
    nfile.open(S+"n.txt");
    chfile.open(S+"ch.txt");
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);
    int seed=run_number*77;
    gsl_rng_set(r,seed);
    cout<<"seed: "<<" ";
    cout<<seed<<"\n";
}

void printAvgEng(int* E, int U, int R){
    /*---------------------------------------------------------------------------------------------
        Writes average energy to output file

        Args:
            E           array of energies
            U           energy ceiling value
            R           population size
    --------------------------------------------------------------------------------------------*/
    double avg = 0;
    // this is to convert F to double   
    for (int i = 0; i < R; i++) {
        avg += (double)E[i];
    }
    avg /= ((double)R);
    efile << U << " ";
    efile << avg << "\n";
}

__device__ int del(char i, char a,char b,char c,char d, int e){
    // Computes energy delta from flipping spin at site i to value e with neighbors a,b,c,d 
    int sum=0;
    if(i==a){
	sum++;
	}
    if(i==b){
	sum++;
	}
    if(i==c){
	sum++;
	}
    if(i==d){
	sum++;
	}
    if(e==a){
	sum--;
	}
    if(e==b){
	sum--;
	}
    if(e==c){
	sum--;
	}
    if(e==d){
	sum--;
	}
return sum;
}

__host__ int h_del(int i, int a,int b,int c,int d, int q){
    int sum=0;
    if(i==a){
	sum++;
	}
    if(i==b){
	sum++;
	}
    if(i==c){
	sum++;
	}
    if(i==d){
	sum++;
	}
    if(q==a){
	sum--;
	}
    if(q==b){
	sum--;
	}
    if(q==c){
	sum--;
	}
    if(q==d){
	sum--;
	}
return sum;
}

// Initializes spin lookup table for efficiency
void initializeSLT(int* SLT,int N,int L){
    int next0;
    int next1;
    int next2;
    int next3;
    int d=4;
    for(int i=0;i<N;i++){
        next0=(i+1)%L+L*(i/L);
        next1=(i+L-1)%L+L*(i/L);
        next2=(i+L)%(L*L)+(L*L)*(i/(L*L));
        next3=(i+(L*L)-L)%(L*L)+(L*L)*(i/(L*L));
        SLT[d*i]=next0;
        SLT[d*i+1]=next1;
        SLT[d*i+2]=next2;
        SLT[d*i+3]=next3;
      }
}

// initializes random number generator on gpu
__global__ void randomSetup(unsigned int seed, curandState_t* states,int R) {

  /* we have to initialize the state */
  if(blockIdx.x*block_width+threadIdx.x<R){
	curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
              blockIdx.x*block_width+threadIdx.x, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &states[blockIdx.x*block_width+threadIdx.x]);
  }else{
	curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
              blockIdx.x, /* the sequence number should be different for each core (unless you want all
                             corblock_widthes to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &states[blockIdx.x*+threadIdx.x]);
	}
}

__host__ void hostEnergy(char* s,int* SLT,int* E,int R,int N, int q){
    /*---------------------------------------------------------------------------------------------
        Calculates energy on the cpu

        Args:
            s               array of all spins in spin-replica order
            SLT             fixed spin lookup table
            E               array of energies to set
            R               fixed population size
            N               fixed number of spins
            q               Potts model parameter

    -----------------------------------------------------------------------------------------------*/
    for(int RN=0;RN<R;RN++){
        float sum=0;
        for(int i=0;i<N;i++){
            sum=sum-.5*h_del(s[RN+R*i],s[SLT[4*i]*R+RN],s[SLT[4*i+1]*R+RN],s[SLT[4*i+2]*R+RN],s[SLT[4*i+3]*R+RN],q+1);	   
        }
        E[RN]=sum;
    }
}

__host__ void initializePopulation(char* s,int* SLT,int* E, int* fam, int R, int N, int q){
    /*---------------------------------------------------------------------------------------------
        Initializes population on cpu by randomly assigning each spin a value from 0 to q-1
    ----------------------------------------------------------------------------------------------*/
    for(int i=0;i<R;i++){
        for(int j=0;j<N;j++){
            int r1=gsl_rng_uniform(r)*q;
            s[i*N+j]=r1;
        }
        // assign each replica a distinct family number
	    fam[i]=i;
    }
    // set energy of each replica
    hostEnergy(s,SLT,E,R,N,q);
}

__global__ void equilibrate(char* s, int* E, int* SLT,int N, int R, float nSteps,int U,int q, curandState_t* states){
    /*---------------------------------------------------------------------------------------------
        Main Microcanonical Monte Carlo loop.  Performs update sweeps on each replica in the 
        population

        Args:
            s                   Array of spins[char] in spin-replica order
            E                   Array of energies[int] for each replica
            SLT                 fixed spin lookup table
            N                   fixed number of spins per replica
            R                   fixed population size
            nSteps              number of monte carlo Sweeps
            U                   energy ceiling value
            q                   Potts model parameter
            states              rng device
    ---------------------------------------------------------------------------------------------*/
    int RN= blockIdx.x*block_width+threadIdx.x;
    for(int k=0;k<(int)(N*nSteps);k++){
        int i=curand(&states[R+RN])%N;
        int j=curand(&states[RN])%q;
        int dE=del(s[RN+R*i],s[SLT[4*i]*R+RN],s[SLT[4*i+1]*R+RN],s[SLT[4*i+2]*R+RN],s[SLT[4*i+3]*R+RN],j);
        if(E[RN]+dE<U){
            E[RN]=E[RN]+dE;
            s[RN+R*i]=j;
        }		
    }
}


__global__ void updateReplicas(char* s,int* E, int* upd,int N,int R){
    /*---------------------------------------------------------------------------------------------
        Updates the population after the resampling step (done on cpu) by replacing indicated 
        replicas by the proper other replica

        Args:
            s                   Array of spins
            E                   Array of energies per replica
            upd                 index of new replica to replace with
            N                   fixed number of spins
            R                   fixed population size
    -----------------------------------------------------------------------------------------------*/
    int RN=blockIdx.x*block_width+threadIdx.x;
    if(upd[RN]!=RN){
        E[RN]=E[upd[RN]];
        for(int i=0;i<N;i++){
            s[RN+i*R]=s[upd[RN]+i*R];
        }
    }
}

__host__ void rho_t(int* fam,int R){
    /*---------------------------------------------------------------------------------------------
        Rho t is a measure of how well equilibrated the population is.  A suitable condition is that
        rho_t << R.

        Args:
            fam                 array of family number for each replica
            R                   fixed population size
    -----------------------------------------------------------------------------------------------*/
    // histogram of family sizes
    std::vector<int> famSize(R);
    for(int i=0;i<R;i++){
	famSize[fam[i]]++;
	}
    double sum=0;
    for(int i=0;i<R;i++){
	    sum += pow(famSize[i],2);
    }
    ptfile<<U<<" ";
    ptfile<<sum/((double)R)<<"\n";	
}

void Swap(int* A, int i, int j){
    int temp = A[i];
    A[i] = A[j];
    A[j] = temp;
}

void quicksort(int* E,int* O,int left, int right){
    int Min = (left+right)/2;
    int i = left;
    int j = right;
    double pivot = E[O[Min]];

    while(left<j || i<right)
    {
        while(E[O[i]]>pivot)
        i++;
        while(E[O[j]]<pivot)
        j--;

        if(i<=j){
            Swap(O,i,j);
            i++;
            j--;
        }
        else{
            if(left<j)
                quicksort(E,O,left, j);
            if(i<right)
                quicksort(E,O,i,right);
            return;
        }
    }
}

__host__ void resample(int* E, int* O,int* upd,int* fam,int R){
    quicksort(E,O,0,R-1);//Sorts O by energy
    int nCull=0;
    e2file<<U<<" ";
    e2file<<E[O[0]]<<"\n";
    while(E[O[nCull]]==U-1){
        nCull++;
        if(nCull==R){
            break;
        }
    }
    // culling fraction
    double X=(((double)nCull)/((double)R));
    Xfile<<U<<" ";
    Xfile<<X<<"\n";
    U--;
    int r1;
    for(int i=0;i<R;i++){
        upd[i]=i;
    }
    if(nCull<R){
        for(int i=0;i<nCull;i++){
            // random selection of unculled replica
            r1=(int)(gsl_rng_uniform(r)*(R-nCull))+nCull;
            // mark position to be updated on gpu
	        upd[O[i]]=O[r1];
            fam[O[i]]=fam[O[r1]];
    	}
    }
}

// Part of disordered cluster histogram algorithm
int BFS(int* SLT, std::vector<int>& taboo, std::vector<int>& visited, int start){
    std::queue<int,std::list<int> > BFS_queue;
    BFS_queue.push(start);
    visited[start]=1;
    int pos=0;
    int C_size=0;
    while(!BFS_queue.empty()){
        pos=BFS_queue.front();
	    BFS_queue.pop();
        C_size++;
        for(int i=0;i<4;i++){
            if( (!taboo[SLT[4*pos+i]]) && (!visited[SLT[4*pos+i]]) ){
                BFS_queue.push(SLT[4*pos+i]);
                visited[SLT[4*pos+i]]=1;
            }
        }
    }
    return C_size;
}

void makeClusterHistogram(char* s, int* SLT, int* E, int q, int N, int L, int R, int numHists, int U){
     /*------------------------------------------------------------------------------------------------
	Disordered Cluster Histogram Algorithm:
	
	Args:
		s		Array holding spin values in spin-replica order
		SLT		Fixed Spin Lookup Table
		E		Array containing energy of each replica
		q		Fixed Potts model parameter
		N		Fixed number of spins per replica
		L		Fixed lattice side length
		R		Fixed Population size
		numHists	Maximum number of histograms to create
		U		Current Energy Ceiling value

	Steps to procedure:
		- Identify replicas with energy one less than the energy ceiling
		- Determine the Majority (ordered) cluster
		- Identify clusters of spins not in the majority phase.  For low
		  enough energies, these should be isolated clusters.
		- compile histogram of cluster sizes 
     -------------------------------------------------------------------------------------------------*/	
     //s stores R replicas in spin-replica order
    std::vector<int> C_hist(N);
    int R_count=0;
    for(int RN=0; RN<R; RN++){
        if(E[RN]==U-1 && R_count<numHists){
            R_count++;
            //find the max value from 0 to q-1 of spins
            std::vector <int> mgnCount(q);
            for(int i=0;i<N;i++){
                mgnCount[s[RN+R*i]]++;
            }
            int arg_M = std::max_element(mgnCount.begin(), mgnCount.end()) - mgnCount.begin(); 
            //determine a good starting point for the base cluster search
            int arg_base=0;
            int maxBaseCluster=0;
            std::vector<int> V1(N);
            std::vector<int> V2(N);   
            for(int i=0;i<N;i++){
                if(s[RN+R*i]!=arg_M){
                    V1[i]=1;
                }
            }
            for(int i=0;i<N;i++){
                if(s[RN+R*i]==arg_M && !V1[i]){
                    int clusterSize=BFS(SLT,V1, V2, i);
                    for(int j=0;j<N;j++){
                        V1[j]+=V2[j];
                        V2[j]=0;
                    }
                    if(clusterSize>maxBaseCluster){
                        maxBaseCluster=clusterSize;
                        arg_base=i;
                    }
                }
            }
            //do the base cluster search
            //reset up V1, V2
            for(int i=0;i<N;i++){
                if(s[RN+R*i]!=arg_M){
                    V1[i]=1;
                }else{
                    V1[i]=0;
                }
                V2[i]=0;
            }
            int backgroundC=BFS(SLT,V1,V2,arg_base);
            //set up V1 and V2 for next section
            //the background cluster becomes V1, V2 is cleared
            V1 = V2;
            std::fill(V2.begin(), V2.end(), 0);
            //Go sequentially through lattice and assign spins to clusters
            for(int i=0;i<N;i++){
                if(!V1[i]){
                    int C_size = BFS(SLT, V1, V2, i);
                    C_hist[C_size]++;
                    auto it1 = V1.begin();
                    for(auto it2=V2.begin(); it2 != V2.end(); ++it2){
                        *it1 += *it2;
                        *it2 = 0;
                        ++it1;
                    }
                }
            }
        }//end if E[RN] == U-1 
    }//end RN for loop
    // write results to output files
    for(int i=0;i<N;i++){
        chfile<<i<<" ";
        chfile<<((double)C_hist[i]) / R_count<<"\n";
    }
}


int main(int argc, char *argv[]){
    // Just a number to label this run of the algorithm, used for data keeping purposes
    // Also, the random number generator is seeded with this, so be careful
    int run_number=atoi(argv[1]);
    // Gpu setup
    // number of replicas is grid_width times block_width (32)
    int grid_width=atoi(argv[2]);
    //Lattice size.  Total number of spins is L squared
    int L=atoi(argv[3]);	
    int N = L*L;
    // Population size, must be a multiple of block_width (32)
    int R=grid_width*block_width;
    setup(run_number);	
    // C style arrays for gpu
    // each array has a host copy h_ and a device copy d_ 
    int* h_SLT;
    char* h_s;
    int* h_E;
    int* fam;
    int* O;
    int* h_upd;
    // device
    char* d_s;
    int* d_E;
    int* d_upd;
    int* d_SLT;
    // total number of spins: N spins per replica times population size R
    size_t RNsize=R*N*sizeof(char);
    dim3 dimGrid(grid_width);
    dim3 dimBlock(block_width);
    // Spin Lookup Table (SLT) for efficiency
    h_SLT=(int*)malloc(nneighbs*N*sizeof(int));
    initializeSLT(h_SLT,N,L);
    // array of all spins, stored in spin-replica order
    h_s=(char*)malloc(RNsize);
    cudaMallocHost((void**)&h_E,R*sizeof(int));
    // family of each replica, used to measure rho t
    fam=(int*)malloc(R*sizeof(int));
    // ordering array, used during resampling step of algorithm
    O=(int*)malloc(R*sizeof(int));
    for(int i=0;i<R;i++){
	O[i]=i;
    }
    // allocate memory
    cudaMallocHost((void**)&h_upd,R*sizeof(int));
    cudaMalloc((void**)&d_s,RNsize);
    cudaMalloc((void**)&d_E,R*sizeof(int));
    cudaMalloc((void**)&d_upd,R*sizeof(int));
    cudaMalloc((void**)&d_SLT,N*nneighbs*sizeof(int));
    // initialize population on host
    initializePopulation(h_s,h_SLT,h_E, fam,R,N,q);
    curandState_t* states;
    cudaMalloc((void**)&states,2*R*sizeof(curandState_t));
    // setup gpu random number generator
    randomSetup<<<2*grid_width, block_width>>>(run_number*77, states,R);
    // copy arrays to device
    cudaMemcpy(d_SLT,h_SLT,N*nneighbs*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_s,h_s,RNsize,cudaMemcpyHostToDevice);
    cudaMemcpy(d_E,h_E,R*sizeof(int),cudaMemcpyHostToDevice);

    int count=0;
    while(U>-2*N){
        count++;
        // Adjust the sweep schedule
        // Most sweeps are performed in the region when simulation is most difficult
        if(U>=-N/2){
            nSteps=2;
        } 
        if(U<-N/2){
            nSteps = 30;
        }
        if(U<-(3*N/2)){
            nSteps= 10;
        }
        nfile<<U<<" ";
        nfile<<nSteps<<"\n";
        // Perform monte carlo sweeps on gpu
        equilibrate<<<dimGrid,dimBlock>>>(d_s, d_E, d_SLT, N, R, nSteps, U, q, states);        	
        cudaThreadSynchronize();
        // Copy energy and spin configuration back to host
        cudaMemcpy(h_E,d_E,R*sizeof(int),cudaMemcpyDeviceToHost);
        cudaMemcpy(h_s, d_s, RNsize, cudaMemcpyDeviceToHost);
        int numHist = 100000;
        // Create disordered cluster size histogram in particular energy range
        if(U<=-1.5*N && U > -2*N ){
            makeClusterHistogram( h_s, h_SLT, h_E, q, N,L, R, numHist, U);	
        }
        // record average energy and rho t
        printAvgEng(h_E, U, R);
        rho_t(fam,R);
        // perform resampling step on cpu
        resample(h_E,O,h_upd,fam,R);
        // copy list of replicas to update back to gpu
        cudaMemcpy(d_upd,h_upd,R*sizeof(int),cudaMemcpyHostToDevice);
        updateReplicas<<<grid_width,block_width>>>(d_s,d_E, d_upd,N,R);//update culled Replicas    
        cudaThreadSynchronize();
	}
	printAvgEng(h_E, U, R);
    cout<<"number: "<<" ";
    cout<<run_number<<"\n";
    cout<<"q: "<<" ";
    cout<<q<<"\n";
    cout<<"seed: "<<"\n";
    cout<<77*run_number<<"\n";
    cout<<"R: "<<" ";
    cout<<R<<"\n";
    cout<<"N: "<<" ";
    cout<<N<<"\n";
    cout<<"Fast rng scheme"<<"\n";
    cout<<"Adaptive SWEEP SCEHDULE nsteps="<<" ";
}



