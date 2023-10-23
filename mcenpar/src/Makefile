# Compilers
MPICXX= mpicxx
NVCC= nvcc

# Flags
CFLAGS=
MPICFLAGS=
GENCODE_SM35=
NVCCFLAGS=$(GENCODE_SM35)
CUDALDFLAGS=-L/usr/local/cuda/lib64 -lcudart

# Binaries
BINDIR=../bin
MCEN_NORMAL_MPI_INDEP=$(BINDIR)/mcenpar
BINARIES=$(MCEN_NORMAL_MPI_INDEP)

all: $(BINARIES)

minepardev.o: minepar.h minepar.cu Makefile
	$(NVCC) $(MPICFLAGS) $(NVCCFLAGS) -o minepardev.o -c minepar.cu

mineparhost.o: minepar.h minepar.cpp Makefile
	$(MPICXX) -o mineparhost.o -c minepar.cpp

$(MCEN_NORMAL_MPI_INDEP): minepardev.o mineparhost.o Makefile
	mkdir -p $(BINDIR)
	$(MPICXX) $(CUDALDFLAGS) -o $(MCEN_NORMAL_MPI_INDEP) minepardev.o mineparhost.o

clean:
	rm -rf *.o *~ $(BINARIES)
