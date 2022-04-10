FC = nvfortran
NVCC = nvcc

GPU_VER1 = -Mcuda=cuda11.2 -ta=tesla:cc80,loadcache:L1
GPU_VER2 = arch=compute_80,code=sm_80

BASEFLAGS = -DNBLOCK=6 -DNTENSOR=16
BASEFLAGS += -DVERIFICATION

OBJS = \
	compdef_gpu.o \
	precision.o \
	cublas_wrapper.o \
	main.o

FFLAGS = -fastsse -O3 $(GPU_VER1) $(BASEFLAGS)
NVCCFLAGS = --generate-code $(GPU_VER2) -O3 --use_fast_math -lineinfo -Xptxas="-v" $(BASEFLAGS)

PROGRAM = a.out

.SUFFIXES: $(SUFFIXES) .F90 .cu

all: $(PROGRAM)
	
.cu.o: 
	$(NVCC) $(NVCCFLAGS) -c $<
.F.o:
	$(FC) $(FFLAGS) -c $<

$(PROGRAM):$(OBJS)
	$(FC) $(FFLAGS) $(OBJS)  -o $@

clean: 
	rm -f *.o *.ptx *~ *.mod $(PROGRAM)
