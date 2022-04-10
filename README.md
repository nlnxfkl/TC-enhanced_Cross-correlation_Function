TC-enhanced Cross-correlation Function
====

TC-enhanced Cross-correlation Function is a fast calculation of cross-correlation function using TF32 Tensor Core operations on NVIDIA Ampere GPU.

## Description
The cross-correlation function can be calculated as a matrix-matrix product, and a significant speed-up can be expected utilizing Tensor Core, which is a matrix-matrix product acceleration unit of the latest NVIDIA GPUs. We target a new precision data type called the TensorFloat-32, which is available in the Ampere architecture. We develop a fast calculation method considering the characteristics of the cross-correlation function and TensorCore. Our method achieved a very high performance of 53.56 TFLOPS in the performance measurement assuming seismic interferometry using actual data, which is 5.97 times faster than cuBLAS, a widely used linear algebra library on NVIDIA GPUs. In addition, the accuracy of the calculation result is sufficiently high compared to the 64-bit floating-point calculation, indicating the applicability of Tensor Core operations using TensorFloat-32 for scientific calculations.

## Requirement
* NVIDIA Ampere GPU with Tensor Core
* CUDA + nvcc compiler *(We use cuda 11.2. It is not guaranteed to run in other versions.)*
* nvfortran *(We use nvfortran 21.3-0. It is not guaranteed to run in other versions.)*

## Usage
### Setting of input files
* The number of time steps in observation data (referred to as matrix B in the paper) and the number of templates are defined in `./data/parameter_setting.dat`.
* In our implementation, the number of timesteps of observation data has to be a multiple of 256.
```
$ cat ./data/paramter_setting.dat
the number of timesteps of observation data
1024
the number of template waves
16
```

* Filename of observation wave (referred to as matrix B in the paper) is defined in `./data/obs_file.dat`.

```
$ cat ./data/obs_file.dat
!filename for observation data
"../data/observation/observe.dat"
```

* Filenames of template waves (referred to as matrix A in the paper) are defined in `./data/tpl_file.dat`.

```
$ head ./data/tpl_file.dat
!filename for template data
"../data/template/template.0001.dat"
"../data/template/template.0002.dat"
"../data/template/template.0003.dat"
"../data/template/template.0004.dat"
"../data/template/template.0005.dat"
"../data/template/template.0006.dat"
"../data/template/template.0007.dat"
"../data/template/template.0008.dat"
"../data/template/template.0009.dat"
```

* Observation data must be stored in the file specified in `./data/obs_file.dat`.  
By default, the format must be ASCII and actual data must be in the 2nd column of each line.

```
$ head ./data/observation/observe.dat
  0.00000000 41.15180778
  0.00500000 22.07329469
  0.01000000  5.03670761
  0.01500000 18.82923041
  0.02000000 16.13020223
  0.02500000 49.64038604
  0.03000000  0.76104837
  0.03500000 32.03866291
  0.04000000 40.56912162
  0.04500000 43.27840729
```

* Template data must be stored in the files specified in `./data/tpl_file.dat`. The format must be the same as that of observation data by default.  *The number of components in each template data cannot be more than 256 for this version.*


### Compile
Modify `GPU_VER1` and `GPU_VER2` in `Makefile` so that specified versions are consistent with actual ones.  
Also, you can compare the results of Tensor Core operation and CPU computation in double precision by adding `-DVERIFICATION` flag.
```
$ head Makefile
FC = nvfortran
NVCC = nvcc

GPU_VER1 = -Mcuda=cuda11.2 -ta=tesla:cc80,loadcache:L1

...

BASEFLAGS += -DVERIFICATION
```
 Now, you can simply compile by  
```
$ make
```  

### Run
You can run the program by simply `./a.out`.  
If all goes well, you can get standard output as below:  
```

...

maximum absolute error to FP64:    5.1826030157187120E-005
computation completed.
Computation time on CPU : 0.4510880E-03 sec
Computation time on GPU : 0.3099442E-04 sec
```
We obtained these results by using NVIDIA Tesla A100 PCIe 40GB GPU with CUDA version 11.2 and nvfortran version 21.3-0.

## Test data
Data in `./data/observation` and `./data/template` are dummy data and generated from a uniform pseudorandom number with the interval [-50:50].

## Publication
Yuma Kikuchi, Kohei Fujita, Tsuyoshi Ichimura, Muneo Hori, and Lalith Maddegedara, “Calculation of Cross-correlation Function Accelerated by Tensor Cores with TensorFloat-32 precision on Ampere GPU”, Proceedings of 11th International Workshop on Advances in High-Performance Computational Earth Sciences: Applications & Frameworks, 2022. (accepted)

## Licence
TC-enhanced Cross-correlation Function, version 1.0.0 (c) 2022 Yuma Kikuchi et al.
TC-enhanced Cross-correlation Function is freely distributable under the terms of an MIT-style license.

## References
* Yamaguchi,T.,Ichimura,T.,Fujita,K.,Kato,A.,Nakagawa,S.:MatchedFiltering Accelerated by Tensor Cores on Volta GPUs With Improved Accuracy Using Half- Precision Variables, http://dx.doi.org/10.1109/LSP.2019.2951305, (2019). https:/ /doi.org/10.1109/lsp.2019.2951305.
