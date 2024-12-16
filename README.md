TC-enhanced Cross-correlation Function
====

TC-enhanced Cross-correlation Function is a fast calculation of cross-correlation function using TF32 Tensor Core operations on NVIDIA Ampere and Hopper GPU.

## Description
The cross-correlation function can be calculated as a matrix-matrix product, and a significant speed-up can be expected utilizing Tensor Core, which is a matrix-matrix product acceleration unit of the latest NVIDIA GPUs. We target a new precision data type called the TensorFloat-32, which is available in the Ampere and newer architecture. We develop a fast calculation method considering the characteristics of the cross-correlation function and TensorCore. Our method achieved a very high performance in seismic interferometry using actual data. In addition, the accuracy of the calculation result is sufficiently high compared to the 64-bit floating-point calculation, indicating the applicability of Tensor Core operations using TensorFloat-32 for scientific calculations.

## Requirement
* NVIDIA Ampere GPU or Hopper GPU
* CUDA + nvcc compiler *(We use cuda 12.6. It is not guaranteed to run in other versions.)*
* nvfortran *(We use nvfortran 24.11-0. It is not guaranteed to run in other versions.)*

## Usage
### Setting of input files
* The number of time steps in observation data (referred to as matrix B in the paper) and the number of templates are defined in `./data/parameter_setting.dat`.
* In this implementation, the number of timesteps of observation data must be a multiple of 256.
* In this implementation, the number of template waves of must be less than 16.
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
"./data/observation/observation.dat"
```

* Filenames of template waves (referred to as matrix A in the paper) are defined in `./data/tpl_file.dat`.

```
$ head ./data/tpl_file.dat
!filename for template data
"./data/template/template.0001.dat"
"./data/template/template.0002.dat"
"./data/template/template.0003.dat"
"./data/template/template.0004.dat"
"./data/template/template.0005.dat"
"./data/template/template.0006.dat"
"./data/template/template.0007.dat"
"./data/template/template.0008.dat"
"./data/template/template.0009.dat"
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

maximum absolute error to FP64:    5.9382377881500581E-005
computation completed.
```
We obtained these results by using NVIDIA Tesla H100 PCIe 80GB GPU with CUDA version 12.6 and nvfortran version 24.11-0.

## Test data
Data in `./data/observation` and `./data/template` are dummy data and generated from a uniform pseudorandom number with the interval [-50:50].

## Publication
Yuma Kikuchi, Kohei Fujita, Tsuyoshi Ichimura, Muneo Hori, and Lalith Maddegedara, “Calculation of Cross-correlation Function Accelerated by Tensor Cores with TensorFloat-32 precision on Ampere GPU”, Proceedings of 11th International Workshop on Advances in High-Performance Computational Earth Sciences: Applications & Frameworks, 2022. (accepted)

Kohei Fujita, Takuma Yamaguchi, Yuma Kikuchi, Tsuyoshi Ichimura, Muneo Hori, Lalith Maddegedara, "Calculation of cross-correlation function accelerated by TensorFloat-32 Tensor Core operations on NVIDIA’s Ampere and Hopper GPUs", Journal of Computational Science, 2023. 

## Licence
For Licence, please refer to the file `LICENSE` in this repository.

## References
* Yamaguchi,T.,Ichimura,T.,Fujita,K.,Kato,A.,Nakagawa,S.:MatchedFiltering Accelerated by Tensor Cores on Volta GPUs With Improved Accuracy Using Half- Precision Variables, http://dx.doi.org/10.1109/LSP.2019.2951305, (2019). https:/ /doi.org/10.1109/lsp.2019.2951305.
