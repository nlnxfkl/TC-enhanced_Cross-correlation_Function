#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <mma.h>

using namespace nvcuda;

extern "C" {

__device__ inline void _tc_matmul(
    float *cv,
    const int* av,
    const int* bv)
{
    asm("{\n\t" 
        "wmma.mma.sync.aligned.row.col.m16n16k8.f32.tf32.tf32.f32 {%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9,%10,%11},{%12,%13,%14,%15},{%16,%17,%18,%19,%20,%21,%22,%23};\n\t"
        "}"
        : "=f"(cv[0]),"=f"(cv[1]),"=f"(cv[2]),"=f"(cv[3]),"=f"(cv[4]),"=f"(cv[5]),"=f"(cv[6]),"=f"(cv[7])
        : "r"(av[0]),"r"(av[1]),"r"(av[2]),"r"(av[3])
          "r"(bv[0]),"r"(bv[1]),"r"(bv[2]),"r"(bv[3])
        "f"(cv[0]),"f"(cv[1]),"f"(cv[2]),"f"(cv[3]),"f"(cv[4]),"f"(cv[5]),"f"(cv[6]),"f"(cv[7]));
}

__device__ inline int _tf322int(float a)
{
  float ftmp;
  ftmp = wmma::__float_to_tf32(a);

  return *reinterpret_cast<int*>(&ftmp);
}

__global__ void compdef_gputc_ (
	const int *nt,
	const int *nm,
	const int *nc,
    const float *obs,
	const float *tpl,
	float *co)
{
    int i,j,k;
    int imat,jmat;
    int thid;

    int2 offset_a;
    int offset_b;
    int tempid;
    int idx;

    __shared__ float cm[16*16];
    __shared__ int obs_s[256+NBLOCK*16-1];

    thid = threadIdx.x%32;
    jmat=thid%16;

    idx = blockDim.x*blockIdx.x+threadIdx.x;
    tempid = (threadIdx.x + blockIdx.x * blockDim.x)/32;

    if(tempid*NBLOCK*16>=(*nc)) return;
	
	float dv[NBLOCK][8];
    for(j = 0; j < NBLOCK; j++) {
        dv[j][0] = 0.0f;
        dv[j][1] = 0.0f;
        dv[j][2] = 0.0f;
        dv[j][3] = 0.0f;
        dv[j][4] = 0.0f;
        dv[j][5] = 0.0f;
        dv[j][6] = 0.0f;
        dv[j][7] = 0.0f;
	}

    int N = (*nm)/256;
    int ii;
	for (ii = 0; ii < N; ++ii) {
    	for(i = threadIdx.x; i < 256+NBLOCK*16-1; i += blockDim.x){
      		obs_s[i] = _tf322int(obs[blockIdx.x*NBLOCK*16+i + ii*256]);
    	}

    int av_left[4], av_right[4];
    int bv_left[4], bv_right[4];
#pragma unroll
    for (imat = 0; imat < 16; imat++) {
	    // matrix A
	    offset_a.x = (imat*16) + (thid/4)*(*nm)     + (thid % 4) + ii*256;
      	offset_a.y = (imat*16) + (thid/4 + 8)*(*nm) + (thid % 4) + ii*256;
      	av_left[0]   = _tf322int(tpl[offset_a.x]);
      	av_left[1]   = _tf322int(tpl[offset_a.y]);
      	av_left[2]   = _tf322int(tpl[offset_a.x + 4]);
      	av_left[3]   = _tf322int(tpl[offset_a.y + 4]);
      	av_right[0]  = _tf322int(tpl[offset_a.x     + 8]);
      	av_right[1]  = _tf322int(tpl[offset_a.y     + 8]);
      	av_right[2]  = _tf322int(tpl[offset_a.x + 4 + 8]);
      	av_right[3]  = _tf322int(tpl[offset_a.y + 4 + 8]);
		
		for (j = 0; j < NBLOCK; j++) {
		    // matrix B
            offset_b = (imat*16) + (j*16) + (thid/4) + (thid % 4);
        	bv_left[0]  = obs_s[offset_b      + 0];
        	bv_left[1]  = obs_s[offset_b      + 4];
        	bv_left[2]  = obs_s[offset_b      + 8];
        	bv_left[3]  = obs_s[offset_b      + 12];
        	bv_right[0] = obs_s[offset_b +  8 + 0];
        	bv_right[1] = obs_s[offset_b +  8 + 4];
        	bv_right[2] = obs_s[offset_b +  8 + 8];
        	bv_right[3] = obs_s[offset_b +  8 + 12];

		    // matrix-matrix product
		    _tc_matmul(dv[j], av_left, bv_left);
		    _tc_matmul(dv[j], av_right, bv_right);
      } //j
    } //imat
} //ii

	for (j = 0; j < NBLOCK; j++) {
  	asm("{\n\t"
        ".reg .u32 r<1>;\n\t"
        "mov.u32 r0, _ZZ14compdef_gputc_E2cm;\n\t"
        "wmma.store.d.aligned.sync.row.m16n16k16.shared.f32 [r0], {%0,%1,%2,%3,%4,%5,%6,%7};\n\t"
        "}"
        :: "f"(dv[j][0]),"f"(dv[j][1]),"f"(dv[j][2]),"f"(dv[j][3]),"f"(dv[j][4]),"f"(dv[j][5]),"f"(dv[j][6]),"f"(dv[j][7]));

    	for (i = thid; i < 256; i += 32) {
      		k = i/16;
      		co[(tempid*NBLOCK+j)*16+jmat+(*nt)*k] = cm[i];
    	}
    } //j

    return;
}

}
