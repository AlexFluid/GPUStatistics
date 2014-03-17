
#ifndef STATISTICS_KERNEL_CU_
#define STATISTICS_KERNEL_CU_

#include "statistics.h"
#include "help_functions.cu"

#include <curand_kernel.h>

__global__ void SetupRandKernel(curandState *states, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= N)
		return;

    /* Each thread gets same seed, a different sequence number, no offset */
    curand_init(1234, idx, 0, &states[idx]);
}

__global__ void DoCalculationsGPU(float* Means, const float* __restrict__ Data, const curandState *states, int NBOOTSTRAPS, int NSAMPLES)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= NBOOTSTRAPS)
		return;

    curandState localState = states[idx];

	float Nf = (float)NSAMPLES - 1.0f;	
	float sum = 0.0f;
	for (int i = 0; i < NSAMPLES; i++)
	{
	    int randomIndex = (int)(curand_uniform(&localState) * Nf);
        sum += Data[randomIndex];
	}
	Means[idx] = sum / (float)NSAMPLES;
}

#endif