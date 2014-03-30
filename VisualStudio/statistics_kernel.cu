
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


__global__ void SetMeanVector(float* MeanVector, int NSAMPLES)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (idx >= NSAMPLES)
		return;
	
	MeanVector[idx] = 1.0f/(float)NSAMPLES;
}


__global__ void BootstrapMeanGPU(float* Means, const float* __restrict__ Data, int NBOOTSTRAPS, int NSAMPLES)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= NBOOTSTRAPS)
		return;

	curandState localState;
	curand_init(1234, idx, 0, &localState);

	float Nf = (float)NSAMPLES - 1.0f;	
	float sum = 0.0f;
	for (int i = 0; i < NSAMPLES; i++)
	{
	    int randomIndex = (int)(curand_uniform(&localState) * Nf);
        sum += Data[randomIndex];
	}
	Means[idx] = sum / (float)NSAMPLES;
}

__global__ void GenerateRandomSamples(float* RandomSamples, const float* __restrict__ Data, int NBOOTSTRAPS, int NSAMPLES)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	//if (idx >= NSAMPLES)
	if (idx >= NBOOTSTRAPS)
		return;
	
    curandState localState;
	curand_init(1234, idx, 0, &localState);

	float Nf = (float)NSAMPLES - 1.0f;	
	//for (int i = 0; i < NBOOTSTRAPS; i++)
	for (int i = 0; i < NSAMPLES; i++)
	{	 
		int randomIndex = (int)(curand_uniform(&localState) * Nf);		
		RandomSamples[idx + i*NBOOTSTRAPS] = Data[randomIndex];		
    }
}


#endif