
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "statistics.h"
#include "statistics_kernel.cu"

#include <cuda.h>
#include <helper_functions.h>
#include <helper_cuda.h>

Statistics::Statistics(int samples, int bootstraps)
{
    NSAMPLES = samples;
	NBOOTSTRAPS = bootstraps;
}

Statistics::~Statistics()
{

}

void Statistics::SetInputDataPointers(float* data)
{
	h_Data = data;
}

void Statistics::SetOutputDataPointers(float* means)
{
	h_Means = means;
}



double Statistics::DoCalculations()
{        
	dim3 dimGrid, dimBlock;

	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);
	curandState *devStates;

	checkCudaErrors(cudaSetDevice(0));

    // Allocate memory on GPU    
	
	sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
	
	cudaMalloc((void **)&d_Means,  NBOOTSTRAPS * sizeof(float));
    cudaMalloc((void **)&d_Data,  NSAMPLES * sizeof(float));
	cudaMalloc((void **)&devStates, NBOOTSTRAPS * sizeof(curandState));
	
	// Copy data to GPU
	cudaMemcpy(d_Data, h_Data, NSAMPLES * sizeof(float), cudaMemcpyHostToDevice);

	// Copy meanprop to constant memory
	//cudaMemcpyToSymbol(c_MeanProp, h_MeanProp, NPar * sizeof(float), 0, cudaMemcpyHostToDevice);
	
    // 512 threads per block
	threadsInX = 512;

    // Round up to get sufficient number of blocks
    blocksInX = (int)ceil((float)NBOOTSTRAPS / (float)threadsInX);
    
    dimGrid  = dim3(blocksInX, 1, 1);
    dimBlock = dim3(threadsInX, 1, 1);

	
	// Initialize states for random number generation
	SetupRandKernel<<<dimGrid, dimBlock>>>(devStates,NBOOTSTRAPS);

    // Do calculations	
	DoCalculationsGPU<<<dimGrid, dimBlock>>>(d_Means, d_Data, devStates, NBOOTSTRAPS, NSAMPLES);
	
	// Copy result to host
	cudaMemcpy(h_Means, d_Means, NBOOTSTRAPS * sizeof(float), cudaMemcpyDeviceToHost);

	checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    double gpuTime = 0.001 * sdkGetTimerValue(&hTimer);
    	
    // Free allocated memory on GPU
	cudaFree( d_Means );
	cudaFree( d_Data );
    cudaFree( devStates );

	sdkDeleteTimer(&hTimer);
	
    cudaDeviceReset();
	return gpuTime;
}

