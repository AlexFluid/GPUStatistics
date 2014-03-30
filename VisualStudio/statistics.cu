
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "statistics.h"
#include "statistics_kernel.cu"

#include <cuda.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cublas_v2.h>

Statistics::Statistics(int samples, int bootstraps)
{
    NSAMPLES = samples;
	NBOOTSTRAPS = bootstraps;
}

Statistics::Statistics(int samples, int bootstraps, int regressors)
{
    NSAMPLES = samples;
	NBOOTSTRAPS = bootstraps;
	NUMBER_OF_REGRESSORS = regressors;
}

Statistics::~Statistics()
{

}

void Statistics::SetInputDataPointers(float* data)
{
	h_Data = data;
}

void Statistics::SetInputDataPointers(float* fit, float* residuals, float* xtxxt)
{
	h_Fit = fit;
	h_Residuals = residuals;
	h_xtxxt = xtxxt;
}

void Statistics::SetOutputDataPointers(float* means)
{
	h_Means = means;
}

void Statistics::SetOutputDataPointersContrasts(float* contrasts)
{
	h_Contrasts = contrasts;
}


double Statistics::BootstrapMean()
{        
	dim3 dimGrid, dimBlock;

	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);
	
	checkCudaErrors(cudaSetDevice(0));

    // Allocate memory on GPU    
	
	sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
	
	cudaMalloc((void **)&d_Means,  NBOOTSTRAPS * sizeof(float));
    cudaMalloc((void **)&d_Data,  NSAMPLES * sizeof(float));
	
	// Copy data to GPU
	cudaMemcpy(d_Data, h_Data, NSAMPLES * sizeof(float), cudaMemcpyHostToDevice);
	
    // 256 threads per block
	threadsInX = 256;

    // Round up to get sufficient number of blocks
    blocksInX = (int)ceil((float)NBOOTSTRAPS / (float)threadsInX);
    
    dimGrid  = dim3(blocksInX, 1, 1);
    dimBlock = dim3(threadsInX, 1, 1);
	
    // Do calculations	
	BootstrapMeanGPU<<<dimGrid, dimBlock>>>(d_Means, d_Data, NBOOTSTRAPS, NSAMPLES);
	
	// Copy result to host
	cudaMemcpy(h_Means, d_Means, NBOOTSTRAPS * sizeof(float), cudaMemcpyDeviceToHost);

	checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    double gpuTime = 0.001 * sdkGetTimerValue(&hTimer);
    	
    // Free allocated memory on GPU
	cudaFree( d_Means );
	cudaFree( d_Data );
    
	sdkDeleteTimer(&hTimer);
	
    cudaDeviceReset();
	return gpuTime;
}

double Statistics::BootstrapMeanCublas()
{        
	cublasStatus_t status;
	cublasHandle_t handle;
	status = cublasCreate(&handle);
		
	dim3 dimGrid, dimBlock;

	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);
	
	//checkCudaErrors(cudaSetDevice(0));

    // Allocate memory on GPU    
	
	sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
	
	cudaMalloc((void **)&d_Means,  NBOOTSTRAPS * sizeof(float));
    cudaMalloc((void **)&d_Data,   NSAMPLES * sizeof(float));
	cudaMalloc((void **)&d_RandomSamples, NBOOTSTRAPS * NSAMPLES * sizeof(float));
	cudaMalloc((void **)&d_MeanVector, NSAMPLES * sizeof(float));

	cudaMemset(d_Means, 0, NBOOTSTRAPS * sizeof(float));

	// Copy data to GPU
	cudaMemcpy(d_Data, h_Data, NSAMPLES * sizeof(float), cudaMemcpyHostToDevice);
	
	
	// 256 threads per block
	threadsInX = 256;
	
    // Round up to get sufficient number of blocks
    blocksInX = (int)ceil((float)NSAMPLES / (float)threadsInX);
	
    dimGrid  = dim3(blocksInX, 1, 1);
    dimBlock = dim3(threadsInX, 1, 1);
	
	// Set mean vector
	SetMeanVector<<<dimGrid, dimBlock>>>(d_MeanVector, NSAMPLES);

	
	// 1024 threads per block
	threadsInX = 256;
	
    // Round up to get sufficient number of blocks
    blocksInX = (int)ceil((float)NBOOTSTRAPS / (float)threadsInX);
	
    dimGrid  = dim3(blocksInX, 1, 1);
    dimBlock = dim3(threadsInX, 1, 1);
				
    // Generate random samples
	GenerateRandomSamples<<<dimGrid, dimBlock>>>(d_RandomSamples, d_Data, NBOOTSTRAPS, NSAMPLES);
					
	// Perform matrix-vector multiplication, to calculate means
	float alpha = 1.0f;
	float beta = 0.0f;
	int stride = 1;
	int leadingDimension = NBOOTSTRAPS;
	status = cublasSgemv(handle, CUBLAS_OP_N, NBOOTSTRAPS, NSAMPLES, &alpha, d_RandomSamples, leadingDimension, d_MeanVector, stride, &beta, d_Means, stride);
	
	

	
	// Copy result to host
	cudaMemcpy(h_Means, d_Means, NBOOTSTRAPS * sizeof(float), cudaMemcpyDeviceToHost);

	
	

	checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    double gpuTime = 0.001 * sdkGetTimerValue(&hTimer);
    
	
		
    // Free allocated memory on GPU
	cudaFree( d_Means );
	cudaFree( d_MeanVector );
	cudaFree( d_Data );
	cudaFree( d_RandomSamples );
    
	sdkDeleteTimer(&hTimer);
	
	
	status = cublasDestroy(handle);

    //cudaDeviceReset();
		
	return gpuTime;
}
