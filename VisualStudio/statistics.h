
#include <cuda.h>
#include <cuda_runtime.h>



#ifndef STATISTICS_H_
#define STATISTICS_H_

class Statistics
{

public:

	Statistics(int samples, int bootstraps);	
	~Statistics();

	void SetInputDataPointers(float*);
	void SetOutputDataPointers(float*);
	
	double BootstrapMean();
	double BootstrapMeanCublas();
    
private:

    int NSAMPLES, NBOOTSTRAPS;

	int threadsInX, threadsInY, threadsInZ;
    int blocksInX, blocksInY, blocksInZ;
	dim3 dimGrid, dimBlock;

	double	processingTime;

	float* h_xtxxt;
	float* h_Contrasts;
	float* h_Residuals;
	float* h_Fit;
	float* h_Means;
	float* h_Data;

	float* d_xtxxt;
	float* d_Contrasts;
	float* d_Residuals;
	float* d_Fit;
	float* d_Means;
	float* d_Data;
	float* d_RandomSamples;
	float* d_MeanVector;
	
};

#endif 