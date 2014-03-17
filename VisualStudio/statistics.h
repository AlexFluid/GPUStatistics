
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

	double DoCalculations();
    
private:

    int NSAMPLES, NBOOTSTRAPS;

	int threadsInX, threadsInY, threadsInZ;
    int blocksInX, blocksInY, blocksInZ;
	dim3 dimGrid, dimBlock;

	double	processingTime;

	float* h_Means;
	float* h_Data;

	float* d_Means;
	float* d_Data;
	
};

#endif 