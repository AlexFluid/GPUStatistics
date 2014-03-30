#include <mex.h>
#include "help_functions.cpp"
#include "statistics.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{	
    
    // Input pointers
    
    double				    *h_Data_double;		
	float					*h_Data;		
    
	//-----------------------
	// Output pointers
    
    double                  *h_Means_double;
    float                   *h_Means;
	
    //---------------------

    /* Check the number of input and output arguments. */
    if(nrhs<2)
    {
        mexErrMsgTxt("Too few input arguments.");
    }
    if(nrhs>2)
    {
        mexErrMsgTxt("Too many input arguments.");
    }

    if(nlhs<2)
    {
        mexErrMsgTxt("Too few output arguments.");
    }
    if(nlhs>2) 
    {
        mexErrMsgTxt("Too many output arguments.");
    }
   
    /* Input arguments */    

    // The data
    h_Data_double =  (double*)mxGetData(prhs[0]);
    int NBOOTSTRAPS = (int)mxGetScalar(prhs[1]);

	const int *ARRAY_DIMENSIONS_DATA = mxGetDimensions(prhs[0]);
	const int NSAMPLES = ARRAY_DIMENSIONS_DATA[0];
	
	// Data sizes
	mexPrintf("Number of bootstraps %i, number of samples %i \n", NBOOTSTRAPS, NSAMPLES);
	mexEvalString("drawnow;");

	//-------------------------------------------------
	// Output to Matlab

    int NUMBER_OF_DIMENSIONS = 2;
	int ARRAY_DIMENSIONS_OUT[2];
	ARRAY_DIMENSIONS_OUT[0] = NBOOTSTRAPS;
	ARRAY_DIMENSIONS_OUT[1] = 1;
	plhs[0] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT,mxDOUBLE_CLASS, mxREAL);
 	h_Means_double = mxGetPr(plhs[0]); 
	
    NUMBER_OF_DIMENSIONS = 2;
	ARRAY_DIMENSIONS_OUT[0] = 1;	
    ARRAY_DIMENSIONS_OUT[1] = 1;	
	plhs[1] = mxCreateNumericArray(NUMBER_OF_DIMENSIONS,ARRAY_DIMENSIONS_OUT,mxDOUBLE_CLASS, mxREAL);
 	double *time = mxGetPr(plhs[1]); 
	
	// ------------------------------------------------
 
	// Allocate memory on the host
    h_Means	           = (float *)mxMalloc(NBOOTSTRAPS * sizeof(float));
	h_Data  	   	   = (float *)mxMalloc(NSAMPLES * sizeof(float));
	
    // Reorder and cast data from doubles to floats
    pack_double2float(h_Data, h_Data_double, NSAMPLES);
	
	//----------
    // Do the calculations on the GPU
    
    Statistics my_calculator(NSAMPLES, NBOOTSTRAPS);

    my_calculator.SetInputDataPointers(h_Data);
    my_calculator.SetOutputDataPointers(h_Means);
    double gputime = my_calculator.BootstrapMeanCublas();
    *time = gputime*1000;
    
    mexPrintf("Calculations took %f ms \n",gputime*1000);
    
	//----------

    // Reorder and cast data from floats to doubles
    unpack_float2double(h_Means_double, h_Means, NBOOTSTRAPS);
    
	// Free all the allocated memory on the host
	mxFree(h_Means);
	mxFree(h_Data);
        
    //cudaThreadExit();
    //CUT_THREADEND;

 	return;
}
