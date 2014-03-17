
void pack_double2float(float* output_float, double* input_double, int N)
{
    for (int i = 0; i < N; i++)
    {
        output_float[i] = (float)input_double[i];
    }
}

void unpack_float2double(double* output_double, float* input_float, int N)
{    
    for (int i = 0; i < N; i++)
    {
        output_double[i] = (double)input_float[i];
    }
}

