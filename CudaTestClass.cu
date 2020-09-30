#include "CudaTestClass.cuh"
__global__ void createArrays(const int num, double *array0, double *array1){
    //initialize arrays
    int index = blockIdx.x * blockDim.x + threadIdx.x, stride = blockDim.x * gridDim.x;

    for(int i = index; i < num; i+=stride){
        array0[i] = 65.2314645;
        array1[i] = 56.5376367;
    }
}
__global__ void multiplyArrays(const int num, double *array0, double *array1){
    //matrix multiplication - result to array0
    int index = blockIdx.x * blockDim.x + threadIdx.x, stride = blockDim.x * gridDim.x;

    for(int i = index; i < num; i += stride){
        array0[i] = array0[i]*array1[i];
    }
}

int main(void){
    std::cout << "Running..." << std::endl;
    const int num = 25000000;
    double *array0, *array1;

    cudaMallocManaged(&array0, sizeof(double)*num);
    cudaMallocManaged(&array1, sizeof(double)*num);

    //ignore errors from Intellisense / launch params
    #ifndef __INTELLISENSE__
    createArrays<<<9766,1024>>>(num, array0, array1);
    #endif

    cudaDeviceSynchronize();

    #ifndef __INTELLISENSE__
    multiplyArrays<<<9766,1024>>>(num, array0, array1);
    #endif

    cudaDeviceSynchronize();

    cudaFree(array0);
    cudaFree(array1);
    
    return 0;
}
