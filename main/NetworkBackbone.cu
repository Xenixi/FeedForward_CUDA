//#include "Network.cuh"
#include "cuda.h"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <iostream>
#include <math.h>
#include "NetworkBackbone.cuh"

__global__ void trainNetwork(float *inputs, float *targets, int iNodes, int hNodes, int oNodes)
{
}

__global__ void queryNetwork(float *inputs)
{
}

NetworkBackbone::NodeParams np;
NetworkBackbone::NetworkBackbone(int iNodes, int oNodes, int hNodes)
{
    np.iNodes = iNodes;
    np.oNodes = oNodes;
    np.hNodes = hNodes;
}


void NetworkBackbone::train(float *inputs, float *targets)
{
    //allocate shared memory GPU
    //Maybe move ALL values to GPU ahead of time?
    cudaMallocManaged(&inputs, sizeof(float) * getInputQuantity());
    cudaMallocManaged(&targets, sizeof(float) * getOutputQuantity());

    //kernel call
    trainNetwork<<<8192, 1024>>>(inputs, targets, getInputQuantity(), getHiddenQuantity(), getOutputQuantity());

    //free

    cudaFree(inputs);
    cudaFree(targets);
}
void NetworkBackbone::query(float *inputs)
{
}
int NetworkBackbone::getInputQuantity()
{
    return np.iNodes;
}
int NetworkBackbone::getOutputQuantity()
{
    return np.oNodes;
}
int NetworkBackbone::getHiddenQuantity()
{
    return np.hNodes;
}
