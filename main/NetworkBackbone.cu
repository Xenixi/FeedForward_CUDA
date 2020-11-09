//#include "Network.cuh"
#include <iostream>
#include <math.h>
#include "NetworkBackbone.cuh"
//runs on GPU
__global__ void initWeights(float *weightsInputHidden, float *weightsHiddenOutput, int iNodes, int hNodes, int oNodes){
    int idx = blockDim.x * blockIdx.x + threadIdx.x, stride = blockDim.x * gridDim.x;
    
    for(int i = idx; i < hNodes*iNodes; i+=stride){

    } 
    for(int i = idx; i < hNodes*oNodes; i+=stride){

    }
}

__global__ void trainNetwork(float *inputs, float *targets, int iNodes, int hNodes, int oNodes)
{


}

__global__ void queryNetwork(float *inputs)
{
}

__device__ float activation(float input)
{
    return input / (abs(input) + 1);
}

/////////////////////
int *NetworkBackbone::getDeviceProperties()
{
    Utils u;
    int *props = new int[3];
    props[0] = u.getSMs();
    props[1] = u.getTB();
    props[2] = u.getTMP();

    return props;
}
int NetworkBackbone::Utils::getSMs()
{
    return getDeviceProps().multiProcessorCount;
}

int NetworkBackbone::Utils::getTB()
{
    return getDeviceProps().maxThreadsPerBlock;
}
int NetworkBackbone::Utils::getTMP()
{
    return getDeviceProps().maxThreadsPerMultiProcessor;
}

cudaDeviceProp NetworkBackbone::Utils::getDeviceProps()
{
    int device;
    cudaGetDevice(&device);
    struct cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);

    return properties;
}
//CPU
//continued
NetworkBackbone::NodeParams np;
NetworkBackbone::NetworkBackbone(int iNodes, int oNodes, int hNodes)
{
    np.iNodes = iNodes;
    np.oNodes = oNodes;
    np.hNodes = hNodes;

    //initialization


}

void NetworkBackbone::train(float *inputs, float *targets)
{
    //allocate shared memory GPU
    //Maybe move ALL values to GPU ahead of time?
    cudaMallocManaged(&inputs, sizeof(float) * getInputQuantity());
    cudaMallocManaged(&targets, sizeof(float) * getOutputQuantity());

    //kernel call
    int blocks = (getDeviceProperties()[0] * getDeviceProperties()[2] / getDeviceProperties()[1]);
    int blockThreads = getDeviceProperties()[1];

    trainNetwork<<<blocks, blockThreads>>>(inputs, targets, getInputQuantity(), getHiddenQuantity(), getOutputQuantity());

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
